use crate::core::error::{SoundError, SoundResult};
use crate::gpu::backend::VkDeviceContext;
use crate::gpu::compute::{BufferBarrierSpec, ComputeContext};
use ash::vk;
use bytemuck::Pod;
use std::sync::{Arc, Mutex};
use vk_mem::{
    Alloc, Allocation, AllocationCreateFlags, Allocator, AllocatorCreateFlags, MemoryUsage,
};

const STAGING_BUFFER_INITIAL_SIZE: vk::DeviceSize = 16 * 1024 * 1024; // 16 MB

/// Allocates Vulkan buffers and provides synchronous staging transfers.
///
/// The allocator is configured for buffer device addresses so buffers can be
/// used by acceleration-structure builds and shader binding tables.
pub struct GpuAllocator {
    allocator: Arc<AllocatorOwner>,
    device_context: Arc<VkDeviceContext>,
    compute: Arc<ComputeContext>,
    transfer: Mutex<TransferContext>,
}

/// Owns a VMA buffer, its allocator, and the device/loader required for destruction.
/// Submitted work must finish before the buffer is released or mapped on the host.
pub struct BufferHandle {
    info: BufferInfo,
    allocation: Allocation,
    allocator: Arc<AllocatorOwner>,
    mapped_access: Mutex<()>,
    usage: vk::BufferUsageFlags,
}

/// Read-only allocation metadata exposed by `BufferHandle` through `Deref`.
/// Mutating a separate copy cannot alter the owner's bounds or destruction handle.
pub struct BufferInfo {
    /// Raw Vulkan buffer handle used in descriptors and command recording.
    pub buffer: vk::Buffer,
    /// Buffer size in bytes.
    pub size: vk::DeviceSize,
}

impl std::ops::Deref for BufferHandle {
    type Target = BufferInfo;

    fn deref(&self) -> &BufferInfo {
        &self.info
    }
}

// VMA copies raw Vulkan handles; its Arc alone does not retain their owners.
// Declaration order keeps the device alive through vmaDestroyAllocator.
struct AllocatorOwner {
    allocator: Allocator,
    _device: Arc<VkDeviceContext>,
}

impl std::ops::Deref for AllocatorOwner {
    type Target = Allocator;

    fn deref(&self) -> &Allocator {
        &self.allocator
    }
}

struct TransferContext {
    staging_buffer: BufferHandle,
}

impl GpuAllocator {
    pub(super) fn new(
        device_context: Arc<VkDeviceContext>,
        compute: Arc<ComputeContext>,
    ) -> SoundResult<Self> {
        let mut allocator_info = vk_mem::AllocatorCreateInfo::new(
            &device_context.instance,
            &device_context.device,
            device_context.physical_device,
        );
        allocator_info.flags |= AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;
        allocator_info.vulkan_api_version = vk::API_VERSION_1_3;
        let allocator = unsafe {
            Arc::new(AllocatorOwner {
                allocator: Allocator::new(allocator_info).map_err(|e| {
                    SoundError::resource_allocation_failed(format!(
                        "GPU allocator creation failed: {e:?}"
                    ))
                })?,
                _device: device_context.clone(),
            })
        };

        let staging_buffer = Self::create_staging_buffer(&allocator, STAGING_BUFFER_INITIAL_SIZE)?;

        let transfer = TransferContext {
            staging_buffer,
        };

        Ok(Self {
            allocator,
            device_context,
            compute,
            transfer: Mutex::new(transfer),
        })
    }

    /// Creates a device-local storage buffer with transfer source/destination usage.
    pub fn create_storage_buffer(&self, size: u64) -> SoundResult<BufferHandle> {
        self.create_buffer(
            size as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryUsage::AutoPreferDevice,
            AllocationCreateFlags::empty(),
        )
    }

    /// Creates a device-local buffer that can be addressed by shaders or AS builds.
    ///
    /// `usage` should describe the caller-specific role; shader-device-address
    /// and transfer-destination usage are added automatically.
    /// Usages requiring extensions disabled by this backend are rejected.
    pub fn create_device_address_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> SoundResult<BufferHandle> {
        self.create_buffer(
            size,
            usage
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryUsage::AutoPreferDevice,
            AllocationCreateFlags::empty(),
        )
    }

    /// Creates a mapped host-visible buffer that also has a device address.
    ///
    /// This is useful for small RT inputs such as instance data or SBT records.
    /// Usages requiring extensions disabled by this backend are rejected.
    pub fn create_host_device_address_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> SoundResult<BufferHandle> {
        self.create_buffer(
            size,
            usage
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryUsage::AutoPreferHost,
            AllocationCreateFlags::MAPPED | AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
        )
    }

    /// Returns the Vulkan device address for a buffer created with address usage.
    /// Panics if the buffer belongs to another allocator or lacks address usage.
    pub fn buffer_device_address(&self, buffer: &BufferHandle) -> vk::DeviceAddress {
        self.validate_owner(buffer)
            .expect("Device address buffer belongs to another allocator");
        assert!(
            buffer
                .usage
                .contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS),
            "Buffer lacks SHADER_DEVICE_ADDRESS usage"
        );
        let info = vk::BufferDeviceAddressInfo::default().buffer(buffer.buffer);
        unsafe { self.device_context.device.get_buffer_device_address(&info) }
    }

    /// Writes bytes directly into a mapped buffer and flushes the written range.
    pub fn write_mapped_bytes(&self, destination: &BufferHandle, data: &[u8]) -> SoundResult<()> {
        self.validate_owner(destination)?;
        let _access = destination
            .mapped_access
            .lock()
            .map_err(|_| SoundError::poisoned_lock("Mapped buffer lock is poisoned"))?;
        if data.len() as vk::DeviceSize > destination.size {
            return Err(SoundError::invalid_argument(format!(
                "Mapped write size {} exceeds destination buffer size {}",
                data.len(),
                destination.size
            )));
        }

        let mapped_ptr = self
            .allocator
            .get_allocation_info(&destination.allocation)
            .mapped_data;
        if mapped_ptr.is_null() {
            return Err(SoundError::invalid_state(
                "Destination buffer is not mapped",
            ));
        }

        // Mapped VMA allocations expose host memory; flush makes writes visible to the GPU.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), mapped_ptr as *mut u8, data.len());
        }
        self.allocator
            .flush_allocation(&destination.allocation, 0, data.len() as u64)?;
        Ok(())
    }

    /// Reads bytes from a mapped buffer after invalidating the requested range.
    pub fn read_mapped_bytes(
        &self,
        source: &BufferHandle,
        size: usize,
        out: &mut [u8],
    ) -> SoundResult<()> {
        self.validate_owner(source)?;
        let _access = source
            .mapped_access
            .lock()
            .map_err(|_| SoundError::poisoned_lock("Mapped buffer lock is poisoned"))?;
        if size as vk::DeviceSize > source.size {
            return Err(SoundError::invalid_argument(format!(
                "Mapped read size {} exceeds source buffer size {}",
                size, source.size
            )));
        }
        if out.len() < size {
            return Err(SoundError::invalid_argument(format!(
                "Mapped read output slice too small: out={}, requested={size}",
                out.len()
            )));
        }

        let mapped_ptr = self
            .allocator
            .get_allocation_info(&source.allocation)
            .mapped_data;
        if mapped_ptr.is_null() {
            return Err(SoundError::invalid_state("Source buffer is not mapped"));
        }

        // Invalidate before reading so host memory observes GPU writes.
        self.allocator
            .invalidate_allocation(&source.allocation, 0, size as u64)?;
        unsafe {
            std::ptr::copy_nonoverlapping(mapped_ptr as *const u8, out.as_mut_ptr(), size);
        }
        Ok(())
    }

    /// Creates a mapped host-visible buffer intended for GPU-to-CPU readback.
    pub fn create_readback_buffer(&self, size: vk::DeviceSize) -> SoundResult<BufferHandle> {
        self.create_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryUsage::AutoPreferHost,
            AllocationCreateFlags::MAPPED | AllocationCreateFlags::HOST_ACCESS_RANDOM,
        )
    }

    /// Consumes a buffer and releases its allocation exactly once through `Drop`.
    ///
    /// Callers must ensure no queued GPU work still uses the buffer.
    pub fn destroy_buffer(&self, handle: BufferHandle) {
        drop(handle);
    }

    /// Fills the whole buffer with zero using the compute queue transfer path.
    pub fn clear_buffer(&self, buffer: &BufferHandle) -> SoundResult<()> {
        self.compute
            .submit_compute_and_wait(|command_buffer| unsafe {
                self.record_clear_buffer(command_buffer, buffer)
            })
    }

    /// Records a checked fill so callers can batch initialization with dependent GPU work.
    ///
    /// # Safety
    /// The command buffer must be recording on this device. Retain the buffer
    /// through completion and add a transfer-write dependency before consuming it.
    pub(crate) unsafe fn record_clear_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        buffer: &BufferHandle,
    ) -> SoundResult<()> {
        self.validate_owner(buffer)?;
        if !buffer.usage.contains(vk::BufferUsageFlags::TRANSFER_DST) || buffer.size % 4 != 0 {
            return Err(SoundError::invalid_argument(
                "Buffer fill requires TRANSFER_DST and a size divisible by four",
            ));
        }
        unsafe {
            self.device_context.device.cmd_fill_buffer(
                command_buffer,
                buffer.buffer,
                0,
                buffer.size,
                0,
            );
        }
        Ok(())
    }

    /// Uploads bytes through the shared staging buffer and waits for completion.
    pub fn upload_bytes(&self, destination: &BufferHandle, data: &[u8]) -> SoundResult<()> {
        self.validate_owner(destination)?;
        if data.is_empty() {
            return Ok(());
        }
        let copy_size = data.len() as u64;
        if copy_size > destination.size {
            return Err(SoundError::invalid_argument(format!(
                "Upload size {} exceeds destination buffer size {}",
                data.len(),
                destination.size
            )));
        }

        let mut transfer = self
            .transfer
            .lock()
            .map_err(|_| SoundError::poisoned_lock("Transfer lock is poisoned"))?;

        // Grow staging lazily so large scene uploads do not require a fixed cap.
        self.ensure_staging_capacity(&mut transfer, data.len() as u64)?;
        self.upload_to_staging(&transfer, data)?;
        self.copy_buffer(&transfer.staging_buffer, destination, copy_size, None)?;
        Ok(())
    }

    /// Uploads a POD slice by viewing it as bytes.
    pub fn upload_typed<T: Pod>(&self, destination: &BufferHandle, data: &[T]) -> SoundResult<()> {
        self.upload_bytes(destination, bytemuck::cast_slice(data))
    }

    /// Downloads bytes through the shared staging buffer and waits for completion.
    pub fn download_bytes(
        &self,
        src: &BufferHandle,
        size: usize,
        out: &mut [u8],
    ) -> SoundResult<()> {
        self.validate_owner(src)?;
        if size == 0 {
            return Ok(());
        }
        let copy_size = size as u64;
        if copy_size > src.size {
            return Err(SoundError::invalid_argument(format!(
                "Download size {} exceeds source buffer size {}",
                size, src.size
            )));
        }

        if out.len() < size {
            return Err(SoundError::invalid_argument(format!(
                "Output slice too small for download: out={}, requested={size}",
                out.len()
            )));
        }

        let mut transfer = self
            .transfer
            .lock()
            .map_err(|_| SoundError::poisoned_lock("Transfer lock is poisoned"))?;

        self.ensure_staging_capacity(&mut transfer, copy_size)?;
        self.copy_buffer(
            src,
            &transfer.staging_buffer,
            copy_size,
            Some(BufferBarrierSpec::transfer_write_to_host_read(
                transfer.staging_buffer.buffer,
            )),
        )?;
        self.download_from_staging(&transfer, size, out)?;

        Ok(())
    }

    /// Downloads `count` POD elements into `out`.
    pub fn download_typed<T: Pod>(
        &self,
        src: &BufferHandle,
        count: usize,
        out: &mut [T],
    ) -> SoundResult<()> {
        if count > out.len() {
            return Err(SoundError::invalid_argument(format!(
                "download_typed count {} exceeds output len {}",
                count,
                out.len()
            )));
        }

        let byte_size = count
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| SoundError::invalid_argument("download_typed byte size overflow"))?;

        self.download_bytes(src, byte_size, bytemuck::cast_slice_mut::<T, u8>(out))?;

        return Ok(());
    }

    fn create_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_usage: MemoryUsage,
        allocation_flags: AllocationCreateFlags,
    ) -> SoundResult<BufferHandle> {
        validate_buffer_usage(usage)?;
        if size == 0 {
            return Err(SoundError::invalid_argument(
                "Vulkan buffers must have non-zero size",
            ));
        }
        if usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER)
            && size > self.device_context.limits.max_storage_buffer_range as u64
        {
            return Err(SoundError::invalid_argument(
                "Storage buffer exceeds maxStorageBufferRange",
            ));
        }
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let allocation_info = vk_mem::AllocationCreateInfo {
            usage: memory_usage,
            flags: allocation_flags,
            ..Default::default()
        };

        let (buffer, allocation) = unsafe {
            self.allocator
                .create_buffer(&buffer_info, &allocation_info)
                .map_err(|e| {
                    SoundError::resource_allocation_failed(format!(
                        "Buffer allocation failed: {e:?}"
                    ))
                })?
        };
        Ok(BufferHandle {
            info: BufferInfo {
                buffer,
                size,
            },
            allocation,
            allocator: self.allocator.clone(),
            mapped_access: Mutex::new(()),
            usage,
        })
    }

    fn create_staging_buffer(
        allocator: &Arc<AllocatorOwner>,
        size: vk::DeviceSize,
    ) -> SoundResult<BufferHandle> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let allocation_info = vk_mem::AllocationCreateInfo {
            usage: MemoryUsage::AutoPreferHost,
            flags: AllocationCreateFlags::MAPPED | AllocationCreateFlags::HOST_ACCESS_RANDOM,
            ..Default::default()
        };

        let (buffer, allocation) = unsafe {
            allocator
                .create_buffer(&buffer_info, &allocation_info)
                .map_err(|e| {
                    SoundError::resource_allocation_failed(format!(
                        "Staging buffer allocation failed: {e:?}"
                    ))
                })?
        };
        Ok(BufferHandle {
            info: BufferInfo {
                buffer,
                size,
            },
            allocation,
            allocator: allocator.clone(),
            mapped_access: Mutex::new(()),
            usage: vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
        })
    }

    fn ensure_staging_capacity(
        &self,
        transfer: &mut TransferContext,
        required_size: vk::DeviceSize,
    ) -> SoundResult<()> {
        if transfer.staging_buffer.size >= required_size {
            return Ok(());
        }

        // Preserve amortized growth while still handling a single very large upload.
        let new_size = std::cmp::max(
            transfer.staging_buffer.size.saturating_mul(2),
            required_size,
        );
        let old = std::mem::replace(
            &mut transfer.staging_buffer,
            Self::create_staging_buffer(&self.allocator, new_size)?,
        );

        // BufferHandle owns the VMA allocation. An explicit VMA free followed
        // by its destructor would free it twice when the staging buffer grows.
        drop(old);

        Ok(())
    }

    fn upload_to_staging(&self, transfer: &TransferContext, data: &[u8]) -> SoundResult<()> {
        let mapped_ptr = self
            .allocator
            .get_allocation_info(&transfer.staging_buffer.allocation)
            .mapped_data;

        if mapped_ptr.is_null() {
            return Err(SoundError::invalid_state("Staging buffer is not mapped"));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), mapped_ptr as *mut u8, data.len());
        }

        self.allocator.flush_allocation(
            &transfer.staging_buffer.allocation,
            0,
            data.len() as u64,
        )?;

        Ok(())
    }

    fn download_from_staging(
        &self,
        transfer: &TransferContext,
        size: usize,
        out: &mut [u8],
    ) -> SoundResult<()> {
        let mapped_ptr = self
            .allocator
            .get_allocation_info(&transfer.staging_buffer.allocation)
            .mapped_data;

        if mapped_ptr.is_null() {
            return Err(SoundError::invalid_state("Staging buffer is not mapped"));
        }

        self.allocator.invalidate_allocation(
            &transfer.staging_buffer.allocation,
            0,
            size as u64,
        )?;

        unsafe {
            std::ptr::copy_nonoverlapping(mapped_ptr as *const u8, out.as_mut_ptr(), size);
        }
        Ok(())
    }

    /// Copies bytes between GPU buffers and optionally records a barrier after the transfer.
    ///
    /// The copy is submitted synchronously on the compute queue. Callers provide
    /// a barrier when later GPU or host work must observe the copied bytes.
    pub(crate) fn copy_buffer(
        &self,
        src: &BufferHandle,
        dst: &BufferHandle,
        size: vk::DeviceSize,
        barrier: Option<BufferBarrierSpec>,
    ) -> SoundResult<()> {
        if size == 0 {
            self.validate_owner(src)?;
            return self.validate_owner(dst);
        }
        self.compute
            .submit_compute_and_wait(|command_buffer| unsafe {
                self.record_copy_buffer(command_buffer, src, dst, size, barrier)
            })
    }

    /// Records a checked prefix copy and an optional dependency for its consumer.
    ///
    /// # Safety
    /// The command buffer must be recording on this device. Retain both buffers
    /// through completion and synchronize earlier writes before this copy. Any
    /// supplied barrier must describe a valid dependency on this queue.
    pub(crate) unsafe fn record_copy_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        src: &BufferHandle,
        dst: &BufferHandle,
        size: vk::DeviceSize,
        barrier: Option<BufferBarrierSpec>,
    ) -> SoundResult<()> {
        self.validate_owner(src)?;
        self.validate_owner(dst)?;
        if size == 0 {
            return Ok(());
        }
        if src.buffer == dst.buffer
            || !src.usage.contains(vk::BufferUsageFlags::TRANSFER_SRC)
            || !dst.usage.contains(vk::BufferUsageFlags::TRANSFER_DST)
        {
            return Err(SoundError::invalid_argument(
                "Buffer copy requires distinct buffers with transfer usage",
            ));
        }
        if size > src.size {
            return Err(SoundError::invalid_argument(format!(
                "Copy size {} exceeds source buffer size {}",
                size, src.size
            )));
        }

        if size > dst.size {
            return Err(SoundError::invalid_argument(format!(
                "Copy size {} exceeds destination buffer size {}",
                size, dst.size
            )));
        }

        unsafe {
            let region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size,
            };

            self.device_context.device.cmd_copy_buffer(
                command_buffer,
                src.buffer,
                dst.buffer,
                std::slice::from_ref(&region),
            );

            if let Some(barrier) = barrier {
                self.compute.record_buffer_barrier(command_buffer, &barrier);
            }
        }
        Ok(())
    }

    /// Rejects cross-device buffers before calling VMA or Vulkan with their handles.
    pub(crate) fn validate_owner(&self, buffer: &BufferHandle) -> SoundResult<()> {
        if !Arc::ptr_eq(&self.allocator, &buffer.allocator) {
            return Err(SoundError::invalid_argument(
                "Buffer belongs to a different GPU allocator",
            ));
        }
        Ok(())
    }

    /// Counts live VMA allocations and requested bytes, excluding reserved pool capacity.
    #[cfg(test)]
    pub(crate) fn live_allocation_stats(&self) -> SoundResult<(u32, u64)> {
        let stats = self.allocator.calculate_statistics()?.total.statistics;
        Ok((stats.allocationCount, stats.allocationBytes))
    }
}

// Only usages supported by the features/extensions enabled in VkBackend are legal.
fn validate_buffer_usage(usage: vk::BufferUsageFlags) -> SoundResult<()> {
    let supported = vk::BufferUsageFlags::TRANSFER_SRC
        | vk::BufferUsageFlags::TRANSFER_DST
        | vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER
        | vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER
        | vk::BufferUsageFlags::UNIFORM_BUFFER
        | vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::INDEX_BUFFER
        | vk::BufferUsageFlags::VERTEX_BUFFER
        | vk::BufferUsageFlags::INDIRECT_BUFFER
        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
        | vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR;
    if usage.is_empty() || !(usage & !supported).is_empty() {
        return Err(SoundError::unsupported_feature(
            "Buffer usage requires disabled extensions or contains unknown bits",
        ));
    }
    Ok(())
}

impl Drop for BufferHandle {
    fn drop(&mut self) {
        unsafe {
            self.allocator
                .destroy_buffer(self.info.buffer, &mut self.allocation);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::backend::VkBackend;

    #[test]
    fn buffer_usage_rejects_disabled_extensions_and_unknown_bits() {
        use vk::BufferUsageFlags as Usage;
        assert!(validate_buffer_usage(Usage::STORAGE_BUFFER | Usage::TRANSFER_DST).is_ok());
        assert!(validate_buffer_usage(Usage::ACCELERATION_STRUCTURE_STORAGE_KHR).is_ok());
        for usage in [
            Usage::empty(),
            Usage::VIDEO_DECODE_SRC_KHR,
            Usage::TRANSFORM_FEEDBACK_BUFFER_EXT,
            Usage::from_raw(u32::MAX),
        ] {
            assert!(validate_buffer_usage(usage).is_err());
        }
    }

    #[test]
    #[ignore = "requires Vulkan; run hardware tests serially"]
    fn audit_buffer_retains_device_after_backend_drop() {
        let gpu = VkBackend::new().unwrap();
        let device = Arc::downgrade(&gpu.device_context());
        let buffer = gpu.memory().create_storage_buffer(16).unwrap();
        gpu.memory()
            .upload_typed(&buffer, &[1_u32, 2, 3, 4])
            .unwrap();
        drop(gpu);
        assert!(device.upgrade().is_some());
        drop(buffer);
        assert!(device.upgrade().is_none());
    }

    #[test]
    #[ignore = "requires Vulkan; run hardware tests serially"]
    fn audit_empty_transfers_and_invalid_allocations() {
        let gpu = VkBackend::new().unwrap();
        assert!(gpu.memory().create_storage_buffer(0).is_err());
        assert!(
            gpu.memory()
                .create_device_address_buffer(16, vk::BufferUsageFlags::VIDEO_DECODE_SRC_KHR)
                .is_err()
        );
        let buffer = gpu.memory().create_storage_buffer(5).unwrap();
        assert!(gpu.memory().clear_buffer(&buffer).is_err());
        gpu.memory().upload_bytes(&buffer, &[]).unwrap();
        gpu.memory().download_bytes(&buffer, 0, &mut []).unwrap();
        assert!(gpu.memory().upload_bytes(&buffer, &[0; 6]).is_err());
        let aligned = gpu.memory().create_storage_buffer(4).unwrap();
        gpu.compute()
            .submit_compute_and_wait(|command_buffer| {
                // The local allocation is retained until this synchronous submission finishes.
                unsafe {
                    gpu.memory().record_clear_buffer(command_buffer, &aligned)?;
                    gpu.compute().record_buffer_barrier(
                        command_buffer,
                        &BufferBarrierSpec::transfer_write_to_device_read(aligned.buffer),
                    );
                }
                Ok(())
            })
            .unwrap();
        let mut output = [1_u32];
        gpu.memory()
            .download_typed(&aligned, 1, &mut output)
            .unwrap();
        assert_eq!(output, [0]);
    }

    #[test]
    #[ignore = "requires Vulkan; run hardware tests serially"]
    fn staging_growth_and_explicit_release_preserve_transfers() {
        let gpu = VkBackend::new().unwrap();
        // Cross both the initial capacity and its first doubling. Round-trip
        // every word so a freed/reused allocation cannot pass by checking a prefix.
        for bytes in [
            STAGING_BUFFER_INITIAL_SIZE + 64,
            2 * STAGING_BUFFER_INITIAL_SIZE + 64,
        ] {
            let input: Vec<u32> = (0..bytes as usize / 4)
                .map(|i| i as u32 ^ 0xa55a_5aa5)
                .collect();
            let buffer = gpu.memory().create_storage_buffer(bytes).unwrap();
            gpu.memory().upload_typed(&buffer, &input).unwrap();
            let mut output = vec![0_u32; input.len()];
            gpu.memory()
                .download_typed(&buffer, output.len(), &mut output)
                .unwrap();
            assert_eq!(input, output);
            gpu.memory().destroy_buffer(buffer);
        }
    }
}
