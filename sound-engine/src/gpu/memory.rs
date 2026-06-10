use crate::error::{SoundError, SoundResult};
use crate::gpu::backend::VkDeviceContext;
use crate::gpu::compute::ComputeContext;
use ash::vk;
use bytemuck::Pod;
use std::sync::{Arc, Mutex};
use vk_mem::{Alloc, Allocation, AllocationCreateFlags, Allocator, AllocatorCreateFlags, MemoryUsage};

const STAGING_BUFFER_INITIAL_SIZE: vk::DeviceSize = 16 * 1024 * 1024; // 16 MB

/// Allocates Vulkan buffers and provides synchronous staging transfers.
///
/// The allocator is configured for buffer device addresses so buffers can be
/// used by acceleration-structure builds and shader binding tables.
pub struct GpuAllocator {
    allocator: Arc<Allocator>,
    device_context: Arc<VkDeviceContext>,
    compute: Arc<ComputeContext>,
    transfer: Mutex<TransferContext>,
}

/// RAII handle for a Vulkan buffer allocated through VMA.
pub struct BufferHandle {
    /// Raw Vulkan buffer handle used in descriptors and command recording.
    pub buffer: vk::Buffer,
    /// Buffer size in bytes.
    pub size: vk::DeviceSize,
    allocation: Allocation,
    allocator: Arc<Allocator>,
}

struct TransferContext {
    staging_buffer: BufferHandle,
}

impl GpuAllocator {
    pub(super) fn new(device_context: Arc<VkDeviceContext>, compute: Arc<ComputeContext>) -> SoundResult<Self> {
        let mut allocator_info = vk_mem::AllocatorCreateInfo::new(
            &device_context.instance,
            &device_context.device,
            device_context.physical_device,
        );
        allocator_info.flags |= AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;
        allocator_info.vulkan_api_version = vk::API_VERSION_1_3;
        let allocator =
            unsafe {
                Arc::new(Allocator::new(allocator_info).map_err(|e| {
                    SoundError::resource_allocation_failed(format!("GPU allocator creation failed: {e:?}"))
                })?)
            };

        let staging_buffer = Self::create_staging_buffer(&allocator, STAGING_BUFFER_INITIAL_SIZE)?;

        let transfer = TransferContext { staging_buffer };

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
    pub fn create_device_address_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> SoundResult<BufferHandle> {
        self.create_buffer(
            size,
            usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryUsage::AutoPreferDevice,
            AllocationCreateFlags::empty(),
        )
    }

    /// Creates a mapped host-visible buffer that also has a device address.
    ///
    /// This is useful for small RT inputs such as instance data or SBT records.
    pub fn create_host_device_address_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> SoundResult<BufferHandle> {
        self.create_buffer(
            size,
            usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryUsage::AutoPreferHost,
            AllocationCreateFlags::MAPPED | AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
        )
    }

    /// Returns the Vulkan device address for a buffer created with address usage.
    pub fn buffer_device_address(&self, buffer: &BufferHandle) -> vk::DeviceAddress {
        let info = vk::BufferDeviceAddressInfo::default().buffer(buffer.buffer);
        unsafe { self.device_context.device.get_buffer_device_address(&info) }
    }

    /// Writes bytes directly into a mapped buffer and flushes the written range.
    pub fn write_mapped_bytes(&self, destination: &BufferHandle, data: &[u8]) -> SoundResult<()> {
        if data.len() as vk::DeviceSize > destination.size {
            return Err(SoundError::invalid_argument(format!(
                "Mapped write size {} exceeds destination buffer size {}",
                data.len(),
                destination.size
            )));
        }

        let mapped_ptr = self.allocator.get_allocation_info(&destination.allocation).mapped_data;
        if mapped_ptr.is_null() {
            return Err(SoundError::invalid_state("Destination buffer is not mapped"));
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
    pub fn read_mapped_bytes(&self, source: &BufferHandle, size: usize, out: &mut [u8]) -> SoundResult<()> {
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

        let mapped_ptr = self.allocator.get_allocation_info(&source.allocation).mapped_data;
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

    /// Destroys a buffer immediately instead of waiting for `Drop`.
    ///
    /// Callers must ensure no queued GPU work still uses the buffer.
    pub fn destroy_buffer(&self, mut handle: BufferHandle) {
        unsafe {
            self.allocator.destroy_buffer(handle.buffer, &mut handle.allocation);
        }
    }

    /// Fills the whole buffer with zero using the compute queue transfer path.
    pub fn clear_buffer(&self, buffer: &BufferHandle) -> SoundResult<()> {
        self.compute.submit_compute_and_wait(|command_buffer| unsafe {
            self.device_context
                .device
                .cmd_fill_buffer(command_buffer, buffer.buffer, 0, buffer.size, 0);

            Ok(())
        })
    }

    /// Uploads bytes through the shared staging buffer and waits for completion.
    pub fn upload_bytes(&self, destination: &BufferHandle, data: &[u8]) -> SoundResult<()> {
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
        self.copy_buffer(&transfer.staging_buffer, destination, copy_size)?;
        Ok(())
    }

    /// Uploads a POD slice by viewing it as bytes.
    pub fn upload_typed<T: Pod>(&self, destination: &BufferHandle, data: &[T]) -> SoundResult<()> {
        self.upload_bytes(destination, bytemuck::cast_slice(data))
    }

    /// Downloads bytes through the shared staging buffer and waits for completion.
    pub fn download_bytes(&self, src: &BufferHandle, size: usize, out: &mut [u8]) -> SoundResult<()> {
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
        self.copy_buffer(src, &transfer.staging_buffer, copy_size)?;
        self.download_from_staging(&transfer, size, out)?;

        Ok(())
    }

    /// Downloads `count` POD elements into `out`.
    pub fn download_typed<T: Pod>(&self, src: &BufferHandle, count: usize, out: &mut [T]) -> SoundResult<()> {
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
                .map_err(|e| SoundError::resource_allocation_failed(format!("Buffer allocation failed: {e:?}")))?
        };
        Ok(BufferHandle {
            buffer,
            size,
            allocation,
            allocator: self.allocator.clone(),
        })
    }

    fn create_staging_buffer(allocator: &Arc<Allocator>, size: vk::DeviceSize) -> SoundResult<BufferHandle> {
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
            allocator.create_buffer(&buffer_info, &allocation_info).map_err(|e| {
                SoundError::resource_allocation_failed(format!("Staging buffer allocation failed: {e:?}"))
            })?
        };
        Ok(BufferHandle {
            buffer,
            size,
            allocation,
            allocator: allocator.clone(),
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
        let new_size = std::cmp::max(transfer.staging_buffer.size * 2, required_size);
        let mut old = std::mem::replace(
            &mut transfer.staging_buffer,
            Self::create_staging_buffer(&self.allocator, new_size)?,
        );

        unsafe {
            self.allocator.destroy_buffer(old.buffer, &mut old.allocation);
        }

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

        self.allocator
            .flush_allocation(&transfer.staging_buffer.allocation, 0, data.len() as u64)?;

        Ok(())
    }

    fn download_from_staging(&self, transfer: &TransferContext, size: usize, out: &mut [u8]) -> SoundResult<()> {
        let mapped_ptr = self
            .allocator
            .get_allocation_info(&transfer.staging_buffer.allocation)
            .mapped_data;

        if mapped_ptr.is_null() {
            return Err(SoundError::invalid_state("Staging buffer is not mapped"));
        }

        self.allocator
            .invalidate_allocation(&transfer.staging_buffer.allocation, 0, size as u64)?;

        unsafe {
            std::ptr::copy_nonoverlapping(mapped_ptr as *const u8, out.as_mut_ptr(), size);
        }
        Ok(())
    }

    fn copy_buffer(
        &self,
        src: &BufferHandle,
        dst: &BufferHandle,
        size: vk::DeviceSize,
    ) -> SoundResult<()> {
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

        self.compute.submit_compute_and_wait(|command_buffer| unsafe {
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

            Ok(())
        })
    }
}

impl Drop for BufferHandle {
    fn drop(&mut self) {
        unsafe {
            self.allocator.destroy_buffer(self.buffer, &mut self.allocation);
        }
    }
}
