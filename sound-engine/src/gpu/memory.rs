//! Buffer allocation entry points and serialized staging transfers.
use crate::core::error::{SoundError, SoundResult};
use crate::gpu::backend::VkDeviceContext;
use crate::gpu::compute::{BufferBarrierSpec, ComputeContext};
use ash::vk;
use bytemuck::Pod;
use std::sync::{Arc, Mutex};
use vk_mem::{AllocationCreateFlags, MemoryUsage};

mod buffer;

use buffer::AllocatorOwner;
pub use buffer::{BufferHandle, BufferInfo};

const STAGING_BUFFER_INITIAL_SIZE: vk::DeviceSize = 16 * 1024 * 1024; // 16 MB

/// Allocates Vulkan buffers and provides synchronous staging transfers.
///
/// The allocator is configured for buffer device addresses so buffers can be
/// used by acceleration-structure builds and shader binding tables.
pub struct GpuAllocator {
    allocator: Arc<AllocatorOwner>,
    device_context: Arc<VkDeviceContext>,
    compute: Arc<ComputeContext>,
    staging_buffer: Mutex<BufferHandle>,
}

impl GpuAllocator {
    /// Owns device-address-capable allocation state and shares serialized transfers.
    pub(super) fn new(
        device_context: Arc<VkDeviceContext>,
        compute: Arc<ComputeContext>,
    ) -> SoundResult<Self> {
        let allocator = AllocatorOwner::new(device_context.clone())?;
        let staging_buffer = Self::create_staging_buffer(&allocator, STAGING_BUFFER_INITIAL_SIZE)?;

        Ok(Self {
            allocator,
            device_context,
            compute,
            staging_buffer: Mutex::new(staging_buffer),
        })
    }

    /// Creates a device-local storage buffer with transfer source/destination usage.
    pub fn create_storage_buffer(&self, size: u64) -> SoundResult<BufferHandle> {
        self.allocator.allocate_buffer(
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
        self.allocator.allocate_buffer(
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
        self.allocator.allocate_buffer(
            size,
            usage
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryUsage::AutoPreferHost,
            AllocationCreateFlags::MAPPED | AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
        )
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
        if !buffer.usage.contains(vk::BufferUsageFlags::TRANSFER_DST)
            || !buffer.size.is_multiple_of(4)
        {
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

        let mut staging_buffer = self
            .staging_buffer
            .lock()
            .map_err(|_| SoundError::poisoned_lock("Transfer lock is poisoned"))?;

        // Grow staging lazily so large scene uploads do not require a fixed cap.
        self.ensure_staging_capacity(&mut staging_buffer, copy_size)?;
        staging_buffer.write_mapped_bytes(data)?;
        self.copy_buffer(&staging_buffer, destination, copy_size, None)?;
        Ok(())
    }

    /// Uploads a POD slice by viewing it as bytes.
    pub fn upload_typed<T: Pod>(&self, destination: &BufferHandle, data: &[T]) -> SoundResult<()> {
        self.upload_bytes(destination, bytemuck::cast_slice(data))
    }

    /// Fills `output` from the source buffer's prefix and waits for completion.
    /// The output slice determines the transfer size; pass a subslice for a partial read.
    pub fn download_bytes(&self, source: &BufferHandle, output: &mut [u8]) -> SoundResult<()> {
        self.validate_owner(source)?;
        if output.is_empty() {
            return Ok(());
        }
        let copy_size = output.len() as vk::DeviceSize;
        if copy_size > source.size {
            return Err(SoundError::invalid_argument(format!(
                "Download size {} exceeds source buffer size {}",
                output.len(),
                source.size
            )));
        }

        let mut staging_buffer = self
            .staging_buffer
            .lock()
            .map_err(|_| SoundError::poisoned_lock("Transfer lock is poisoned"))?;

        self.ensure_staging_capacity(&mut staging_buffer, copy_size)?;
        self.copy_buffer(
            source,
            &staging_buffer,
            copy_size,
            Some(BufferBarrierSpec::transfer_write_to_host_read(
                staging_buffer.buffer,
            )),
        )?;
        staging_buffer.read_mapped_bytes(output)
    }

    /// Fills `output` with POD elements from the source buffer's prefix.
    /// Shares the synchronous staging transfer used by [`Self::download_bytes`].
    pub fn download_typed<T: Pod>(
        &self,
        source: &BufferHandle,
        output: &mut [T],
    ) -> SoundResult<()> {
        self.download_bytes(source, bytemuck::cast_slice_mut(output))
    }

    fn create_staging_buffer(
        allocator: &Arc<AllocatorOwner>,
        size: vk::DeviceSize,
    ) -> SoundResult<BufferHandle> {
        allocator.allocate_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryUsage::AutoPreferHost,
            AllocationCreateFlags::MAPPED | AllocationCreateFlags::HOST_ACCESS_RANDOM,
        )
    }

    fn ensure_staging_capacity(
        &self,
        staging_buffer: &mut BufferHandle,
        required_size: vk::DeviceSize,
    ) -> SoundResult<()> {
        if staging_buffer.size >= required_size {
            return Ok(());
        }

        // Transfers hold the staging lock through GPU completion, so the old
        // allocation can drop here. Doubling amortizes repeated upload growth.
        let new_size = staging_buffer.size.saturating_mul(2).max(required_size);
        *staging_buffer = Self::create_staging_buffer(&self.allocator, new_size)?;
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

    /// Rejects buffers from another allocator before recording work on this queue.
    pub(crate) fn validate_owner(&self, buffer: &BufferHandle) -> SoundResult<()> {
        if !Arc::ptr_eq(&self.allocator, &buffer.allocator) {
            return Err(SoundError::invalid_argument(
                "Buffer belongs to a different GPU allocator",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/gpu/memory.rs"]
mod tests;
