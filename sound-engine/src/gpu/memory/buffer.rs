//! Buffer allocation, host mapping, and device lifetime ownership.
use crate::core::error::{SoundError, SoundResult};
use crate::gpu::backend::VkDeviceContext;
use ash::vk;
use std::sync::{Arc, Mutex};
use vk_mem::{
    Alloc, Allocation, AllocationCreateFlags, Allocator, AllocatorCreateFlags, MemoryUsage,
};

/// Owns a VMA buffer, its allocator, and the device/loader required for destruction.
/// Submitted work must finish before the buffer is released or mapped on the host.
pub struct BufferHandle {
    info: BufferInfo,
    allocation: Allocation,
    /// Shared allocation owner, retained even after the backend is released.
    pub(super) allocator: Arc<AllocatorOwner>,
    mapped_access: Mutex<()>,
    /// Effective Vulkan usage, checked before recording fills or copies.
    pub(super) usage: vk::BufferUsageFlags,
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

/// Retains the Vulkan device and loader through VMA allocator destruction.
/// Field order releases VMA before the device handles that it copies.
pub(super) struct AllocatorOwner {
    allocator: Allocator,
    device_context: Arc<VkDeviceContext>,
}

impl std::ops::Deref for AllocatorOwner {
    type Target = Allocator;

    fn deref(&self) -> &Allocator {
        &self.allocator
    }
}

impl BufferHandle {
    /// Returns this buffer's address on its owning device.
    /// The address remains valid until the buffer is released.
    ///
    /// # Panics
    /// Panics if the buffer was created without shader-device-address usage.
    pub fn device_address(&self) -> vk::DeviceAddress {
        assert!(
            self.usage
                .contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS),
            "Buffer lacks SHADER_DEVICE_ADDRESS usage"
        );
        let info = vk::BufferDeviceAddressInfo::default().buffer(self.buffer);
        unsafe {
            self.allocator
                .device_context
                .device
                .get_buffer_device_address(&info)
        }
    }

    /// Writes a prefix of this mapped buffer and flushes it for device access.
    /// Serializes host access; callers must finish GPU work that uses the buffer first.
    pub fn write_mapped_bytes(&self, data: &[u8]) -> SoundResult<()> {
        let _access = self
            .mapped_access
            .lock()
            .map_err(|_| SoundError::poisoned_lock("Mapped buffer lock is poisoned"))?;
        if data.len() as vk::DeviceSize > self.size {
            return Err(SoundError::invalid_argument(format!(
                "Mapped write size {} exceeds destination buffer size {}",
                data.len(),
                self.size
            )));
        }

        let mapped_ptr = self
            .allocator
            .get_allocation_info(&self.allocation)
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
            .flush_allocation(&self.allocation, 0, data.len() as u64)?;
        Ok(())
    }

    /// Invalidates and reads a prefix after the caller has completed device writes.
    /// Serializes host access without submitting work or waiting on a queue.
    pub(super) fn read_mapped_bytes(&self, output: &mut [u8]) -> SoundResult<()> {
        let _access = self
            .mapped_access
            .lock()
            .map_err(|_| SoundError::poisoned_lock("Mapped buffer lock is poisoned"))?;
        if output.len() as vk::DeviceSize > self.size {
            return Err(SoundError::invalid_argument(format!(
                "Mapped read size {} exceeds source buffer size {}",
                output.len(),
                self.size
            )));
        }

        let mapped_ptr = self
            .allocator
            .get_allocation_info(&self.allocation)
            .mapped_data;
        if mapped_ptr.is_null() {
            return Err(SoundError::invalid_state("Source buffer is not mapped"));
        }

        // GPU completion and invalidation are both needed for non-coherent host reads.
        self.allocator
            .invalidate_allocation(&self.allocation, 0, output.len() as u64)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                mapped_ptr as *const u8,
                output.as_mut_ptr(),
                output.len(),
            );
        }
        Ok(())
    }
}

impl AllocatorOwner {
    /// Creates address-capable VMA state that retains its Vulkan device.
    pub(super) fn new(device_context: Arc<VkDeviceContext>) -> SoundResult<Arc<Self>> {
        let mut allocator_info = vk_mem::AllocatorCreateInfo::new(
            &device_context.instance,
            &device_context.device,
            device_context.physical_device,
        );
        allocator_info.flags |= AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;
        allocator_info.vulkan_api_version = vk::API_VERSION_1_3;
        unsafe {
            Ok(Arc::new(Self {
                allocator: Allocator::new(allocator_info).map_err(|e| {
                    SoundError::resource_allocation_failed(format!(
                        "GPU allocator creation failed: {e:?}"
                    ))
                })?,
                device_context,
            }))
        }
    }

    /// Allocates and binds a buffer, rejecting unsupported usage and invalid sizes.
    pub(super) fn allocate_buffer(
        self: &Arc<Self>,
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
            allocator: self.clone(),
            mapped_access: Mutex::new(()),
            usage,
        })
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
#[path = "../../../tests/unit/gpu/memory/buffer.rs"]
mod tests;
