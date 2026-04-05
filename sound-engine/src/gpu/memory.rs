use crate::gpu::GpuError;
use crate::gpu::backend::VkDeviceContext;
use ash::vk;
use std::sync::Arc;
use vk_mem::{Alloc, Allocation, AllocationCreateFlags, Allocator, MemoryUsage};

const STAGING_BUFFER_INITIAL_SIZE: vk::DeviceSize = 16 * 1024 * 1024; // 16 MB

pub(crate) struct GpuAllocator {
    allocator: Allocator,
    device_context: Arc<VkDeviceContext>,
    transfer_command_buffer: vk::CommandBuffer,
    transfer_fence: vk::Fence,
    staging: StagingRing,
}

pub(crate) struct BufferHandle {
    pub buffer: vk::Buffer,
    pub size: vk::DeviceSize,
    allocation: Allocation,
}

pub(crate) struct StagingRing {
    buffer: BufferHandle,
}

pub(crate) struct UploadSlice {
    pub staging_offset: vk::DeviceSize,
    pub size: vk::DeviceSize,
}

impl GpuAllocator {
    pub(super) fn new(device_context: Arc<VkDeviceContext>) -> Result<Self, GpuError> {
        let allocator_info = vk_mem::AllocatorCreateInfo::new(
            &device_context.instance,
            &device_context.device,
            device_context.physical_device,
        );
        let allocator = unsafe { Allocator::new(allocator_info)? };

        let transfer_command_buffer_alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(device_context.queues.transfer_command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let transfer_command_buffer =
            unsafe { device_context.device.allocate_command_buffers(&transfer_command_buffer_alloc_info)? }[0];

        let transfer_fence_info = vk::FenceCreateInfo::default();
        let transfer_fence = unsafe { device_context.device.create_fence(&transfer_fence_info, None)? };

        let staging = StagingRing {
            buffer: Self::create_staging_buffer(&allocator, STAGING_BUFFER_INITIAL_SIZE)?,
        };

        Ok(Self {
            allocator,
            device_context,
            transfer_command_buffer,
            transfer_fence,
            staging,
        })
    }

    pub fn create_storage_buffer(&mut self, size: vk::DeviceSize) -> Result<BufferHandle, GpuError> {
        self.create_buffer(
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryUsage::AutoPreferDevice,
            AllocationCreateFlags::empty(),
        )
    }

    pub fn create_readback_buffer(&mut self, size: vk::DeviceSize) -> Result<BufferHandle, GpuError> {
        self.create_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryUsage::AutoPreferHost,
            AllocationCreateFlags::MAPPED | AllocationCreateFlags::HOST_ACCESS_RANDOM,
        )
    }

    pub fn destroy_buffer(&mut self, mut handle: BufferHandle) {
        unsafe {
            self.allocator.destroy_buffer(handle.buffer, &mut handle.allocation);
        }
    }

    pub fn upload_bytes(&mut self, destination: &BufferHandle, data: &[u8]) -> Result<(), GpuError> {
        if data.len() as u64 > destination.size {
            return Err(std::io::Error::other(format!(
                "Upload size {} exceeds destination buffer size {}",
                data.len(),
                destination.size
            ))
            .into());
        }

        self.upload_to_staging(data)?;
        self.copy_buffer(self.staging.buffer.buffer, destination.buffer, data.len() as u64)?;
        Ok(())
    }

    pub fn download_bytes(&mut self, src: &BufferHandle, size: usize) -> Result<Vec<u8>, GpuError> {
        if size as u64 > src.size {
            return Err(std::io::Error::other(format!(
                "Download size {} exceeds source buffer size {}",
                size, src.size
            ))
            .into());
        }

        self.ensure_staging_capacity(size as u64)?;
        self.copy_buffer(src.buffer, self.staging.buffer.buffer, size as u64)?;
        self.download_from_staging(size)
    }

    fn create_buffer(
        &mut self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_usage: MemoryUsage,
        allocation_flags: AllocationCreateFlags,
    ) -> Result<BufferHandle, GpuError> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let allocation_info = vk_mem::AllocationCreateInfo {
            usage: memory_usage,
            flags: allocation_flags,
            ..Default::default()
        };

        let (buffer, allocation) = unsafe { self.allocator.create_buffer(&buffer_info, &allocation_info)? };
        Ok(BufferHandle {
            buffer,
            size,
            allocation,
        })
    }

    fn create_staging_buffer(allocator: &Allocator, size: vk::DeviceSize) -> Result<BufferHandle, GpuError> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let allocation_info = vk_mem::AllocationCreateInfo {
            usage: MemoryUsage::AutoPreferHost,
            flags: AllocationCreateFlags::MAPPED | AllocationCreateFlags::HOST_ACCESS_RANDOM,
            ..Default::default()
        };

        let (buffer, allocation) = unsafe { allocator.create_buffer(&buffer_info, &allocation_info)? };
        Ok(BufferHandle {
            buffer,
            size,
            allocation,
        })
    }

    fn ensure_staging_capacity(&mut self, required_size: vk::DeviceSize) -> Result<(), GpuError> {
        if self.staging.buffer.size >= required_size {
            return Ok(());
        }

        let new_size = std::cmp::max(self.staging.buffer.size * 2, required_size);
        let mut old = std::mem::replace(
            &mut self.staging.buffer,
            Self::create_staging_buffer(&self.allocator, new_size)?,
        );

        unsafe {
            self.allocator.destroy_buffer(old.buffer, &mut old.allocation);
        }

        Ok(())
    }

    fn upload_to_staging(&mut self, data: &[u8]) -> Result<UploadSlice, GpuError> {
        self.ensure_staging_capacity(data.len() as u64)?;
        let mapped_ptr = self
            .allocator
            .get_allocation_info(&self.staging.buffer.allocation)
            .mapped_data;

        if mapped_ptr.is_null() {
            return Err(std::io::Error::other("Staging buffer is not mapped").into());
        }

        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), mapped_ptr as *mut u8, data.len());
        }

        self.allocator
            .flush_allocation(&self.staging.buffer.allocation, 0, data.len() as u64)?;

        Ok(UploadSlice {
            staging_offset: 0,
            size: data.len() as u64,
        })
    }

    fn download_from_staging(&self, size: usize) -> Result<Vec<u8>, GpuError> {
        let mapped_ptr = self
            .allocator
            .get_allocation_info(&self.staging.buffer.allocation)
            .mapped_data;

        if mapped_ptr.is_null() {
            return Err(std::io::Error::other("Staging buffer is not mapped").into());
        }

        self.allocator
            .invalidate_allocation(&self.staging.buffer.allocation, 0, size as u64)?;

        let mut out = vec![0_u8; size];
        unsafe {
            std::ptr::copy_nonoverlapping(mapped_ptr as *const u8, out.as_mut_ptr(), size);
        }
        Ok(out)
    }

    fn copy_buffer(&self, src: vk::Buffer, dst: vk::Buffer, size: vk::DeviceSize) -> Result<(), GpuError> {
        let command_buffer = self.transfer_command_buffer;

        let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device_context
                .device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;
            self.device_context
                .device
                .begin_command_buffer(command_buffer, &begin_info)?;

            let region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size,
            };
            self.device_context
                .device
                .cmd_copy_buffer(command_buffer, src, dst, std::slice::from_ref(&region));

            self.device_context.device.end_command_buffer(command_buffer)?;

            self.device_context
                .device
                .reset_fences(std::slice::from_ref(&self.transfer_fence))?;

            let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));
            self.device_context.device.queue_submit(
                self.device_context.queues.transfer_queue,
                std::slice::from_ref(&submit_info),
                self.transfer_fence,
            )?;

            self.device_context
                .device
                .wait_for_fences(std::slice::from_ref(&self.transfer_fence), true, u64::MAX)?;
        }

        Ok(())
    }
}

impl Drop for GpuAllocator {
    fn drop(&mut self) {
        unsafe {
            self.allocator
                .destroy_buffer(self.staging.buffer.buffer, &mut self.staging.buffer.allocation);
            self.device_context.device.free_command_buffers(
                self.device_context.queues.transfer_command_pool,
                std::slice::from_ref(&self.transfer_command_buffer),
            );
            self.device_context.device.destroy_fence(self.transfer_fence, None);
        }
    }
}
