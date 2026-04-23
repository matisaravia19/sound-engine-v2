use crate::gpu::backend::VkDeviceContext;
use crate::gpu::{GpuError, backend::QueueSet};
use ash::vk;
use bytemuck::Pod;
use std::sync::{Arc, Mutex};
use vk_mem::{Alloc, Allocation, AllocationCreateFlags, Allocator, MemoryUsage};

const STAGING_BUFFER_INITIAL_SIZE: vk::DeviceSize = 16 * 1024 * 1024; // 16 MB

pub(crate) struct GpuAllocator {
    allocator: Arc<Allocator>,
    device_context: Arc<VkDeviceContext>,
    transfer: Mutex<TransferContext>,
}

pub(crate) struct BufferHandle {
    pub buffer: vk::Buffer,
    pub size: vk::DeviceSize,
    allocation: Allocation,
    allocator: Arc<Allocator>,
}

struct TransferContext {
    queue: QueueSet,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    staging_buffer: BufferHandle,
}

struct UploadSlice {
    pub staging_offset: vk::DeviceSize,
    pub size: vk::DeviceSize,
}

impl GpuAllocator {
    pub(super) fn new(device_context: Arc<VkDeviceContext>, transfer_queue: QueueSet) -> Result<Self, GpuError> {
        let allocator_info = vk_mem::AllocatorCreateInfo::new(
            &device_context.instance,
            &device_context.device,
            device_context.physical_device,
        );
        let allocator = unsafe { Arc::new(Allocator::new(allocator_info)?) };

        let transfer_command_buffer_alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(transfer_queue.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let transfer_command_buffer = unsafe {
            device_context
                .device
                .allocate_command_buffers(&transfer_command_buffer_alloc_info)?
        }[0];

        let transfer_fence_info = vk::FenceCreateInfo::default();
        let transfer_fence = unsafe { device_context.device.create_fence(&transfer_fence_info, None)? };

        let staging_buffer = Self::create_staging_buffer(&allocator, STAGING_BUFFER_INITIAL_SIZE)?;

        let transfer = TransferContext {
            queue: transfer_queue,
            command_buffer: transfer_command_buffer,
            fence: transfer_fence,
            staging_buffer,
        };

        Ok(Self {
            allocator,
            device_context,
            transfer: Mutex::new(transfer),
        })
    }

    pub fn create_storage_buffer(&self, size: u64) -> Result<BufferHandle, GpuError> {
        self.create_buffer(
            size as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryUsage::AutoPreferDevice,
            AllocationCreateFlags::empty(),
        )
    }

    pub fn create_readback_buffer(&self, size: vk::DeviceSize) -> Result<BufferHandle, GpuError> {
        self.create_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryUsage::AutoPreferHost,
            AllocationCreateFlags::MAPPED | AllocationCreateFlags::HOST_ACCESS_RANDOM,
        )
    }

    pub fn destroy_buffer(&self, mut handle: BufferHandle) {
        unsafe {
            self.allocator.destroy_buffer(handle.buffer, &mut handle.allocation);
        }
    }

    pub fn upload_bytes(&self, destination: &BufferHandle, data: &[u8]) -> Result<(), GpuError> {
        if data.len() as u64 > destination.size {
            return Err(std::io::Error::other(format!(
                "Upload size {} exceeds destination buffer size {}",
                data.len(),
                destination.size
            ))
            .into());
        }

        let mut transfer = self
            .transfer
            .lock()
            .map_err(|_| std::io::Error::other("Transfer lock is poisoned"))?;

        self.ensure_staging_capacity(&mut transfer, data.len() as u64)?;
        self.upload_to_staging(&transfer, data)?;
        self.copy_buffer(&transfer, &transfer.staging_buffer, destination)?;
        Ok(())
    }

    pub fn upload_typed<T: Pod>(&self, destination: &BufferHandle, data: &[T]) -> Result<(), GpuError> {
        self.upload_bytes(destination, bytemuck::cast_slice(data))
    }

    pub fn download_bytes(&self, src: &BufferHandle, size: usize, out: &mut [u8]) -> Result<(), GpuError> {
        if size as u64 > src.size {
            return Err(std::io::Error::other(format!(
                "Download size {} exceeds source buffer size {}",
                size, src.size
            ))
            .into());
        }

        let mut transfer = self
            .transfer
            .lock()
            .map_err(|_| std::io::Error::other("Transfer lock is poisoned"))?;

        self.ensure_staging_capacity(&mut transfer, size as u64)?;
        self.copy_buffer(&transfer, &src, &transfer.staging_buffer)?;
        self.download_from_staging(&transfer, size, out)?;

        Ok(())
    }

    pub fn download_typed<T: Pod>(&self, src: &BufferHandle, count: usize, out: &mut [T]) -> Result<(), GpuError> {
        let element_size = std::mem::size_of::<T>();
        let byte_size = count
            .checked_mul(element_size)
            .ok_or_else(|| std::io::Error::other("download_typed byte size overflow"))?;

        self.download_bytes(src, byte_size, bytemuck::cast_slice_mut::<T, u8>(out))?;

        return Ok(());
    }

    fn create_buffer(
        &self,
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
            allocator: self.allocator.clone(),
        })
    }

    fn create_staging_buffer(allocator: &Arc<Allocator>, size: vk::DeviceSize) -> Result<BufferHandle, GpuError> {
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
            allocator: allocator.clone(),
        })
    }

    fn ensure_staging_capacity(
        &self,
        transfer: &mut TransferContext,
        required_size: vk::DeviceSize,
    ) -> Result<(), GpuError> {
        if transfer.staging_buffer.size >= required_size {
            return Ok(());
        }

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

    fn upload_to_staging(&self, transfer: &TransferContext, data: &[u8]) -> Result<UploadSlice, GpuError> {
        let mapped_ptr = self
            .allocator
            .get_allocation_info(&transfer.staging_buffer.allocation)
            .mapped_data;

        if mapped_ptr.is_null() {
            return Err(std::io::Error::other("Staging buffer is not mapped").into());
        }

        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), mapped_ptr as *mut u8, data.len());
        }

        self.allocator
            .flush_allocation(&transfer.staging_buffer.allocation, 0, data.len() as u64)?;

        Ok(UploadSlice {
            staging_offset: 0,
            size: data.len() as u64,
        })
    }

    fn download_from_staging(&self, transfer: &TransferContext, size: usize, out: &mut [u8]) -> Result<(), GpuError> {
        let mapped_ptr = self
            .allocator
            .get_allocation_info(&transfer.staging_buffer.allocation)
            .mapped_data;

        if mapped_ptr.is_null() {
            return Err(std::io::Error::other("Staging buffer is not mapped").into());
        }

        self.allocator
            .invalidate_allocation(&transfer.staging_buffer.allocation, 0, size as u64)?;

        unsafe {
            std::ptr::copy_nonoverlapping(mapped_ptr as *const u8, out.as_mut_ptr(), size);
        }
        Ok(())
    }

    fn copy_buffer(&self, transfer: &TransferContext, src: &BufferHandle, dst: &BufferHandle) -> Result<(), GpuError> {
        let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device_context
                .device
                .reset_command_buffer(transfer.command_buffer, vk::CommandBufferResetFlags::empty())?;
            self.device_context
                .device
                .begin_command_buffer(transfer.command_buffer, &begin_info)?;

            let region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: src.size,
            };
            self.device_context.device.cmd_copy_buffer(
                transfer.command_buffer,
                src.buffer,
                dst.buffer,
                std::slice::from_ref(&region),
            );

            self.device_context.device.end_command_buffer(transfer.command_buffer)?;

            self.device_context
                .device
                .reset_fences(std::slice::from_ref(&transfer.fence))?;

            let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&transfer.command_buffer));
            self.device_context.device.queue_submit(
                transfer.queue.handle,
                std::slice::from_ref(&submit_info),
                transfer.fence,
            )?;

            self.device_context
                .device
                .wait_for_fences(std::slice::from_ref(&transfer.fence), true, u64::MAX)?;
        }

        Ok(())
    }
}

impl Drop for GpuAllocator {
    fn drop(&mut self) {
        let transfer = self
            .transfer
            .lock()
            .expect("Transfer lock is poisoned during GpuAllocator drop");

        unsafe {
            self.device_context.device.free_command_buffers(
                transfer.queue.command_pool,
                std::slice::from_ref(&transfer.command_buffer),
            );

            self.device_context.device.destroy_fence(transfer.fence, None);

            self.device_context.device.queue_wait_idle(transfer.queue.handle).ok();
            self.device_context
                .device
                .destroy_command_pool(transfer.queue.command_pool, None);
        }
    }
}

impl Drop for BufferHandle {
    fn drop(&mut self) {
        unsafe {
            self.allocator.destroy_buffer(self.buffer, &mut self.allocation);
        }
    }
}
