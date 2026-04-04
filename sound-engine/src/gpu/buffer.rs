use crate::gpu::context::GpuContext;
use crate::gpu::errors::GpuError;
use ash::vk::{Buffer, DeviceSize, PhysicalDevice};
use ash::{vk, Device, Instance};
use vk_mem::{Alloc, Allocation, AllocationCreateFlags, Allocator, MemoryUsage};

const STAGING_BUFFER_INITIAL_SIZE: DeviceSize = 16 * 1024 * 1024; // 16 MB

pub struct GpuBuffer {
    size: DeviceSize,
    buffer: Buffer,
    allocation: Allocation,
}

pub(super) struct BufferContext {
    allocator: Allocator,
    staging_buffer: GpuBuffer,
}

impl GpuBuffer {
    pub fn size(&self) -> DeviceSize {
        self.size
    }

    pub fn buffer(&self) -> Buffer {
        self.buffer
    }
}

impl GpuContext {
    pub fn create_buffer(&self, size: DeviceSize) -> Result<GpuBuffer, GpuError> {
        self.buffers_ref()?.create_buffer(size)
    }

    pub fn destroy_buffer(&self, buffer: GpuBuffer) {
        if let Ok(buffers) = self.buffers_ref() {
            buffers.destroy_buffer(buffer);
        }
    }

    pub fn upload_to_buffer(&mut self, buffer: &GpuBuffer, data: &[u8]) -> Result<(), GpuError> {
        {
            let buffers = self.buffers_mut()?;
            buffers.upload_to_staging(data)?;
        }

        self.buffers_ref()?
            .copy_from_staging(self, buffer, data.len() as u64)?;

        Ok(())
    }

    pub fn download_from_buffer(&mut self, buffer: &GpuBuffer, size: usize) -> Result<Vec<u8>, GpuError> {
        self.buffers_ref()?.copy_to_staging(self, buffer, size as u64)?;
        self.buffers_ref()?.download_from_staging(size)
    }
}

impl BufferContext {
    pub fn init(
        instance: &Instance,
        physical_device: PhysicalDevice,
        device: &Device,
    ) -> Result<BufferContext, GpuError> {
        let allocator_create_info = vk_mem::AllocatorCreateInfo::new(&instance, &device, physical_device);
        let allocator = unsafe { Allocator::new(allocator_create_info)? };

        let staging_buffer = Self::create_staging_buffer(&allocator, STAGING_BUFFER_INITIAL_SIZE)?;

        Ok(BufferContext {
            allocator,
            staging_buffer,
        })
    }

    pub fn create_buffer(&self, size: DeviceSize) -> Result<GpuBuffer, GpuError> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let create_info = vk_mem::AllocationCreateInfo {
            usage: MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };

        let (buffer, allocation) = unsafe { self.allocator.create_buffer(&buffer_info, &create_info)? };

        Ok(GpuBuffer {
            size,
            buffer,
            allocation,
        })
    }

    pub fn destroy_buffer(&self, mut buffer: GpuBuffer) {
        unsafe {
            self.allocator
                .destroy_buffer(buffer.buffer, &mut buffer.allocation);
        }
    }

    pub fn upload_to_staging(&mut self, data: &[u8]) -> Result<(), GpuError> {
        self.ensure_staging_capacity(data.len() as DeviceSize)?;

        let mapped_ptr = self
            .allocator
            .get_allocation_info(&self.staging_buffer.allocation)
            .mapped_data;

        if mapped_ptr.is_null() {
            return Err(std::io::Error::other("Failed to map staging buffer memory").into());
        }

        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), mapped_ptr as *mut u8, data.len());
        }

        self.allocator
            .flush_allocation(&self.staging_buffer.allocation, 0, data.len() as u64)?;

        Ok(())
    }

    pub fn copy_from_staging(&self, context: &GpuContext, destination: &GpuBuffer, size: u64) -> Result<(), GpuError> {
        context.immediate_submit(|cmd| {
            let copy_region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size,
            };

            unsafe {
                context.device().cmd_copy_buffer(
                    cmd,
                    self.staging_buffer.buffer,
                    destination.buffer,
                    std::slice::from_ref(&copy_region),
                );
            }

            Ok(())
        })?;

        Ok(())
    }

    pub fn copy_to_staging(&self, context: &GpuContext, source: &GpuBuffer, size: u64) -> Result<(), GpuError> {
        context.immediate_submit(|cmd| {
            let copy_region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size,
            };

            unsafe {
                context.device().cmd_copy_buffer(
                    cmd,
                    source.buffer,
                    self.staging_buffer.buffer,
                    std::slice::from_ref(&copy_region),
                );
            }

            Ok(())
        })?;

        Ok(())
    }

    pub fn download_from_staging(&self, size: usize) -> Result<Vec<u8>, GpuError> {
        let mapped_ptr = self
            .allocator
            .get_allocation_info(&self.staging_buffer.allocation)
            .mapped_data;

        if mapped_ptr.is_null() {
            return Err(std::io::Error::other("Failed to map staging buffer memory").into());
        }

        self.allocator
            .invalidate_allocation(&self.staging_buffer.allocation, 0, size as u64)?;

        let mut data = vec![0u8; size];
        unsafe {
            std::ptr::copy_nonoverlapping(mapped_ptr as *const u8, data.as_mut_ptr(), size);
        }

        Ok(data)
    }

    fn create_staging_buffer(allocator: &Allocator, size: DeviceSize) -> Result<GpuBuffer, GpuError> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let create_info = vk_mem::AllocationCreateInfo {
            usage: MemoryUsage::AutoPreferHost,
            flags: AllocationCreateFlags::MAPPED | AllocationCreateFlags::HOST_ACCESS_RANDOM,
            ..Default::default()
        };

        let (buffer, allocation) = unsafe { allocator.create_buffer(&buffer_info, &create_info)? };

        Ok(GpuBuffer {
            size,
            buffer,
            allocation,
        })
    }

    fn ensure_staging_capacity(&mut self, required_size: DeviceSize) -> Result<(), GpuError> {
        if self.staging_buffer.size >= required_size {
            return Ok(());
        }

        let new_size = std::cmp::max(self.staging_buffer.size * 2, required_size);
        let mut old_staging = std::mem::replace(
            &mut self.staging_buffer,
            Self::create_staging_buffer(&self.allocator, new_size)?,
        );

        unsafe {
            self.allocator
                .destroy_buffer(old_staging.buffer, &mut old_staging.allocation);
        }

        Ok(())
    }
}

impl Drop for BufferContext {
    fn drop(&mut self) {
        unsafe {
            self.allocator
                .destroy_buffer(self.staging_buffer.buffer, &mut self.staging_buffer.allocation);
        }
    }
}
