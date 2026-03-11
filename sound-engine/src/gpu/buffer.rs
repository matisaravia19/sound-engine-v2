use crate::gpu::context::GpuContext;
use crate::gpu::errors::GpuError;
use ash::vk::{Buffer, DeviceSize, PhysicalDevice};
use ash::{Device, Instance, vk};
use std::sync::{Arc, Weak};
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
    pub fn new(context: Weak<GpuContext>, size: DeviceSize, buffer: Buffer, allocation: Allocation) -> Self {
        Self {
            buffer,
            allocation,
            size,
        }
    }

    pub fn size(&self) -> DeviceSize {
        self.size
    }

    pub fn buffer(&self) -> Buffer {
        self.buffer
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

    pub fn upload_to_staging(&mut self, data: &[u8]) -> Result<(), GpuError> {
        self.ensure_staging_capacity(data.len() as DeviceSize)?;

        let mapped_ptr = self
            .allocator
            .get_allocation_info(&self.staging_buffer.allocation)
            .mapped_data;

        if mapped_ptr.is_null() {
            return Err("Failed to map staging buffer memory".into());
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

    fn create_staging_buffer(allocator: &Allocator, size: DeviceSize) -> Result<GpuBuffer, GpuError> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(STAGING_BUFFER_INITIAL_SIZE)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let create_info = vk_mem::AllocationCreateInfo {
            usage: MemoryUsage::AutoPreferHost,
            flags: AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
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

        unsafe {
            self.allocator
                .destroy_buffer(self.staging_buffer.buffer, &mut self.staging_buffer.allocation);
        }

        self.staging_buffer = Self::create_staging_buffer(&self.allocator, new_size)?;

        Ok(())
    }
}
