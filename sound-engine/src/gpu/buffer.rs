use crate::gpu::context::GpuContext;
use ash::vk::{Buffer, DeviceSize};
use std::sync::Weak;
use vk_mem::Allocation;

pub struct GpuBuffer {
    context: Weak<GpuContext>,
    size: DeviceSize,
    buffer: Buffer,
    allocation: Allocation,
}

impl GpuBuffer {
    pub fn new(context: Weak<GpuContext>, size: DeviceSize, buffer: Buffer, allocation: Allocation) -> Self {
        Self {
            context,
            buffer,
            allocation,
            size,
        }
    }

    pub fn size(&self) -> DeviceSize {
        self.size
    }
}

// impl GpuBufferType {
//     pub fn usage(&self) -> vk::BufferUsageFlags {
//         match self {
//             GpuBufferType::Storage => {
//                 vk::BufferUsageFlags::STORAGE_BUFFER
//                     | vk::BufferUsageFlags::TRANSFER_SRC
//                     | vk::BufferUsageFlags::TRANSFER_DST
//             }
//             GpuBufferType::Staging => vk::BufferUsageFlags::TRANSFER_SRC,
//             GpuBufferType::Readback => vk::BufferUsageFlags::TRANSFER_DST,
//         }
//     }
//
//     pub fn memory_usage(&self) -> vk_mem::MemoryUsage {
//         match self {
//             GpuBufferType::Storage => vk_mem::MemoryUsage::AutoPreferDevice,
//             GpuBufferType::Staging => vk_mem::MemoryUsage::AutoPreferHost,
//             GpuBufferType::Readback => vk_mem::MemoryUsage::AutoPreferHost,
//         }
//     }
//
//     pub fn memory_flags(&self) -> vk_mem::AllocationCreateFlags {
//         match self {
//             GpuBufferType::Storage => vk_mem::AllocationCreateFlags::empty(),
//             GpuBufferType::Staging => vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
//             GpuBufferType::Readback => vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM,
//         }
//     }
// }
