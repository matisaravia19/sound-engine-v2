use crate::gpu::GpuError;
use crate::gpu::backend::{QueueSet, VkDeviceContext};
use crate::gpu::shader::ShaderLibrary;
use ash::vk;
use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex, RwLock};

mod descriptor;
mod dispatch;
mod pipeline;
mod submit;

pub(crate) use dispatch::{DescriptorWrite, DispatchSpec};
pub(crate) use pipeline::{ComputePipelineSpec, DescriptorBindingSpec, PushConstantSpec};

const COMMAND_BUFFER_POOL_SIZE: u32 = 12;
const DESCRIPTOR_SETS_PER_POOL: u32 = 64;

pub(crate) struct ComputeContext {
    device_context: Arc<VkDeviceContext>,
    shaders: Arc<ShaderLibrary>,
    next_token: AtomicU64,
    in_flight: Mutex<HashMap<u64, InFlightSubmission>>,
    queue: RwLock<QueueSet>,
    available_command_buffers: Mutex<Vec<vk::CommandBuffer>>,
    pipelines: Mutex<HashMap<PipelineId, ComputePipeline>>,
}

pub(crate) struct FrameToken(pub u64);

struct InFlightSubmission {
    fence: vk::Fence,
    command_buffer: vk::CommandBuffer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PipelineId(u32);

struct ComputePipeline {
    handle: vk::Pipeline,
    layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_bindings: Vec<vk::DescriptorSetLayoutBinding<'static>>,
    push_constant_ranges: Vec<vk::PushConstantRange>,
    pool_sizes_template: Vec<vk::DescriptorPoolSize>,
    descriptor_pools: Vec<vk::DescriptorPool>,
}

impl ComputeContext {
    pub(super) fn new(
        device_context: Arc<VkDeviceContext>,
        shaders: Arc<ShaderLibrary>,
        compute_queue: QueueSet,
    ) -> Result<Self, GpuError> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(compute_queue.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(COMMAND_BUFFER_POOL_SIZE);
        let preallocated_command_buffers = unsafe { device_context.device.allocate_command_buffers(&alloc_info)? };

        Ok(Self {
            device_context,
            shaders,
            next_token: AtomicU64::new(1),
            in_flight: Mutex::new(HashMap::new()),
            queue: RwLock::new(compute_queue),
            available_command_buffers: Mutex::new(preallocated_command_buffers),
            pipelines: Mutex::new(HashMap::new()),
        })
    }

    pub unsafe fn queue_handle(&self) -> Result<vk::Queue, GpuError> {
        let queue = self
            .queue
            .read()
            .map_err(|_| std::io::Error::other("Compute queue lock is poisoned"))?;
        Ok(queue.handle)
    }

    pub unsafe fn command_pool_handle(&self) -> Result<vk::CommandPool, GpuError> {
        let queue = self
            .queue
            .read()
            .map_err(|_| std::io::Error::other("Compute queue lock is poisoned"))?;
        Ok(queue.command_pool)
    }

    fn drop_command_pool(&self) -> Result<(), GpuError> {
        if let Ok(mut available) = self.available_command_buffers.lock() {
            available.clear();
        }

        if let Ok(queue) = self.queue.write() {
            unsafe {
                self.device_context.device.queue_wait_idle(queue.handle).ok();
                self.device_context
                    .device
                    .destroy_command_pool(queue.command_pool, None);
            }
        }

        Ok(())
    }

    fn drop_pipelines(&self) -> Result<(), GpuError> {
        if let Ok(mut pipelines) = self.pipelines.lock() {
            unsafe {
                for (_, pipeline) in pipelines.drain() {
                    for pool in &pipeline.descriptor_pools {
                        self.device_context.device.destroy_descriptor_pool(*pool, None);
                    }

                    self.device_context.device.destroy_pipeline(pipeline.handle, None);
                    self.device_context
                        .device
                        .destroy_pipeline_layout(pipeline.layout, None);
                    self.device_context
                        .device
                        .destroy_descriptor_set_layout(pipeline.descriptor_set_layout, None);
                }
            }
        }

        Ok(())
    }
}

impl Drop for ComputeContext {
    fn drop(&mut self) {
        self.wait_for_all()
            .expect("Failed to wait for compute context to be idle during drop");

        self.drop_pipelines().expect("Failed to drop compute pipelines");
        self.drop_command_pool().expect("Failed to drop compute command pool");
    }
}
