use crate::error::{SoundError, SoundResult};
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
mod sync;

pub use dispatch::{DescriptorWrite, DispatchSpec};
pub use pipeline::{ComputePipelineSpec, DescriptorBindingSpec, PushConstantSpec};
pub use sync::{BufferBarrierSpec, MemoryBarrierSpec};

const COMMAND_BUFFER_POOL_SIZE: u32 = 12;
const DESCRIPTOR_SETS_PER_POOL: u32 = 64;

/// Owns compute pipelines and command submission state for the backend.
///
/// The context records work onto a preallocated pool of command buffers and
/// tracks in-flight submissions by `FrameToken`.
pub struct ComputeContext {
    device_context: Arc<VkDeviceContext>,
    shaders: Arc<ShaderLibrary>,
    next_token: AtomicU64,
    in_flight: Mutex<HashMap<u64, InFlightSubmission>>,
    queue: RwLock<QueueSet>,
    available_command_buffers: Mutex<Vec<vk::CommandBuffer>>,
    pipelines: Mutex<HashMap<PipelineId, ComputePipeline>>,
}

/// Token returned for an asynchronous compute submission.
pub struct FrameToken(pub u64);

struct InFlightSubmission {
    fence: vk::Fence,
    command_buffer: vk::CommandBuffer,
}

/// Stable handle to a compute pipeline stored in `ComputeContext`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineId(u32);

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
    ) -> SoundResult<Self> {
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

    /// Returns the raw compute queue for integrations that record directly.
    ///
    /// # Safety
    ///
    /// Callers must preserve the queue synchronization guarantees expected by
    /// the compute context.
    pub unsafe fn queue_handle(&self) -> SoundResult<vk::Queue> {
        let queue = self
            .queue
            .read()
            .map_err(|_| SoundError::poisoned_lock("Compute queue lock is poisoned"))?;
        Ok(queue.handle)
    }

    /// Returns the raw compute command pool for integrations that need it.
    ///
    /// # Safety
    ///
    /// Callers must not free or reset command buffers owned by this context.
    pub unsafe fn command_pool_handle(&self) -> SoundResult<vk::CommandPool> {
        let queue = self
            .queue
            .read()
            .map_err(|_| SoundError::poisoned_lock("Compute queue lock is poisoned"))?;
        Ok(queue.command_pool)
    }

    fn drop_command_pool(&self) -> SoundResult<()> {
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

    fn drop_pipelines(&self) -> SoundResult<()> {
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
