use crate::core::error::{SoundError, SoundResult};
use crate::gpu::backend::{QueueSet, VkDeviceContext};
use crate::gpu::descriptor::{DescriptorPools, DescriptorSetAllocation};
use crate::gpu::resource::PipelineResources;
use crate::gpu::shader::ShaderLibrary;
use ash::vk;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64};
use std::sync::{Arc, Mutex, RwLock};

mod dispatch;
mod pipeline;
mod submit;
mod sync;

pub use dispatch::DispatchSpec;
pub use pipeline::{ComputePipelineSpec, PushConstantSpec};
pub use submit::Submission;
pub use sync::{BufferBarrierSpec, MemoryBarrierSpec};

const COMMAND_BUFFER_POOL_SIZE: u32 = 12;

/// Owns compute pipelines and command submission state for the backend.
///
/// The context records work onto a preallocated pool of command buffers.
/// Submission handles borrow it and expose an explicit completion wait.
pub struct ComputeContext {
    device_context: Arc<VkDeviceContext>,
    shaders: Arc<ShaderLibrary>,
    next_submission_id: AtomicU64,
    next_pipeline_id: AtomicU32,
    in_flight: Mutex<HashMap<u64, InFlightSubmission>>,
    queue: RwLock<QueueSet>,
    available_command_buffers: Mutex<Vec<vk::CommandBuffer>>,
    descriptor_sets_by_command_buffer:
        Mutex<HashMap<vk::CommandBuffer, Vec<DescriptorSetAllocation>>>,
    // Vulkan requires external synchronization for both allocation and freeing
    // from a descriptor pool, including pools shared with RT submissions.
    descriptor_pool_access: Mutex<()>,
    pipelines: Mutex<HashMap<PipelineId, ComputePipeline>>,
}

struct InFlightSubmission {
    fence: vk::Fence,
    command_buffer: vk::CommandBuffer,
}

/// Stable handle to a compute pipeline stored in `ComputeContext`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineId(u32);

struct ComputePipeline {
    resources: PipelineResources,
    descriptor_bindings: Vec<vk::DescriptorSetLayoutBinding<'static>>,
    push_constant_ranges: Vec<vk::PushConstantRange>,
    descriptor_pools: DescriptorPools,
}

impl ComputeContext {
    /// Shares device ownership and allocates buffers from the supplied command pool.
    pub(super) fn new(
        device_context: Arc<VkDeviceContext>,
        shaders: Arc<ShaderLibrary>,
        compute_queue: QueueSet,
    ) -> SoundResult<Self> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(compute_queue.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(COMMAND_BUFFER_POOL_SIZE);
        let preallocated_command_buffers =
            match unsafe { device_context.device.allocate_command_buffers(&alloc_info) } {
                Ok(buffers) => buffers,
                Err(error) => {
                    unsafe {
                        device_context
                            .device
                            .destroy_command_pool(compute_queue.command_pool, None)
                    };
                    return Err(error.into());
                }
            };

        Ok(Self {
            device_context,
            shaders,
            next_submission_id: AtomicU64::new(1),
            next_pipeline_id: AtomicU32::new(0),
            in_flight: Mutex::new(HashMap::new()),
            queue: RwLock::new(compute_queue),
            available_command_buffers: Mutex::new(preallocated_command_buffers),
            descriptor_sets_by_command_buffer: Mutex::new(HashMap::new()),
            descriptor_pool_access: Mutex::new(()),
            pipelines: Mutex::new(HashMap::new()),
        })
    }

    /// Serializes integrations that use both the queue and its command pool.
    pub(crate) fn with_queue<T>(
        &self,
        operation: impl FnOnce(vk::Queue, vk::CommandPool) -> SoundResult<T>,
    ) -> SoundResult<T> {
        let queue = self
            .queue
            .write()
            .map_err(|_| SoundError::poisoned_lock("Compute queue lock is poisoned"))?;
        operation(queue.handle, queue.command_pool)
    }

    /// Serializes descriptor pool allocation and freeing across compute and RT.
    pub(crate) fn descriptor_pool_guard(&self) -> std::sync::MutexGuard<'_, ()> {
        self.descriptor_pool_access
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }

    fn drop_command_pool(&self) {
        if let Ok(mut available) = self.available_command_buffers.lock() {
            available.clear();
        }

        {
            let queue = self
                .queue
                .write()
                .unwrap_or_else(|error| error.into_inner());
            unsafe {
                self.device_context
                    .device
                    .queue_wait_idle(queue.handle)
                    .ok();
                self.device_context
                    .device
                    .destroy_command_pool(queue.command_pool, None);
            }
        }
    }

    fn drop_pipelines(&self) {
        self.pipelines
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clear();
    }

    /// Releases a private synchronous user's pipeline after its final dispatch.
    pub(crate) fn release_pipeline(&self, id: PipelineId) {
        self.pipelines
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remove(&id);
    }

    /// Retains a descriptor set until the command buffer's fence completes.
    pub(super) fn track_descriptor_set(
        &self,
        command_buffer: vk::CommandBuffer,
        allocation: DescriptorSetAllocation,
    ) {
        self.descriptor_sets_by_command_buffer
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .entry(command_buffer)
            .or_default()
            .push(allocation);
    }

    /// Frees descriptor sets after all commands that reference them complete.
    fn release_descriptor_sets(&self, command_buffer: vk::CommandBuffer) -> SoundResult<()> {
        let _pool_access = self.descriptor_pool_guard();
        let allocations = self
            .descriptor_sets_by_command_buffer
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remove(&command_buffer)
            .unwrap_or_default();
        for allocation in allocations {
            unsafe {
                self.device_context
                    .device
                    .free_descriptor_sets(allocation.pool, std::slice::from_ref(&allocation.set))?;
            }
        }
        Ok(())
    }
}

impl Drop for ComputeContext {
    fn drop(&mut self) {
        // Device loss and poisoned locks must not turn cleanup into a second
        // panic. A lost device still permits Vulkan object destruction.
        let _ = self.wait_for_all();
        let queue = self
            .queue
            .get_mut()
            .unwrap_or_else(|error| error.into_inner());
        unsafe {
            let _ = self.device_context.device.queue_wait_idle(queue.handle);
        }
        let submissions: Vec<_> = self
            .in_flight
            .get_mut()
            .unwrap_or_else(|error| error.into_inner())
            .drain()
            .map(|(_, value)| value)
            .collect();
        for submission in submissions {
            unsafe {
                self.device_context
                    .device
                    .destroy_fence(submission.fence, None);
            }
            let _ = self.release_descriptor_sets(submission.command_buffer);
        }
        self.drop_pipelines();
        self.drop_command_pool();
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/gpu/compute/mod.rs"]
mod tests;
