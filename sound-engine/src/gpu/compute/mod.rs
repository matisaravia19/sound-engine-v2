use crate::core::error::{SoundError, SoundResult};
use crate::gpu::backend::VkDeviceContext;
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
    // Retire the command pool before releasing shader and pipeline owners.
    queue: RwLock<ComputeQueue>,
    device_context: Arc<VkDeviceContext>,
    shaders: Arc<ShaderLibrary>,
    next_submission_id: AtomicU64,
    next_pipeline_id: AtomicU32,
    in_flight: Mutex<HashMap<u64, InFlightSubmission>>,
    available_command_buffers: Mutex<Vec<vk::CommandBuffer>>,
    descriptor_sets_by_command_buffer:
        Mutex<HashMap<vk::CommandBuffer, Vec<DescriptorSetAllocation>>>,
    // Vulkan requires external synchronization for both allocation and freeing
    // from a descriptor pool, including pools shared with RT submissions.
    descriptor_pool_access: Mutex<()>,
    pipelines: Mutex<HashMap<PipelineId, ComputePipeline>>,
}

/// Owns the command pool and retains its device, including during failed initialization.
/// The enclosing context serializes queue/pool access and drains tracked submissions first.
struct ComputeQueue {
    handle: vk::Queue,
    command_pool: vk::CommandPool,
    device_context: Arc<VkDeviceContext>,
}

impl ComputeQueue {
    fn new(device_context: Arc<VkDeviceContext>, family: u32) -> SoundResult<Self> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe {
            device_context
                .device
                .create_command_pool(&pool_info, None)?
        };
        let handle = unsafe { device_context.device.get_device_queue(family, 0) };
        Ok(Self {
            handle,
            command_pool,
            device_context,
        })
    }
}

impl Drop for ComputeQueue {
    fn drop(&mut self) {
        unsafe {
            // Native FFT initialization also submits outside the tracked registry.
            let _ = self.device_context.device.queue_wait_idle(self.handle);
            self.device_context
                .device
                .destroy_command_pool(self.command_pool, None);
        }
    }
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
    /// Creates the compute queue's command pool and preallocates its submission buffers.
    pub(super) fn new(
        device_context: Arc<VkDeviceContext>,
        shaders: Arc<ShaderLibrary>,
        compute_family: u32,
    ) -> SoundResult<Self> {
        let queue = ComputeQueue::new(device_context.clone(), compute_family)?;
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(queue.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(COMMAND_BUFFER_POOL_SIZE);
        let preallocated_command_buffers = unsafe {
            device_context
                .device
                .allocate_command_buffers(&alloc_info)?
        };

        Ok(Self {
            queue: RwLock::new(queue),
            device_context,
            shaders,
            next_submission_id: AtomicU64::new(1),
            next_pipeline_id: AtomicU32::new(0),
            in_flight: Mutex::new(HashMap::new()),
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
        // Draining recycles every tracked fence and descriptor set even after
        // device loss. Field-owned pipelines retain the device through their drop.
        let _ = self.wait_for_all();
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/gpu/compute/mod.rs"]
mod tests;
