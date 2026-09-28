use crate::core::error::{SoundError, SoundResult};
use crate::gpu::backend::{QueueSet, VkDeviceContext};
use crate::gpu::shader::ShaderLibrary;
use ash::vk;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64};
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

/// Token returned for an asynchronous compute submission.
pub struct FrameToken(pub u64);

struct InFlightSubmission {
    fence: vk::Fence,
    command_buffer: vk::CommandBuffer,
}

/// Descriptor set retained until the command buffer using it has completed.
#[derive(Clone, Copy)]
pub(crate) struct DescriptorSetAllocation {
    /// Pool that owns the descriptor set.
    pub pool: vk::DescriptorPool,
    /// Set referenced by commands recorded into one submission.
    pub set: vk::DescriptorSet,
}

/// Stable handle to a compute pipeline stored in `ComputeContext`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineId(u32);

struct ComputePipeline {
    device_context: Arc<VkDeviceContext>,
    descriptor_bindings: Vec<vk::DescriptorSetLayoutBinding<'static>>,
    handle: vk::Pipeline,
    layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    push_constant_ranges: Vec<vk::PushConstantRange>,
    pool_sizes_template: Vec<vk::DescriptorPoolSize>,
    descriptor_pools: Vec<vk::DescriptorPool>,
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
            next_token: AtomicU64::new(1),
            next_pipeline_id: AtomicU32::new(0),
            in_flight: Mutex::new(HashMap::new()),
            queue: RwLock::new(compute_queue),
            available_command_buffers: Mutex::new(preallocated_command_buffers),
            descriptor_sets_by_command_buffer: Mutex::new(HashMap::new()),
            descriptor_pool_access: Mutex::new(()),
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

    fn drop_command_pool(&self) -> SoundResult<()> {
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

        Ok(())
    }

    fn drop_pipelines(&self) -> SoundResult<()> {
        self.pipelines
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clear();
        Ok(())
    }

    /// Releases a private synchronous user's pipeline after its final dispatch.
    pub(crate) fn release_pipeline(&self, id: PipelineId) {
        self.pipelines
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remove(&id);
    }

    /// Returns the live pipeline count for lifetime regression tests.
    #[cfg(test)]
    pub(crate) fn pipeline_count(&self) -> usize {
        self.pipelines.lock().unwrap().len()
    }

    /// Retains a descriptor set until the command buffer's fence completes.
    pub(crate) fn track_descriptor_set(
        &self,
        command_buffer: vk::CommandBuffer,
        allocation: DescriptorSetAllocation,
    ) -> SoundResult<()> {
        self.descriptor_sets_by_command_buffer
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .entry(command_buffer)
            .or_default()
            .push(allocation);
        Ok(())
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

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe {
            for pool in &self.descriptor_pools {
                self.device_context
                    .device
                    .destroy_descriptor_pool(*pool, None);
            }
            self.device_context
                .device
                .destroy_pipeline(self.handle, None);
            self.device_context
                .device
                .destroy_pipeline_layout(self.layout, None);
            self.device_context
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
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
        let _ = self.drop_pipelines();
        let _ = self.drop_command_pool();
    }
}
