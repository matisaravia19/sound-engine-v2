use crate::gpu::GpuError;
use crate::gpu::backend::{QueueSet, VkDeviceContext};
use ash::vk;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

pub(crate) struct ComputeContext {
    device_context: Arc<VkDeviceContext>,
    next_token: AtomicU64,
    in_flight: Mutex<HashMap<u64, vk::Fence>>,
    queue: RwLock<QueueSet>,
}

pub(crate) struct FrameToken(pub u64);

impl ComputeContext {
    pub(super) fn new(device_context: Arc<VkDeviceContext>, compute_queue: QueueSet) -> Self {
        Self {
            device_context,
            next_token: AtomicU64::new(1),
            in_flight: Mutex::new(HashMap::new()),
            queue: RwLock::new(compute_queue),
        }
    }

    pub fn submit_compute(&self, command_buffer: vk::CommandBuffer) -> Result<FrameToken, GpuError> {
        let token = self.next_token.fetch_add(1, Ordering::Relaxed);

        let fence = self.device_context.create_fence()?;

        let queue = self
            .queue
            .write()
            .map_err(|_| std::io::Error::other("Compute queue lock is poisoned"))?;

        let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));
        unsafe {
            self.device_context
                .device
                .queue_submit(queue.handle, std::slice::from_ref(&submit_info), fence)?;
        }

        self.in_flight
            .lock()
            .map_err(|_| std::io::Error::other("Compute in_flight lock is poisoned"))?
            .insert(token, fence);

        Ok(FrameToken(token))
    }

    pub fn wait_for(&self, token: FrameToken) -> Result<(), GpuError> {
        let fence = self
            .in_flight
            .lock()
            .map_err(|_| std::io::Error::other("Compute in_flight lock is poisoned"))?
            .remove(&token.0)
            .ok_or_else(|| std::io::Error::other(format!("Unknown frame token {}", token.0)))?;

        unsafe {
            self.device_context
                .device
                .wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)?;
            self.device_context.device.destroy_fence(fence, None);
        }

        Ok(())
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
}

impl Drop for ComputeContext {
    fn drop(&mut self) {
        if let Ok(mut in_flight) = self.in_flight.lock() {
            for (_, fence) in in_flight.drain() {
                unsafe {
                    let _ = self
                        .device_context
                        .device
                        .wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX);
                    self.device_context.device.destroy_fence(fence, None);
                }
            }
        }

        let queue = self
            .queue
            .write()
            .expect("Compute queue lock is poisoned during ComputeContext drop");

        unsafe {
            self.device_context.device.queue_wait_idle(queue.handle).ok();
            self.device_context
                .device
                .destroy_command_pool(queue.command_pool, None);
        }
    }
}
