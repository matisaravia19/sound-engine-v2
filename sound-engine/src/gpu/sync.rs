use crate::gpu::GpuError;
use crate::gpu::backend::VkDeviceContext;
use ash::vk;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

pub(crate) struct SyncContext {
    device_context: Arc<VkDeviceContext>,
    next_token: AtomicU64,
    in_flight: Mutex<HashMap<u64, vk::Fence>>,
}

pub(crate) struct FrameToken(pub u64);

impl SyncContext {
    pub(super) fn new(device_context: Arc<VkDeviceContext>) -> Self {
        Self {
            device_context,
            next_token: AtomicU64::new(1),
            in_flight: Mutex::new(HashMap::new()),
        }
    }

    pub fn submit_compute(&self, command_buffer: vk::CommandBuffer) -> Result<FrameToken, GpuError> {
        let token = self.next_token.fetch_add(1, Ordering::Relaxed);

        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe { self.device_context.device.create_fence(&fence_info, None)? };

        let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));
        unsafe {
            self.device_context.device.queue_submit(
                self.device_context.queues.compute_queue,
                std::slice::from_ref(&submit_info),
                fence,
            )?;
        }

        self.in_flight
            .lock()
            .map_err(|_| std::io::Error::other("Sync in_flight lock is poisoned"))?
            .insert(token, fence);

        Ok(FrameToken(token))
    }

    pub fn wait_for(&self, token: FrameToken) -> Result<(), GpuError> {
        let fence = self
            .in_flight
            .lock()
            .map_err(|_| std::io::Error::other("Sync in_flight lock is poisoned"))?
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
}

impl Drop for SyncContext {
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
    }
}
