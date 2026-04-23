use crate::gpu::GpuError;
use crate::gpu::backend::{QueueSet, VkDeviceContext};
use ash::vk;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

const COMMAND_BUFFER_POOL_SIZE: u32 = 12;

pub(crate) struct ComputeContext {
    device_context: Arc<VkDeviceContext>,
    next_token: AtomicU64,
    in_flight: Mutex<HashMap<u64, InFlightSubmission>>,
    queue: RwLock<QueueSet>,
    available_command_buffers: Mutex<Vec<vk::CommandBuffer>>,
}

pub(crate) struct FrameToken(pub u64);

struct InFlightSubmission {
    fence: vk::Fence,
    command_buffer: vk::CommandBuffer,
}

impl ComputeContext {
    pub(super) fn new(device_context: Arc<VkDeviceContext>, compute_queue: QueueSet) -> Result<Self, GpuError> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(compute_queue.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(COMMAND_BUFFER_POOL_SIZE);
        let preallocated_command_buffers = unsafe { device_context.device.allocate_command_buffers(&alloc_info)? };

        Ok(Self {
            device_context,
            next_token: AtomicU64::new(1),
            in_flight: Mutex::new(HashMap::new()),
            queue: RwLock::new(compute_queue),
            available_command_buffers: Mutex::new(preallocated_command_buffers),
        })
    }

    pub fn submit_compute<F>(&self, record: F) -> Result<FrameToken, GpuError>
    where
        F: FnOnce(vk::CommandBuffer) -> Result<(), GpuError>,
    {
        let token = self.next_token.fetch_add(1, Ordering::Relaxed);
        let fence = self.device_context.create_fence()?;

        let queue = self
            .queue
            .write()
            .map_err(|_| std::io::Error::other("Compute queue lock is poisoned"))?;

        let command_buffer = self.acquire_command_buffer(&queue)?;

        unsafe {
            self.device_context
                .device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;

            let begin_info = vk::CommandBufferBeginInfo::default();
            self.device_context
                .device
                .begin_command_buffer(command_buffer, &begin_info)?;
        }

        record(command_buffer)?;

        unsafe {
            self.device_context.device.end_command_buffer(command_buffer)?;
        }

        let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));
        unsafe {
            self.device_context
                .device
                .queue_submit(queue.handle, std::slice::from_ref(&submit_info), fence)?;
        }

        self.in_flight
            .lock()
            .map_err(|_| std::io::Error::other("Compute in_flight lock is poisoned"))?
            .insert(token, InFlightSubmission { fence, command_buffer });

        Ok(FrameToken(token))
    }

    fn acquire_command_buffer(&self, queue: &QueueSet) -> Result<vk::CommandBuffer, GpuError> {
        if let Some(command_buffer) = self
            .available_command_buffers
            .lock()
            .map_err(|_| std::io::Error::other("Compute command buffer pool lock is poisoned"))?
            .pop()
        {
            return Ok(command_buffer);
        }

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(queue.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe { self.device_context.device.allocate_command_buffers(&alloc_info)? }[0];
        Ok(command_buffer)
    }

    pub fn submit_compute_and_wait<F>(&self, record: F) -> Result<(), GpuError>
    where
        F: FnOnce(vk::CommandBuffer) -> Result<(), GpuError>,
    {
        let token = self.submit_compute(record)?;
        self.wait_for(token)
    }

    pub fn wait_for(&self, token: FrameToken) -> Result<(), GpuError> {
        let in_flight = self
            .in_flight
            .lock()
            .map_err(|_| std::io::Error::other("Compute in_flight lock is poisoned"))?
            .remove(&token.0)
            .ok_or_else(|| std::io::Error::other(format!("Unknown frame token {}", token.0)))?;

        unsafe {
            self.device_context
                .device
                .wait_for_fences(std::slice::from_ref(&in_flight.fence), true, u64::MAX)?;
            self.device_context.device.destroy_fence(in_flight.fence, None);
        }

        self.available_command_buffers
            .lock()
            .map_err(|_| std::io::Error::other("Compute command buffer pool lock is poisoned"))?
            .push(in_flight.command_buffer);

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
        let queue = self
            .queue
            .write()
            .expect("Compute queue lock is poisoned during ComputeContext drop");

        if let Ok(mut in_flight) = self.in_flight.lock() {
            for (_, submission) in in_flight.drain() {
                unsafe {
                    let _ = self.device_context.device.wait_for_fences(
                        std::slice::from_ref(&submission.fence),
                        true,
                        u64::MAX,
                    );
                    self.device_context.device.destroy_fence(submission.fence, None);
                }
            }
        }

        if let Ok(mut available) = self.available_command_buffers.lock() {
            available.clear();
        }

        unsafe {
            self.device_context.device.queue_wait_idle(queue.handle).ok();
            self.device_context
                .device
                .destroy_command_pool(queue.command_pool, None);
        }
    }
}
