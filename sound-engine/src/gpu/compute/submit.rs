use super::*;
use std::sync::atomic::Ordering;

impl ComputeContext {
    /// Records work into a command buffer and submits it to the compute queue.
    ///
    /// The returned token must be waited on before resources written by the
    /// submission are read on the host.
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

        // Reuse command buffers after their fence has completed.
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

    /// Submits compute work and waits for completion before returning.
    pub fn submit_compute_and_wait<F>(&self, record: F) -> Result<(), GpuError>
    where
        F: FnOnce(vk::CommandBuffer) -> Result<(), GpuError>,
    {
        let token = self.submit_compute(record)?;
        self.wait_for(token)
    }

    /// Waits for a previously submitted frame token and recycles its command buffer.
    pub fn wait_for(&self, token: FrameToken) -> Result<(), GpuError> {
        let in_flight = self
            .in_flight
            .lock()
            .map_err(|_| std::io::Error::other("Compute in_flight lock is poisoned"))?
            .remove(&token.0)
            .ok_or_else(|| std::io::Error::other(format!("Unknown frame token {}", token.0)))?;

        self.wait_for_submission(&in_flight)
    }

    pub(super) fn wait_for_all(&self) -> Result<(), GpuError> {
        let mut in_flight = self
            .in_flight
            .lock()
            .map_err(|_| std::io::Error::other("Compute in_flight lock is poisoned"))?;

        for (_, submission) in in_flight.drain() {
            self.wait_for_submission(&submission)?;
        }

        Ok(())
    }

    fn wait_for_submission(&self, submission: &InFlightSubmission) -> Result<(), GpuError> {
        unsafe {
            self.device_context
                .device
                .wait_for_fences(std::slice::from_ref(&submission.fence), true, u64::MAX)?;
            self.device_context.device.destroy_fence(submission.fence, None);
        }

        self.available_command_buffers
            .lock()
            .map_err(|_| std::io::Error::other("Compute command buffer pool lock is poisoned"))?
            .push(submission.command_buffer);

        Ok(())
    }
}
