use super::*;
use std::sync::atomic::Ordering;

// Reclaims unsubmitted work on every error path, including a recording panic.
// The queue/pool write guard in submit_compute outlives this guard.
struct PendingSubmission<'a> {
    context: &'a ComputeContext,
    submission: Option<InFlightSubmission>,
}

impl Drop for PendingSubmission<'_> {
    fn drop(&mut self) {
        if let Some(submission) = self.submission.take() {
            let _ = self.context.recycle_submission(submission);
        }
    }
}

impl ComputeContext {
    /// Records and submits work on the serialized compute queue.
    ///
    /// # Safety
    /// The caller must retain every referenced GPU resource until `wait_for`
    /// completes and prevent conflicting host accesses in the meantime.
    /// Dependencies inside `record` remain the caller's responsibility.
    pub unsafe fn submit_compute<F>(&self, record: F) -> SoundResult<FrameToken>
    where
        F: FnOnce(vk::CommandBuffer) -> SoundResult<()>,
    {
        let queue = self
            .queue
            .write()
            .map_err(|_| SoundError::poisoned_lock("Compute queue lock is poisoned"))?;
        let command_buffer = self.acquire_command_buffer(&queue)?;
        let mut pending = PendingSubmission {
            context: self,
            submission: Some(InFlightSubmission {
                fence: vk::Fence::null(),
                command_buffer,
            }),
        };
        let fence = self.device_context.create_fence()?;
        pending.submission.as_mut().unwrap().fence = fence;
        unsafe {
            self.device_context
                .device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;
            self.device_context
                .device
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())?;
        }
        // A fence makes completion observable to the CPU, but does not replace
        // a device memory dependency between uploads, AS builds, FFTs and traces.
        unsafe {
            self.record_memory_barrier(
                command_buffer,
                &MemoryBarrierSpec {
                    src_stage: vk::PipelineStageFlags::ALL_COMMANDS,
                    dst_stage: vk::PipelineStageFlags::ALL_COMMANDS,
                    src_access: vk::AccessFlags::MEMORY_WRITE,
                    dst_access: vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
                },
            );
        }
        record(command_buffer)?;
        unsafe {
            self.record_memory_barrier(
                command_buffer,
                &MemoryBarrierSpec {
                    src_stage: vk::PipelineStageFlags::ALL_COMMANDS,
                    dst_stage: vk::PipelineStageFlags::HOST,
                    src_access: vk::AccessFlags::MEMORY_WRITE,
                    dst_access: vk::AccessFlags::HOST_READ,
                },
            );
        }
        unsafe {
            self.device_context
                .device
                .end_command_buffer(command_buffer)?;
        }
        // Acquire tracking before submission so no fallible lock can strand
        // successfully submitted work without a fence owner.
        let mut in_flight = self
            .in_flight
            .lock()
            .map_err(|_| SoundError::poisoned_lock("Compute in_flight lock is poisoned"))?;
        let token = self
            .next_token
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| SoundError::resource_allocation_failed("Frame token space exhausted"))?;
        let submit_info =
            vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));
        unsafe {
            self.device_context.device.queue_submit(
                queue.handle,
                std::slice::from_ref(&submit_info),
                fence,
            )?;
        }
        in_flight.insert(token, pending.submission.take().unwrap());
        Ok(FrameToken(token))
    }

    fn acquire_command_buffer(&self, queue: &QueueSet) -> SoundResult<vk::CommandBuffer> {
        if let Some(buffer) = self
            .available_command_buffers
            .lock()
            .map_err(|_| SoundError::poisoned_lock("Compute command buffer pool lock is poisoned"))?
            .pop()
        {
            return Ok(buffer);
        }
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(queue.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        Ok(unsafe { self.device_context.device.allocate_command_buffers(&info)? }[0])
    }

    /// Submits work and waits before borrowed resources can be released.
    pub fn submit_compute_and_wait<F>(&self, record: F) -> SoundResult<()>
    where
        F: FnOnce(vk::CommandBuffer) -> SoundResult<()>,
    {
        // No borrow can end before the completion wait below, including its error paths.
        let token = unsafe { self.submit_compute(record)? };
        self.wait_for(token)
    }

    /// Waits for a token and recycles its descriptors, fence and command buffer.
    /// Transient allocation failures are reported only after completion or device
    /// loss, so synchronous callers can safely release their borrowed resources.
    pub fn wait_for(&self, token: FrameToken) -> SoundResult<()> {
        let mut in_flight = self
            .in_flight
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let submission = in_flight
            .get(&token.0)
            .ok_or_else(|| SoundError::not_found(format!("Unknown frame token {}", token.0)))?;
        let mut wait_error = None;
        loop {
            let result = unsafe {
                self.device_context.device.wait_for_fences(
                    std::slice::from_ref(&submission.fence),
                    true,
                    u64::MAX,
                )
            };
            match result {
                Ok(()) => break,
                Err(vk::Result::ERROR_DEVICE_LOST) => {
                    wait_error = Some(vk::Result::ERROR_DEVICE_LOST);
                    break;
                }
                Err(error) => {
                    // An OOM/timeout does not prove completion. Returning here
                    // would let synchronous callers destroy buffers still in use.
                    wait_error.get_or_insert(error);
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
        }
        let submission = in_flight.remove(&token.0).unwrap();
        self.recycle_submission(submission)?;
        wait_error.map_or(Ok(()), |error| Err(error.into()))
    }

    /// Completes tracked submissions without losing ownership after a failed wait.
    pub(crate) fn wait_for_all(&self) -> SoundResult<()> {
        let mut first_error = None;
        loop {
            let token = self
                .in_flight
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .keys()
                .next()
                .copied();
            match token {
                Some(token) => {
                    if let Err(error) = self.wait_for(FrameToken(token)) {
                        first_error.get_or_insert(error);
                    }
                }
                None => return first_error.map_or(Ok(()), Err),
            }
        }
    }

    fn recycle_submission(&self, submission: InFlightSubmission) -> SoundResult<()> {
        unsafe {
            self.device_context
                .device
                .destroy_fence(submission.fence, None);
        }
        let released = self.release_descriptor_sets(submission.command_buffer);
        self.available_command_buffers
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .push(submission.command_buffer);
        released
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires Vulkan; run hardware tests serially"]
    fn audit_failed_recording_recycles_command_buffer_and_fence() {
        let gpu = crate::gpu::backend::VkBackend::new().unwrap();
        for _ in 0..32 {
            assert!(
                // The deliberately failing closure records no resource access.
                unsafe {
                    gpu.compute().submit_compute(|_| {
                        Err(SoundError::invalid_state("intentional recording failure"))
                    })
                }
                .is_err()
            );
            assert!(gpu.compute().in_flight.lock().unwrap().is_empty());
            assert_eq!(
                gpu.compute()
                    .available_command_buffers
                    .lock()
                    .unwrap()
                    .len(),
                COMMAND_BUFFER_POOL_SIZE as usize
            );
        }
        gpu.compute().submit_compute_and_wait(|_| Ok(())).unwrap();
    }
}
