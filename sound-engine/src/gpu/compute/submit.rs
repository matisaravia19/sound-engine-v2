use super::*;
use std::sync::atomic::Ordering;

/// Completion handle for asynchronous work submitted through one compute context.
///
/// The borrow keeps the context alive. Call [`Self::wait`] before releasing the
/// submitted work's resources; the handle does not own those resources or wait on drop.
#[must_use = "submitted GPU work must complete before its resources are released"]
pub struct Submission<'context> {
    context: &'context ComputeContext,
    id: u64,
}

impl Submission<'_> {
    /// Waits for completion on the originating context and recycles submission resources.
    ///
    /// Transient allocation failures are reported only after completion or device loss,
    /// so callers can release the resources referenced by their submitted commands.
    pub fn wait(self) -> SoundResult<()> {
        self.context.complete_submission(self.id)
    }
}

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
    /// The caller must retain every referenced GPU resource until [`Submission::wait`]
    /// completes and prevent conflicting host accesses in the meantime.
    /// Dependencies inside `record` remain the caller's responsibility.
    pub unsafe fn submit_compute<F>(&self, record: F) -> SoundResult<Submission<'_>>
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
        let id = self
            .next_submission_id
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| {
                SoundError::resource_allocation_failed("Submission identifier space exhausted")
            })?;
        let submit_info =
            vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));
        unsafe {
            self.device_context.device.queue_submit(
                queue.handle,
                std::slice::from_ref(&submit_info),
                fence,
            )?;
        }
        in_flight.insert(id, pending.submission.take().unwrap());
        Ok(Submission {
            context: self,
            id,
        })
    }

    fn acquire_command_buffer(&self, queue: &ComputeQueue) -> SoundResult<vk::CommandBuffer> {
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
        unsafe { self.submit_compute(record)? }.wait()
    }

    fn complete_submission(&self, id: u64) -> SoundResult<()> {
        let mut in_flight = self
            .in_flight
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let submission = in_flight
            .get(&id)
            .ok_or_else(|| SoundError::not_found(format!("Unknown submission {id}")))?;
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
        let submission = in_flight.remove(&id).unwrap();
        self.recycle_submission(submission)?;
        wait_error.map_or(Ok(()), |error| Err(error.into()))
    }

    /// Completes and recycles tracked submissions, returning the first error after draining all.
    pub(crate) fn wait_for_all(&self) -> SoundResult<()> {
        let mut first_error = None;
        loop {
            let id = self
                .in_flight
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .keys()
                .next()
                .copied();
            match id {
                Some(id) => {
                    if let Err(error) = self.complete_submission(id) {
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
#[path = "../../../tests/unit/gpu/compute/submit.rs"]
mod tests;
