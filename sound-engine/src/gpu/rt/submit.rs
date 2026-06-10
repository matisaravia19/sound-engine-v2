use super::*;
use crate::error::SoundResult;
use crate::gpu::compute::FrameToken;

/// Submission token returned by RT work recorded on the compute queue.
pub type RtFrameToken = FrameToken;

/// Convenience submit methods for RT command recording.
///
/// RT currently uses the same compute queue and command-buffer pool as compute
/// dispatches.
impl RtContext {
    pub fn submit_rt<F>(&self, record: F) -> SoundResult<RtFrameToken>
    where
        F: FnOnce(vk::CommandBuffer) -> SoundResult<()>,
    {
        self.compute.submit_compute(record)
    }

    pub fn submit_rt_and_wait<F>(&self, record: F) -> SoundResult<()>
    where
        F: FnOnce(vk::CommandBuffer) -> SoundResult<()>,
    {
        self.compute.submit_compute_and_wait(record)
    }
}
