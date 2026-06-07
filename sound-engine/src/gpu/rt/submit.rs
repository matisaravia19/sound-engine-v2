use super::*;
use crate::gpu::GpuError;
use crate::gpu::compute::{ComputeContext, FrameToken};

/// Submission token returned by RT work recorded on the compute queue.
pub type RtFrameToken = FrameToken;

/// Convenience submit methods for RT command recording.
///
/// RT currently uses the same compute queue and command-buffer pool as compute
/// dispatches.
pub trait RtSubmitExt {
    /// Records RT work and submits it through the compute context.
    fn submit_rt<F>(&self, compute: &ComputeContext, record: F) -> Result<RtFrameToken, GpuError>
    where
        F: FnOnce(vk::CommandBuffer) -> Result<(), GpuError>;

    /// Records RT work, submits it, and waits for completion.
    fn submit_rt_and_wait<F>(&self, compute: &ComputeContext, record: F) -> Result<(), GpuError>
    where
        F: FnOnce(vk::CommandBuffer) -> Result<(), GpuError>;
}

impl RtSubmitExt for RtContext {
    fn submit_rt<F>(&self, compute: &ComputeContext, record: F) -> Result<RtFrameToken, GpuError>
    where
        F: FnOnce(vk::CommandBuffer) -> Result<(), GpuError>,
    {
        compute.submit_compute(record)
    }

    fn submit_rt_and_wait<F>(&self, compute: &ComputeContext, record: F) -> Result<(), GpuError>
    where
        F: FnOnce(vk::CommandBuffer) -> Result<(), GpuError>,
    {
        compute.submit_compute_and_wait(record)
    }
}
