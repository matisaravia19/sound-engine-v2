use super::AcousticQuery;
use super::spatial::ValidatedAcousticQuery;
use crate::core::error::{SoundError, SoundResult};
use crate::gpu::backend::VkBackend;
use crate::gpu::memory::BufferHandle;
use crate::scene::SceneVersion;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

/// Shared GPU-resident time-domain impulse response with immutable provenance.
///
/// Construction verifies the sample layout against the retained allocation.
/// Cache entries, readback, and convolution share the buffer's ownership. The
/// producer must finish its writes and make them visible before publishing it;
/// the acoustic pipeline does this before returning a snapshot.
#[derive(Clone)]
pub struct GpuIrSnapshot {
    buffer: Arc<BufferHandle>,
    sample_rate: u32,
    sample_count: u32,
    scene_version: SceneVersion,
    query: ValidatedAcousticQuery,
}

impl GpuIrSnapshot {
    /// Wraps a completed GPU IR, validating its extent and canonicalizing its query.
    /// The buffer may contain padding beyond the declared sample count.
    pub fn new(
        buffer: Arc<BufferHandle>,
        sample_rate: u32,
        sample_count: u32,
        scene_version: SceneVersion,
        query: AcousticQuery,
    ) -> SoundResult<Self> {
        Self::from_validated_query(
            buffer,
            sample_rate,
            sample_count,
            scene_version,
            query.canonicalized()?,
        )
    }

    /// Publishes a completed IR while preserving the exact query used by the pipeline.
    pub(super) fn from_validated_query(
        buffer: Arc<BufferHandle>,
        sample_rate: u32,
        sample_count: u32,
        scene_version: SceneVersion,
        query: ValidatedAcousticQuery,
    ) -> SoundResult<Self> {
        validate_layout(buffer.size, sample_rate, sample_count)?;
        Ok(Self {
            buffer,
            sample_rate,
            sample_count,
            scene_version,
            query,
        })
    }

    /// Borrows the shared sample buffer, whose owner keeps its allocator and device alive.
    pub fn buffer(&self) -> &Arc<BufferHandle> {
        &self.buffer
    }

    /// Returns the full retained allocation size, including any unused padding.
    pub fn byte_size(&self) -> u64 {
        self.buffer.size
    }

    /// Returns the sample rate used to generate the IR.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Returns the number of valid stereo samples at the start of the buffer.
    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// Returns the scene revision represented by this response.
    pub fn scene_version(&self) -> SceneVersion {
        self.scene_version
    }

    /// Returns the canonical source/listener query that produced this response.
    pub fn query(&self) -> AcousticQuery {
        self.query.as_query()
    }

    /// Downloads the samples and measures their summed stereo energy.
    /// `gpu` must own this snapshot's buffer; a foreign backend is rejected.
    pub fn download(&self, gpu: &VkBackend) -> SoundResult<IrSnapshot> {
        let mut samples = vec![IrSample::zero(); self.sample_count as usize];
        gpu.memory().download_typed(&self.buffer, &mut samples)?;
        let energy = samples
            .iter()
            .map(|sample| sample.left * sample.left + sample.right * sample.right)
            .sum();
        Ok(IrSnapshot {
            sample_rate: self.sample_rate,
            samples,
            energy,
            scene_version: self.scene_version,
            query_id: self.query().query_id,
        })
    }
}

fn validate_layout(buffer_bytes: u64, sample_rate: u32, sample_count: u32) -> SoundResult<()> {
    if sample_rate == 0 || sample_count == 0 {
        return Err(SoundError::invalid_argument(
            "GPU IR sample rate and sample count must be nonzero",
        ));
    }
    let sample_bytes = u64::from(sample_count) * std::mem::size_of::<IrSample>() as u64;
    if sample_bytes > buffer_bytes {
        return Err(SoundError::invalid_argument(
            "IR sample count exceeds its GPU buffer",
        ));
    }
    Ok(())
}

/// Stereo impulse-response sample stored as left and right channel values.
///
/// This layout matches the shader's std430 `IrSample`. Values are signed
/// pressure contributions rather than squared energy.
#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct IrSample {
    /// Left output channel sample value.
    pub left: f32,
    /// Right output channel sample value.
    pub right: f32,
}

impl IrSample {
    /// Creates a new stereo IR sample with the given left and right values.
    pub fn new(left: f32, right: f32) -> Self {
        Self {
            left,
            right,
        }
    }

    /// Creates a zero-valued stereo IR sample.
    pub fn zero() -> Self {
        Self {
            left: 0.0,
            right: 0.0,
        }
    }
}

/// CPU snapshot ready for convolution after IR construction.
#[derive(Debug, Clone)]
pub struct IrSnapshot {
    /// Sample rate used to generate `samples`.
    pub sample_rate: u32,
    /// Impulse-response samples.
    pub samples: Vec<IrSample>,
    /// Sum of squared sample values for diagnostics and sanity checks.
    pub energy: f32,
    /// Scene version represented by this IR.
    pub scene_version: SceneVersion,
    /// Query identifier copied from the acoustic request.
    pub query_id: u32,
}

#[cfg(test)]
#[path = "../../tests/unit/acoustics/ir.rs"]
mod tests;
