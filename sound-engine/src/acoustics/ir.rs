use crate::scene::SceneVersion;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

use crate::gpu::memory::BufferHandle;

/// GPU-resident time-domain impulse response.
///
/// The optional buffer is reference-counted so cache entries and convolution
/// setup can share ownership. Ray-tracing writes are made visible before the
/// acoustic pipeline returns a snapshot.
#[derive(Clone)]
pub struct GpuIrSnapshot {
    /// GPU buffer containing `IrSample` values.
    ///
    /// `None` represents an invalid/test snapshot and is rejected by auralization and readback methods.
    pub buffer: Option<Arc<BufferHandle>>,
    /// Buffer size in bytes.
    pub byte_size: usize,
    /// Sample rate used to generate the IR.
    pub sample_rate: u32,
    /// Number of `IrSample` entries in `buffer`.
    pub sample_count: u32,
    /// Sum of squared sample values when known without readback.
    pub energy: Option<f32>,
    /// Scene version represented by this IR.
    pub scene_version: SceneVersion,
    /// Query identifier copied from the acoustic request.
    pub query_id: u32,
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
        Self { left, right }
    }

    /// Creates a zero-valued stereo IR sample.
    pub fn zero() -> Self {
        Self { left: 0.0, right: 0.0 }
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
