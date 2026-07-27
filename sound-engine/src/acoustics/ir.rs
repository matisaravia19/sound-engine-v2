use crate::scene::SceneVersion;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::sync::Arc;

use crate::gpu::memory::BufferHandle;

/// One listener capture event produced by acoustic ray tracing.
#[derive(Debug, Clone, Copy)]
pub struct AcousticHit {
    /// Direction from which energy arrives at the listener in world space.
    pub incoming_direction: Vec3,
    /// Direction from listener center to the hit point on the listener sphere.
    pub listener_hit_direction: Vec3,
    /// Arrival time relative to source emission.
    pub arrival_time_seconds: f32,
    /// Acoustic energy carried by this hit before listener orientation binning.
    pub energy: f32,
}

/// Direction-independent hit snapshot ready for listener-orientation binning.
#[derive(Debug, Clone)]
pub struct AcousticHitSnapshot {
    /// Listener capture events recorded by ray tracing.
    pub hits: Vec<AcousticHit>,
    /// Scene version represented by the traced paths.
    pub scene_version: SceneVersion,
    /// Query identifier copied from the acoustic request.
    pub query_id: u32,
    /// Source position in world meters used for tracing.
    pub source_position: Vec3,
    /// Acoustic source energy used for tracing.
    pub source_energy: f32,
    /// Listener position in world meters used for tracing.
    pub listener_position: Vec3,
    /// Number of listener hits dropped because the GPU hit buffer filled.
    pub overflow_count: u32,
}

/// GPU-resident listener hit snapshot ready for orientation-dependent IR binning.
#[derive(Clone)]
pub struct GpuAcousticHitSnapshot {
    /// GPU buffer containing a hit header followed by packed hit records.
    ///
    /// `None` represents an invalid/test snapshot and is rejected by GPU pipeline methods.
    pub buffer: Option<Arc<BufferHandle>>,
    /// Number of hit records the buffer can hold.
    pub hit_capacity: u32,
    /// Total storage buffer size in bytes.
    pub byte_size: usize,
    /// Scene version represented by the traced paths.
    pub scene_version: SceneVersion,
    /// Query identifier copied from the acoustic request.
    pub query_id: u32,
    /// Source position in world meters used for tracing.
    pub source_position: Vec3,
    /// Acoustic source energy used for tracing.
    pub source_energy: f32,
    /// Listener position in world meters used for tracing.
    pub listener_position: Vec3,
}

/// GPU-resident time-domain impulse response.
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
