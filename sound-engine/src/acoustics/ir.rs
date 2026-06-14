use crate::scene::SceneVersion;
use bytemuck::{Pod, Zeroable};

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
