//! Spatial playback requests and tracked source identifiers.

use crate::acoustics::PointSource;
use crate::auralization::SoundId;
use glam::Vec3;

/// Identifier for a tracked movable sound source.
pub type SoundSourceId = u64;

/// Request used by [`super::SoundEngine`] to play a sound at a world-space position.
#[derive(Debug, Clone, Copy)]
pub struct PlaySpatialSoundRequest {
    /// Dry sound asset to render.
    pub sound_id: SoundId,
    /// World-space source used to build or reuse the acoustic IR.
    pub source: PointSource,
    /// Linear playback volume applied to the dry sound before convolution.
    pub volume: f32,
}

impl PlaySpatialSoundRequest {
    /// Creates a play request with unit source amplitude and unit volume.
    pub fn new(sound_id: SoundId, source_position: Vec3) -> Self {
        Self {
            sound_id,
            source: PointSource::new(source_position),
            volume: 1.0,
        }
    }
}
