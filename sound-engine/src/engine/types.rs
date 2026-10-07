//! Spatial playback requests and listener/source coordinates.

use crate::auralization::SoundId;
use glam::Vec3;

/// Identifier for a tracked movable sound source.
pub type SoundSourceId = u64;

/// Listener pose used for spatial acoustic queries.
#[derive(Debug, Clone, Copy)]
pub struct ListenerPose {
    /// Listener position in world meters.
    pub position: Vec3,
    /// Listener forward direction in world space.
    pub forward: Vec3,
    /// Listener up direction in world space.
    ///
    /// Valid poses need not provide an exactly orthogonal unit vector; the
    /// engine orthogonalizes it against `forward` before acoustic queries.
    pub up: Vec3,
}

impl Default for ListenerPose {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            forward: -Vec3::Z,
            up: Vec3::Y,
        }
    }
}

impl ListenerPose {
    /// Creates a listener pose from a position and approximate forward/up axes.
    pub fn new(position: Vec3, forward: Vec3, up: Vec3) -> Self {
        Self {
            position,
            forward,
            up,
        }
    }

    /// Returns the unit right-ear axis derived from this pose.
    ///
    /// Invalid or collinear orientation vectors produce the zero vector; public
    /// engine entry points reject those poses before using them.
    pub fn right(self) -> Vec3 {
        self.forward
            .normalize_or_zero()
            .cross(self.up)
            .normalize_or_zero()
    }
}

/// Point sound source rendered through the current acoustic scene.
#[derive(Debug, Clone, Copy)]
pub struct PointSource {
    /// Source position in world meters.
    pub position: Vec3,
    /// Reference pressure amplitude used when generating an impulse response.
    pub amplitude: f32,
}

impl PointSource {
    /// Creates a point source with unit reference pressure amplitude.
    pub fn new(position: Vec3) -> Self {
        Self {
            position,
            amplitude: 1.0,
        }
    }
}

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
