//! Acoustic endpoints, orientation normalization, and source validation.

use crate::core::error::{SoundError, SoundResult};
use glam::Vec3;

/// Source/listener pair that should produce one impulse response.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcousticQuery {
    /// Application-defined identifier copied into the returned IR snapshot.
    pub query_id: u32,
    /// Source position and reference pressure amplitude.
    pub source: PointSource,
    /// Listener position and orientation in world space.
    pub listener: ListenerPose,
}

impl AcousticQuery {
    /// Validates the source and returns an orthonormal listener basis.
    pub(crate) fn canonicalized(mut self) -> SoundResult<Self> {
        validate_source(self.source)?;
        self.listener = canonicalize_listener(self.listener)?;
        Ok(self)
    }
}

/// Listener pose used for spatial acoustic queries.
#[derive(Debug, Clone, Copy, PartialEq)]
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
#[derive(Debug, Clone, Copy, PartialEq)]
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

/// Validates finite source parameters before they reach acoustic queries.
pub(crate) fn validate_source(source: PointSource) -> SoundResult<()> {
    if !source.amplitude.is_finite() || source.amplitude < 0.0 {
        return Err(SoundError::invalid_argument(
            "source amplitude must be finite and non-negative",
        ));
    }
    if !source.position.is_finite() {
        return Err(SoundError::invalid_argument(
            "source position must be finite",
        ));
    }
    Ok(())
}

/// Validates and orthonormalizes listener parameters before acoustic queries.
pub(crate) fn canonicalize_listener(listener: ListenerPose) -> SoundResult<ListenerPose> {
    if !listener.position.is_finite() {
        return Err(SoundError::invalid_argument(
            "listener position must be finite",
        ));
    }
    if !listener.forward.is_finite() || !listener.up.is_finite() {
        return Err(SoundError::invalid_argument(
            "listener forward and up vectors must be finite",
        ));
    }
    let forward = listener.forward.normalize_or_zero();
    let approximate_up = listener.up.normalize_or_zero();
    let approximate_right = forward.cross(approximate_up);
    if forward == Vec3::ZERO || approximate_right.length_squared() <= 1.0e-8 {
        return Err(SoundError::invalid_argument(
            "listener forward and up vectors must be non-zero and non-collinear",
        ));
    }
    let right = approximate_right.normalize();
    let up = right.cross(forward);
    Ok(ListenerPose {
        position: listener.position,
        forward,
        up,
    })
}

#[cfg(test)]
#[path = "../../tests/unit/acoustics/spatial.rs"]
mod tests;
