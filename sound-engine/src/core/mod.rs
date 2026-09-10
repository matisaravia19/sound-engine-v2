use glam::Vec3;

pub mod config;
pub mod debug;
pub mod error;

/// Listener pose used by the acoustic renderer.
#[derive(Debug, Clone, Copy)]
pub struct ListenerPose {
    /// Listener position in world meters.
    pub position: Vec3,
    /// Listener forward direction in world space.
    pub forward: Vec3,
    /// Listener up direction in world space.
    pub up: Vec3,
}

/// Point sound source rendered through the current acoustic scene.
#[derive(Debug, Clone, Copy)]
pub struct PointSource {
    /// Source position in world meters.
    pub position: Vec3,
    /// Reference pressure amplitude before acoustic attenuation.
    pub amplitude: f32,
}
