use crate::acoustics::{AcousticPipeline, AcousticQuery, IrCache, IrCacheConfig, IrCacheQuery, IrSnapshot};
use crate::auralization::convolver::PartitionedConvolver;
use crate::auralization::{ImpulseResponseId, VoiceId};
use crate::core::config::EngineConfig;
use crate::core::error::SoundResult;
use crate::gpu::backend::VkBackend;
use crate::scene::{SceneDescription, SceneManager, SceneUpdates};
use glam::Vec3;
use std::sync::Arc;

pub mod config;
pub mod debug;
pub mod error;

/// Listener pose used by the acoustic renderer.
#[derive(Debug, Clone, Copy)]
pub struct ListenerPose {
    /// Listener position in world meters.
    pub position: Vec3,
    /// Listener right-ear direction in world space.
    pub right: Vec3,
}

/// Point sound source rendered through the current acoustic scene.
#[derive(Debug, Clone, Copy)]
pub struct PointSource {
    /// Source position in world meters.
    pub position: Vec3,
    /// Reference pressure amplitude before acoustic attenuation.
    pub amplitude: f32,
}
