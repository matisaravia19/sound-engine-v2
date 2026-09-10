//! Sound engine prototype with GPU acceleration for convolution and ray tracing.

/// Acoustic queries and impulse-response generation.
pub mod acoustics;
/// Convolution, sound storage, and spatial audio rendering.
pub mod auralization;
/// Shared configuration, errors, and debug exports.
pub mod core;
pub mod engine;
pub mod gpu;
#[cfg(feature = "playback")]
/// Audio-device playback and real-time output integration.
pub mod playback;
pub mod scene;

pub use engine::{ListenerPose, PlaySpatialSoundRequest, PointSource, SoundEngine, SoundSourceId};
