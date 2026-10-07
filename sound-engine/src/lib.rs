//! Spatial audio with Vulkan ray tracing and streaming GPU convolution.
#![warn(missing_docs)]

/// Acoustic queries and impulse-response generation.
pub mod acoustics;
/// Convolution, sound storage, and spatial audio rendering.
pub mod auralization;
/// Shared configuration, errors, and validation.
pub mod core;
/// Synchronous rendering and background playback orchestration.
pub mod engine;
/// Vulkan resource ownership, compute dispatch, and ray tracing.
pub mod gpu;
#[cfg(feature = "playback")]
/// Audio-device playback and real-time output integration.
pub mod playback;
/// Scene geometry, materials, updates, and lazy GPU synchronization.
pub mod scene;

pub use acoustics::{ListenerPose, PointSource};
pub use engine::{PlaySpatialSoundRequest, SoundEngine, SoundSourceId};
