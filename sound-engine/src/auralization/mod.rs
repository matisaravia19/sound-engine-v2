pub mod assets;
pub mod convolver;
mod engine;
mod voice;

pub use assets::{SoundAsset, SoundBank, SoundId};
pub use engine::{AuralizationEngine, PlaySoundRequest};
pub use voice::{Voice, VoiceId};

/// Identifier for an impulse response registered in the convolver.
pub type ImpulseResponseId = u64;
