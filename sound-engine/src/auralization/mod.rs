/// Owned sound assets and stable sound identifiers.
pub mod assets;
/// Streaming GPU convolution and impulse-response snapshots.
pub mod convolver;
mod engine;

pub use assets::{SoundAsset, SoundBank, SoundId};
pub use engine::{AuralizationEngine, PlaySoundRequest, VoiceId};
