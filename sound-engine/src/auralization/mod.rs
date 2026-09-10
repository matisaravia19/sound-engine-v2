/// Owned sound assets and stable sound identifiers.
pub mod assets;
/// Streaming GPU convolution and impulse-response snapshots.
pub mod convolver;
mod engine;
mod voice;

pub use assets::{SoundAsset, SoundBank, SoundId};
pub use engine::{AuralizationEngine, PlaySoundRequest};
pub use voice::{Voice, VoiceId};

/// Number of audio blocks used to ramp between live spatial IR updates.
pub(crate) const IR_TRANSITION_BLOCKS: usize = 4;

/// Identifier for an impulse response registered in the convolver.
pub type ImpulseResponseId = u64;
