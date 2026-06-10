pub mod convolver;
mod engine;

use crate::error::{SoundError, SoundResult};

// pub(crate) use engine::AuralizationEngine;

/// Number of input samples processed per audio block.
pub const BLOCK_SIZE: u64 = 1024;
/// Default impulse-response length in samples.
pub const IR_SIZE: u64 = 44100;
/// Default audio sample rate in Hz.
pub const SAMPLE_RATE: u32 = 44100;
/// FFT length used for partitioned convolution.
pub const FFT_SIZE: u64 = (BLOCK_SIZE + IR_SIZE - 1).next_power_of_two();
/// Number of overlap samples carried between audio blocks.
pub const TAIL_SIZE: usize = (FFT_SIZE - BLOCK_SIZE) as usize;

/// Output channel layout used by the auralization engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutputChannels {
    /// Two-channel stereo output.
    Stereo,
    /// Two-channel binaural output.
    Binaural,
}

impl OutputChannels {
    /// Returns the number of interleaved output channels.
    pub(crate) fn count(self) -> usize {
        match self {
            OutputChannels::Stereo | OutputChannels::Binaural => 2,
        }
    }
}

/// Runtime configuration for the auralization engine.
pub(crate) struct AuralizationConfig {
    /// Audio sample rate in Hz.
    pub sample_rate: u32,
    /// Number of frames processed per callback block.
    pub block_size: usize,
    /// Output channel layout.
    pub output_channels: OutputChannels,
}

impl AuralizationConfig {
    /// Validates basic non-zero audio parameters.
    pub(crate) fn validate(&self) -> SoundResult<()> {
        if self.sample_rate == 0 {
            return Err(SoundError::invalid_argument("sample_rate must be > 0"));
        }
        if self.block_size == 0 {
            return Err(SoundError::invalid_argument("block_size must be > 0"));
        }
        Ok(())
    }
}

/// Identifier for a playable dry sound asset.
pub(crate) type SoundId = u32;
/// Identifier for an impulse response registered in the convolver.
pub type ImpulseResponseId = u64;
/// Identifier for a currently playing voice.
pub type VoiceId = u64;

/// CPU-side impulse-response data ready for convolution.
#[derive(Debug, Clone)]
pub(crate) struct IrSnapshot {
    /// Sample rate used to generate the IR.
    pub sample_rate: u32,
    /// Mono IR samples.
    pub samples: Vec<f32>,
    /// Total IR energy used for diagnostics or normalization decisions.
    pub energy: f32,
}

/// Source of dry mono audio blocks for active voices.
pub(crate) trait SoundBankReader {
    /// Renders one voice into `out_mono` and reports whether it is still active.
    fn render_voice_block(
        &self,
        sound: SoundId,
        voice: VoiceId,
        out_mono: &mut [f32],
    ) -> SoundResult<VoiceRenderResult>;
}

/// Result of rendering one dry voice block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VoiceRenderResult {
    /// Voice has more samples to render.
    Active,
    /// Voice finished during or before this block.
    Finished,
}
