pub mod convolver;
mod engine;

use std::error::Error;

// pub(crate) use engine::AuralizationEngine;

pub const BLOCK_SIZE: u64 = 1024;
pub const IR_SIZE: u64 = 44100;
pub const SAMPLE_RATE: u32 = 44100;
pub const FFT_SIZE: u64 = (BLOCK_SIZE + IR_SIZE - 1).next_power_of_two();
pub const TAIL_SIZE: usize = (FFT_SIZE - BLOCK_SIZE) as usize;

pub type AudioError = Box<dyn Error + Send + Sync>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutputChannels {
    Stereo,
    Binaural,
}

impl OutputChannels {
    pub(crate) fn count(self) -> usize {
        match self {
            OutputChannels::Stereo | OutputChannels::Binaural => 2,
        }
    }
}

pub(crate) struct AuralizationConfig {
    pub sample_rate: u32,
    pub block_size: usize,
    pub output_channels: OutputChannels,
}

impl AuralizationConfig {
    pub(crate) fn validate(&self) -> Result<(), AudioError> {
        if self.sample_rate == 0 {
            return Err(std::io::Error::other("sample_rate must be > 0").into());
        }
        if self.block_size == 0 {
            return Err(std::io::Error::other("block_size must be > 0").into());
        }
        Ok(())
    }
}

pub(crate) type SoundId = u32;
pub type ImpulseResponseId = u64;
pub type VoiceId = u64;

#[derive(Debug, Clone)]
pub(crate) struct IrSnapshot {
    pub sample_rate: u32,
    pub samples: Vec<f32>,
    pub energy: f32,
}

pub(crate) trait SoundBankReader {
    fn render_voice_block(
        &self,
        sound: SoundId,
        voice: VoiceId,
        out_mono: &mut [f32],
    ) -> Result<VoiceRenderResult, AudioError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VoiceRenderResult {
    Active,
    Finished,
}
