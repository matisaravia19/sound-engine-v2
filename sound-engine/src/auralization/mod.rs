mod convolver;
mod engine;
use std::error::Error;
use std::sync::Arc;

pub(crate) use engine::AuralizationEngine;

pub(crate) type AudioError = Box<dyn Error + Send + Sync>;

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

#[derive(Debug, Clone)]
pub(crate) struct IrSnapshot {
    pub sample_rate: u32,
    pub samples: Vec<f32>,
    pub energy: f32,
}

pub(crate) type VoiceId = u64;

pub(crate) struct VoiceState {
    pub voice_id: VoiceId,
    pub sound: SoundId,
    pub ir: Arc<IrSnapshot>,
}

pub(crate) type VoiceTable = Vec<VoiceState>;

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
