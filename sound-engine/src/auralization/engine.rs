use crate::acoustics::IrSample;
use crate::auralization::assets::{SoundAsset, SoundBank, SoundId};
use crate::auralization::convolver::{ImpulseResponse, PartitionedConvolver};
use crate::auralization::voice::{Voice, VoiceState};
use crate::auralization::{ImpulseResponseId, VoiceId};
use crate::core::config::EngineConfig;
use crate::error::{ErrorCode, SoundError, SoundResult};
use crate::gpu::backend::VkBackend;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Request used to start a dry sound through a registered impulse response.
#[derive(Debug, Clone, Copy)]
pub struct PlaySoundRequest {
    /// Dry asset to render.
    pub sound_id: SoundId,
    /// Impulse response that spatializes the sound.
    pub impulse_response_id: ImpulseResponseId,
    /// Linear dry gain applied before convolution.
    pub gain: f32,
}

impl PlaySoundRequest {
    /// Creates a play request with unit gain.
    pub fn new(sound_id: SoundId, impulse_response_id: ImpulseResponseId) -> Self {
        Self {
            sound_id,
            impulse_response_id,
            gain: 1.0,
        }
    }
}

/// Block-based auralization engine for dry sound assets and acoustic IRs.
///
/// The engine owns decoded sound assets, active voice cursors, and the
/// GPU-backed convolver. It performs no device I/O; callers drive it by
/// requesting sounds and pulling interleaved output blocks.
pub struct AuralizationEngine {
    config: EngineConfig,
    convolver: PartitionedConvolver,
    sound_bank: SoundBank,
    impulse_responses: HashMap<ImpulseResponseId, Arc<ImpulseResponse>>,
    active_voices: HashMap<VoiceId, Voice>,
    next_voice_id: VoiceId,
    dry_block: Vec<f32>,
    wet_block: Vec<f32>,
    finished_voices: Vec<VoiceId>,
}

impl AuralizationEngine {
    /// Creates an engine with an empty sound bank and convolver.
    pub fn new(gpu: Arc<VkBackend>, config: EngineConfig) -> SoundResult<Self> {
        validate_config(config)?;
        let block_size = config.auralization.block_size;
        let output_channels = config.sound.output_channels.count();
        let convolver = PartitionedConvolver::new(gpu, config)?;

        Ok(Self {
            config,
            convolver,
            sound_bank: SoundBank::new(),
            impulse_responses: HashMap::new(),
            active_voices: HashMap::new(),
            next_voice_id: 1,
            dry_block: vec![0.0; block_size],
            wet_block: vec![0.0; block_size * output_channels],
            finished_voices: Vec::new(),
        })
    }

    /// Returns the engine sample rate used for all loaded assets and IRs.
    pub fn sample_rate(&self) -> u32 {
        self.config.sound.sample_rate
    }

    /// Returns the number of frames rendered per block.
    pub fn block_size(&self) -> usize {
        self.config.auralization.block_size
    }

    /// Returns the number of interleaved output channels per frame.
    pub fn output_channels(&self) -> usize {
        self.config.sound.output_channels.count()
    }

    /// Returns an immutable view of the sound bank.
    pub fn sound_bank(&self) -> &SoundBank {
        &self.sound_bank
    }

    /// Returns a mutable view of the sound bank.
    pub fn sound_bank_mut(&mut self) -> &mut SoundBank {
        &mut self.sound_bank
    }

    /// Loads a WAV asset into the sound bank at the engine sample rate.
    pub fn load_wav_sound(&mut self, path: impl AsRef<Path>) -> SoundResult<SoundId> {
        self.sound_bank.load_wav(path, self.config.sound.sample_rate)
    }

    /// Inserts a mono sound asset into the engine sound bank.
    pub fn insert_sound(&mut self, asset: SoundAsset) -> SoundResult<SoundId> {
        if asset.sample_rate() != self.config.sound.sample_rate {
            return Err(SoundError::invalid_argument(format!(
                "sound asset sample rate {} must match engine sample rate {}",
                asset.sample_rate(),
                self.config.sound.sample_rate
            )));
        }
        Ok(self.sound_bank.insert(asset))
    }

    /// Registers an impulse response for later playback.
    pub fn register_impulse_response(&mut self, id: ImpulseResponseId, samples: &[IrSample]) -> SoundResult<()> {
        let impulse_response = self.convolver.create_impulse_response(samples)?;
        self.impulse_responses.insert(id, impulse_response);
        Ok(())
    }

    /// Removes an impulse response from the engine registry.
    ///
    /// Existing voices keep shared ownership of the uploaded GPU buffer and can
    /// continue rendering with the old IR until they finish or are stopped.
    pub fn forget_impulse_response(&mut self, id: ImpulseResponseId) {
        self.impulse_responses.remove(&id);
    }

    /// Starts playing a sound through a registered impulse response.
    pub fn play_sound(&mut self, request: PlaySoundRequest) -> SoundResult<VoiceId> {
        if !request.gain.is_finite() {
            return Err(SoundError::invalid_argument("sound gain must be finite"));
        }
        let asset = self.sound_bank.get(request.sound_id)?;
        if asset.is_empty() {
            return Err(SoundError::invalid_argument("sound asset must not be empty"));
        }

        let impulse_response = self
            .impulse_responses
            .get(&request.impulse_response_id)
            .ok_or_else(|| {
                SoundError::not_found(format!("Unknown impulse response id {}", request.impulse_response_id))
            })?;
        let voice_id = self.next_voice_id;
        self.next_voice_id = self.next_voice_id.saturating_add(1);
        self.active_voices.insert(
            voice_id,
            Voice::new(
                voice_id,
                request.sound_id,
                request.gain,
                impulse_response.clone(),
                self.convolver.tail_size(),
                self.convolver.fft_size(),
            ),
        );

        Ok(voice_id)
    }

    /// Stops an active voice and releases its convolution state.
    pub fn stop_voice(&mut self, voice_id: VoiceId) {
        self.active_voices.remove(&voice_id);
    }

    /// Renders and mixes one output block from all active voices.
    pub fn render_block(&mut self, output_block: &mut [f32]) -> SoundResult<()> {
        let output_channels = self.config.sound.output_channels.count();
        crate::debug_validate!(
            output_block.len() == self.config.auralization.block_size * output_channels,
            ErrorCode::InvalidArgument,
            "output_block length {} must equal block size {} times channel count {}",
            output_block.len(),
            self.config.auralization.block_size,
            output_channels
        );

        output_block.fill(0.0);
        self.finished_voices.clear();

        for (voice_id, voice) in self.active_voices.iter_mut() {
            self.wet_block.fill(0.0);
            match voice.state {
                VoiceState::Playing => {
                    let asset = self.sound_bank.get(voice.sound_id)?;
                    render_asset_block(asset, voice, &mut self.dry_block);
                    self.convolver
                        .process_block(voice, &self.dry_block, &mut self.wet_block)?;
                    if voice.cursor >= asset.len() {
                        voice.state = VoiceState::FlushingTail;
                    }
                    mix_into(output_block, &self.wet_block);
                }
                VoiceState::FlushingTail => {
                    if self.convolver.flush_tail_block(voice, &mut self.wet_block)? {
                        mix_into(output_block, &self.wet_block);
                    } else {
                        self.finished_voices.push(*voice_id);
                    }
                }
            }
        }

        for voice_id in self.finished_voices.drain(..) {
            self.active_voices.remove(&voice_id);
        }

        Ok(())
    }

    /// Returns whether the engine currently has active or tail-flushing voices.
    pub fn has_active_voices(&self) -> bool {
        !self.active_voices.is_empty()
    }

    /// Returns the number of active or tail-flushing voices.
    pub fn active_voice_count(&self) -> usize {
        self.active_voices.len()
    }
}

fn validate_config(config: EngineConfig) -> SoundResult<()> {
    if config.sound.sample_rate == 0 {
        return Err(SoundError::invalid_argument("sample_rate must be > 0"));
    }
    if config.sound.ir_num_samples == 0 {
        return Err(SoundError::invalid_argument("ir_num_samples must be > 0"));
    }
    if config.auralization.block_size == 0 {
        return Err(SoundError::invalid_argument("block_size must be > 0"));
    }
    Ok(())
}

fn render_asset_block(asset: &SoundAsset, voice: &mut Voice, out_mono: &mut [f32]) {
    render_asset_samples(asset, &mut voice.cursor, voice.gain, out_mono);
}

fn render_asset_samples(asset: &SoundAsset, cursor: &mut usize, gain: f32, out_mono: &mut [f32]) {
    out_mono.fill(0.0);
    let available = asset.len().saturating_sub(*cursor);
    let frames_to_copy = out_mono.len().min(available);
    let source = &asset.samples()[*cursor..*cursor + frames_to_copy];
    for (out_sample, source_sample) in out_mono.iter_mut().zip(source.iter().copied()) {
        *out_sample = source_sample * gain;
    }
    *cursor += frames_to_copy;
}

fn mix_into(output: &mut [f32], wet_block: &[f32]) {
    for (output_sample, wet_sample) in output.iter_mut().zip(wet_block.iter().copied()) {
        *output_sample += wet_sample;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_sample_rate() {
        let mut config = test_config();
        config.sound.sample_rate = 0;

        let error = validate_config(config).unwrap_err();

        assert_eq!(error.code(), ErrorCode::InvalidArgument);
    }

    #[test]
    fn renders_asset_block_with_gain_and_padding() {
        let asset = SoundAsset::from_mono_samples(48_000, vec![0.5, -0.25]).unwrap();
        let mut cursor = 0;
        let mut block = vec![1.0; 4];

        render_asset_samples(&asset, &mut cursor, 2.0, &mut block);

        assert_eq!(block, vec![1.0, -0.5, 0.0, 0.0]);
        assert_eq!(cursor, 2);
    }

    fn test_config() -> EngineConfig {
        EngineConfig {
            sound: crate::core::config::SoundConfig {
                output_channels: crate::core::config::OutputChannels::Stereo,
                sample_rate: 48_000,
                ir_num_samples: 128,
            },
            acoustics: crate::core::config::AcousticsConfig {
                listener_half_extent: glam::Vec3::splat(0.2),
                rays_per_query: 1,
                max_bounces: 0,
            },
            cache: crate::core::config::IrCacheConfig::default(),
            auralization: crate::core::config::AuralizationConfig { block_size: 64 },
        }
    }
}
