use crate::acoustics::{GpuIrSnapshot, IrSample};
use crate::auralization::assets::{SoundAsset, SoundBank, SoundId};
use crate::auralization::convolver::{ConvolutionStream, FftConvolver, ImpulseResponse};
use crate::core::config::{AuralizationConfig, SoundConfig};
use crate::core::error::ErrorCode;
use crate::core::error::{SoundError, SoundResult};
use crate::gpu::backend::VkBackend;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Opaque identifier for a voice in the auralization engine that started it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VoiceId(
    /// Monotonic engine-local value; never reused after playback ends.
    pub(crate) u64,
);

impl std::fmt::Display for VoiceId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

/// Request used to start a dry sound through a prepared impulse response.
#[derive(Clone)]
pub struct PlaySoundRequest {
    /// Dry asset to render.
    pub sound_id: SoundId,
    /// Prepared filter shared with the voice for its entire playback lifetime.
    pub impulse_response: Arc<ImpulseResponse>,
    /// Linear dry gain applied before convolution.
    pub gain: f32,
}

impl PlaySoundRequest {
    /// Creates a play request with unit gain.
    pub fn new(sound_id: SoundId, impulse_response: Arc<ImpulseResponse>) -> Self {
        Self {
            sound_id,
            impulse_response,
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
    sound: SoundConfig,
    sound_bank: SoundBank,
    active_voices: HashMap<VoiceId, ActiveVoice>,
    next_voice_id: u64,
    dry_block: Vec<f32>,
    wet_block: Vec<f32>,
    finished_voices: Vec<VoiceId>,
    // Drop voices before the convolver, whose backend Arc retains the device
    // through teardown of its plans, scratch buffers, and pipelines.
    convolver: FftConvolver,
}

/// Retains immutable dry samples even if the application's sound bank removes them.
struct ActiveVoice {
    /// Dry samples retained even after removal from the sound bank.
    asset: SoundAsset,
    /// Next dry frame to render; reaching the asset end starts tail flushing.
    cursor: usize,
    /// Linear gain applied before convolution.
    gain: f32,
    /// Filter history independent of the asset and its playback cursor.
    convolution: ConvolutionStream,
}

impl AuralizationEngine {
    /// Creates an engine with an empty sound bank and convolver.
    pub fn new(
        gpu: Arc<VkBackend>,
        sound: SoundConfig,
        config: AuralizationConfig,
    ) -> SoundResult<Self> {
        let block_size = config.block_size;
        let output_channels = sound.output_channels.count();
        let convolver = FftConvolver::new(gpu, sound, config)?;

        Ok(Self {
            sound,
            sound_bank: SoundBank::new(),
            active_voices: HashMap::new(),
            next_voice_id: 1,
            dry_block: vec![0.0; block_size],
            wet_block: vec![0.0; block_size * output_channels],
            finished_voices: Vec::new(),
            convolver,
        })
    }

    /// Returns the engine sample rate used for all loaded assets and IRs.
    pub fn sample_rate(&self) -> u32 {
        self.sound.sample_rate
    }

    /// Returns the number of frames rendered per block.
    pub fn block_size(&self) -> usize {
        self.dry_block.len()
    }

    /// Returns the number of interleaved output channels per frame.
    pub fn output_channels(&self) -> usize {
        self.sound.output_channels.count()
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
        self.sound_bank.load_wav(path, self.sound.sample_rate)
    }

    /// Inserts a mono sound asset into the engine sound bank.
    pub fn insert_sound(&mut self, asset: SoundAsset) -> SoundResult<SoundId> {
        if asset.sample_rate() != self.sound.sample_rate {
            return Err(SoundError::invalid_argument(format!(
                "sound asset sample rate {} must match engine sample rate {}",
                asset.sample_rate(),
                self.sound.sample_rate
            )));
        }
        self.sound_bank.insert(asset)
    }

    /// Borrows a valid dry asset so spatial callers can reject it before preparing GPU IRs.
    pub(crate) fn playable_sound(&self, sound_id: SoundId) -> SoundResult<&SoundAsset> {
        let asset = self.sound_bank.get(sound_id)?;
        if asset.sample_rate() != self.sound.sample_rate {
            return Err(SoundError::invalid_argument(
                "Sound sample rate must match the engine",
            ));
        }
        if asset.is_empty() {
            return Err(SoundError::invalid_argument(
                "sound asset must not be empty",
            ));
        }
        Ok(asset)
    }

    /// Uploads and transforms CPU samples into a shareable playback filter.
    ///
    /// The returned owner keeps its GPU storage alive independently of this engine.
    pub fn prepare_impulse_response(
        &mut self,
        samples: &[IrSample],
    ) -> SoundResult<Arc<ImpulseResponse>> {
        self.convolver.create_impulse_response(samples)
    }

    /// Copies and transforms a GPU time-domain IR into a shareable playback filter.
    ///
    /// Voices retain the result; callers may drop their own reference after playing it.
    pub fn prepare_impulse_response_from_gpu(
        &mut self,
        snapshot: &GpuIrSnapshot,
    ) -> SoundResult<Arc<ImpulseResponse>> {
        self.convolver.create_impulse_response_from_gpu_ir(snapshot)
    }

    /// Replaces the impulse response used by an active voice.
    ///
    /// The voice keeps its dry cursor and crossfades from the outgoing filter,
    /// including its overlap tail, to the new uploaded IR spectrum.
    pub fn update_voice_impulse_response(
        &mut self,
        voice_id: VoiceId,
        impulse_response: Arc<ImpulseResponse>,
    ) -> SoundResult<()> {
        let voice = self
            .active_voices
            .get_mut(&voice_id)
            .ok_or_else(|| SoundError::not_found(format!("Unknown voice id {}", voice_id)))?;

        self.convolver
            .update_impulse_response(&mut voice.convolution, impulse_response)
    }

    /// Starts playing a sound through a compatible prepared impulse response.
    ///
    /// The filter must share this engine's backend, FFT size, and sample rate.
    pub fn play_sound(&mut self, request: PlaySoundRequest) -> SoundResult<VoiceId> {
        if !request.gain.is_finite() {
            return Err(SoundError::invalid_argument("sound gain must be finite"));
        }
        let asset = self.playable_sound(request.sound_id)?.clone();

        let convolution = self.convolver.create_stream(request.impulse_response)?;
        let voice_id = VoiceId(crate::core::id::allocate_id(
            &mut self.next_voice_id,
            "Voice",
        )?);
        self.active_voices.insert(
            voice_id,
            ActiveVoice {
                asset,
                cursor: 0,
                gain: request.gain,
                convolution,
            },
        );

        Ok(voice_id)
    }

    /// Stops an active voice and releases its convolution state.
    pub fn stop_voice(&mut self, voice_id: VoiceId) {
        self.active_voices.remove(&voice_id);
    }

    /// Returns whether the given voice is still active or flushing its tail.
    pub fn has_voice(&self, voice_id: VoiceId) -> bool {
        self.active_voices.contains_key(&voice_id)
    }

    /// Checks the output shape before spatial updates or voice cursors can advance.
    pub(crate) fn validate_output_block(&self, output_block: &[f32]) -> SoundResult<()> {
        crate::validate!(
            output_block.len() == self.wet_block.len(),
            ErrorCode::InvalidArgument,
            "output_block length {} must equal block size {} times channel count {}",
            output_block.len(),
            self.block_size(),
            self.output_channels()
        );
        Ok(())
    }

    /// Renders and mixes one output block from all active voices.
    ///
    /// The provided `output_block` must have length equal to `block_size * output_channels`.
    /// This method will zero-fill the buffer and then accumulate (mix) the rendered
    /// audio from all active voices into it. For multichannel output (e.g. stereo),
    /// samples are written interleaved (e.g. L,R,L,R...).
    pub fn render_block(&mut self, output_block: &mut [f32]) -> SoundResult<()> {
        self.validate_output_block(output_block)?;

        output_block.fill(0.0);
        self.finished_voices.clear();

        for (voice_id, voice) in self.active_voices.iter_mut() {
            if voice.cursor < voice.asset.len() {
                render_asset_samples(
                    &voice.asset,
                    &mut voice.cursor,
                    voice.gain,
                    &mut self.dry_block,
                );
                self.convolver.process_block(
                    &mut voice.convolution,
                    &self.dry_block,
                    &mut self.wet_block,
                )?;
            } else if !self
                .convolver
                .flush_tail_block(&mut voice.convolution, &mut self.wet_block)?
            {
                self.finished_voices.push(*voice_id);
                continue;
            }
            mix_into(output_block, &self.wet_block);
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
#[path = "../../tests/unit/auralization/engine.rs"]
mod tests;
