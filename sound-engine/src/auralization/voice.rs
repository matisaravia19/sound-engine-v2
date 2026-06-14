use crate::auralization::assets::SoundId;
use crate::auralization::convolver::ImpulseResponse;
use std::sync::Arc;

/// Identifier for a currently playing voice.
pub type VoiceId = u64;

/// Runtime state for one active sound voice.
///
/// The auralization engine owns voices and advances their dry playback cursor.
/// The convolver mutates only the embedded convolution state while processing
/// and flushing overlap samples.
pub struct Voice {
    /// Unique id assigned by the auralization engine.
    pub(crate) id: VoiceId,
    /// Dry sound asset rendered by this voice.
    pub(crate) sound_id: SoundId,
    /// Current dry sample cursor within the asset.
    pub(crate) cursor: usize,
    /// Linear dry gain applied before convolution.
    pub(crate) gain: f32,
    /// High-level playback phase owned by the auralization engine.
    pub(crate) state: VoiceState,
    /// Per-voice overlap and FFT scratch state used by the convolver.
    pub(crate) convolution: VoiceConvolutionState,
}

impl Voice {
    /// Creates a voice with playback starting at the beginning of the asset.
    pub fn new(
        id: VoiceId,
        sound_id: SoundId,
        gain: f32,
        impulse_response: Arc<ImpulseResponse>,
        tail_size: usize,
        fft_size: usize,
    ) -> Self {
        Self {
            id,
            sound_id,
            cursor: 0,
            gain,
            state: VoiceState::Playing,
            convolution: VoiceConvolutionState::new(impulse_response, tail_size, fft_size),
        }
    }

    /// Returns the unique id assigned to this voice.
    pub fn id(&self) -> VoiceId {
        self.id
    }
}

/// High-level playback phase for an active voice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VoiceState {
    /// Dry samples are still being fed into the convolver.
    Playing,
    /// Dry samples are exhausted and only the convolution tail remains.
    FlushingTail,
}

/// Per-voice convolution state.
///
/// The state keeps a shared reference to the uploaded IR spectrum plus CPU
/// buffers for overlap-add and GPU transfer. It is owned by the voice so the
/// convolver does not need its own voice registry or synchronization.
pub(crate) struct VoiceConvolutionState {
    /// Frequency-domain IR used when convolving this voice.
    pub(crate) impulse_response: Arc<ImpulseResponse>,
    /// Overlap tail carried between processed blocks.
    pub(crate) tail: Vec<[f32; 2]>,
    /// Current read position while draining `tail`.
    pub(crate) tail_position: usize,
    /// Complex CPU staging buffer used for FFT input and output.
    pub(crate) complex_cpu_buffer: Vec<[f32; 2]>,
}

impl VoiceConvolutionState {
    /// Allocates zeroed overlap and staging buffers for a registered IR.
    pub(crate) fn new(impulse_response: Arc<ImpulseResponse>, tail_size: usize, fft_size: usize) -> Self {
        Self {
            impulse_response,
            tail: vec![[0.0, 0.0]; tail_size],
            tail_position: 0,
            complex_cpu_buffer: vec![[0.0, 0.0]; fft_size],
        }
    }
}
