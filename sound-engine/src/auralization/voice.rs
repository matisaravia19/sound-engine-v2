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
    /// Previous convolution state retained while an IR update crossfades.
    pub(crate) ir_transition: Option<VoiceIrTransition>,
    /// Most recent IR requested while another transition is still active.
    ///
    /// Coalescing rapid updates prevents an in-progress outgoing filter from
    /// being dropped abruptly when listener poses arrive faster than the ramp.
    pub(crate) pending_impulse_response: Option<Arc<ImpulseResponse>>,
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
            ir_transition: None,
            pending_impulse_response: None,
        }
    }

    /// Returns the unique id assigned to this voice.
    pub fn id(&self) -> VoiceId {
        self.id
    }

    /// Starts a sample-ramped transition from the current IR to `impulse_response`.
    pub(crate) fn transition_impulse_response(
        &mut self,
        impulse_response: Arc<ImpulseResponse>,
        tail_size: usize,
        fft_size: usize,
        output_samples: usize,
        transition_frames: usize,
    ) {
        debug_assert_eq!(self.convolution.tail.len(), tail_size);
        // The overlap tail summarizes every dry sample emitted before this IR
        // update. Both transition branches must retain it; only contributions
        // from new dry samples should diverge between the old and new filters.
        let next =
            VoiceConvolutionState::with_carried_tail(impulse_response, &self.convolution, fft_size);
        let previous = std::mem::replace(&mut self.convolution, next);
        self.ir_transition = Some(VoiceIrTransition {
            previous,
            output_buffer: vec![0.0; output_samples],
            processed_frames: 0,
            total_frames: transition_frames.max(1),
        });
    }

    /// Starts an IR transition or coalesces it behind the active transition.
    pub(crate) fn queue_impulse_response_transition(
        &mut self,
        impulse_response: Arc<ImpulseResponse>,
        tail_size: usize,
        fft_size: usize,
        output_samples: usize,
        transition_frames: usize,
    ) {
        if self.ir_transition.is_some() {
            self.pending_impulse_response = Some(impulse_response);
        } else {
            self.transition_impulse_response(
                impulse_response,
                tail_size,
                fft_size,
                output_samples,
                transition_frames,
            );
        }
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

/// Previous filter state and ramp progress for one live IR replacement.
pub(crate) struct VoiceIrTransition {
    /// Convolution state using the outgoing IR, including its overlap tail.
    pub(crate) previous: VoiceConvolutionState,
    /// Interleaved scratch output rendered by the outgoing filter.
    pub(crate) output_buffer: Vec<f32>,
    /// Output frames already blended since this transition began.
    pub(crate) processed_frames: usize,
    /// Total output frames across which the incoming IR reaches full weight.
    pub(crate) total_frames: usize,
}

impl VoiceConvolutionState {
    /// Allocates zeroed overlap and staging buffers for a registered IR.
    pub(crate) fn new(
        impulse_response: Arc<ImpulseResponse>,
        tail_size: usize,
        fft_size: usize,
    ) -> Self {
        Self {
            impulse_response,
            tail: vec![[0.0, 0.0]; tail_size],
            tail_position: 0,
            complex_cpu_buffer: vec![[0.0, 0.0]; fft_size],
        }
    }

    /// Creates a replacement filter while preserving pre-update overlap history.
    fn with_carried_tail(
        impulse_response: Arc<ImpulseResponse>,
        previous: &Self,
        fft_size: usize,
    ) -> Self {
        Self {
            impulse_response,
            tail: previous.tail.clone(),
            tail_position: previous.tail_position,
            complex_cpu_buffer: vec![[0.0, 0.0]; fft_size],
        }
    }
}
