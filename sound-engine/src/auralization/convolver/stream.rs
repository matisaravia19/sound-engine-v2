//! Per-stream overlap history and impulse-response transitions.

use super::ImpulseResponse;
use std::sync::Arc;

/// Independent convolution history for one mono input stream.
///
/// Created by [`super::FftConvolver::create_stream`]. The stream retains its IR
/// spectra and overlap samples independently of sound assets or playback cursors.
/// Use it with a convolver on the same backend and with matching dimensions and
/// output channels. Mutable borrowing excludes concurrent updates and rendering.
pub struct ConvolutionStream {
    /// Current filter and its overlap history.
    pub(super) current: FilterState,
    /// Previous filter retained until its replacement ramp finishes.
    pub(super) ir_transition: Option<IrTransition>,
    /// Latest IR queued while another transition is active.
    pub(super) pending_impulse_response: Option<Arc<ImpulseResponse>>,
    /// Interleaved output layout fixed when the stream is created.
    pub(super) output_channels: usize,
}

impl ConvolutionStream {
    /// Allocates zeroed history for a validated IR and convolver layout.
    pub(super) fn new(
        impulse_response: Arc<ImpulseResponse>,
        tail_size: usize,
        fft_size: usize,
        output_channels: usize,
    ) -> Self {
        Self {
            current: FilterState::new(impulse_response, tail_size, fft_size),
            ir_transition: None,
            pending_impulse_response: None,
            output_channels,
        }
    }

    /// Starts a sample-ramped transition from the current IR to `impulse_response`.
    pub(super) fn transition_impulse_response(
        &mut self,
        impulse_response: Arc<ImpulseResponse>,
        output_samples: usize,
        transition_frames: usize,
    ) {
        // The overlap tail summarizes every dry sample emitted before this IR
        // update. Both transition branches must retain it; only contributions
        // from new dry samples should diverge between the old and new filters.
        let next = FilterState::with_carried_tail(impulse_response, &self.current);
        let previous = std::mem::replace(&mut self.current, next);
        self.ir_transition = Some(IrTransition {
            previous,
            output_buffer: vec![0.0; output_samples],
            processed_frames: 0,
            total_frames: transition_frames.max(1),
        });
    }

    /// Starts an IR transition or coalesces it behind the active transition.
    pub(super) fn queue_impulse_response_transition(
        &mut self,
        impulse_response: Arc<ImpulseResponse>,
        output_samples: usize,
        transition_frames: usize,
    ) {
        // Reaffirming the incoming filter cancels a queued replacement without
        // restarting the active ramp or duplicating an unchanged filter.
        if Arc::ptr_eq(&self.current.impulse_response, &impulse_response) {
            self.pending_impulse_response = None;
            return;
        }
        if self.ir_transition.is_some() {
            self.pending_impulse_response = Some(impulse_response);
        } else {
            self.transition_impulse_response(impulse_response, output_samples, transition_frames);
        }
    }
}

/// One filter's shared GPU spectrum, overlap history, and CPU staging buffers.
pub(super) struct FilterState {
    /// Frequency-domain IR used by this filter.
    pub(super) impulse_response: Arc<ImpulseResponse>,
    /// Overlap tail carried between processed blocks.
    pub(super) tail: Vec<[f32; 2]>,
    /// Current read position while draining `tail`.
    pub(super) tail_position: usize,
    /// Complex CPU staging buffer used for FFT input and output.
    pub(super) complex_cpu_buffer: Vec<[f32; 2]>,
}

/// Previous filter state and ramp progress for one live IR replacement.
pub(super) struct IrTransition {
    /// Convolution state using the outgoing IR, including its overlap tail.
    pub(super) previous: FilterState,
    /// Interleaved scratch output rendered by the outgoing filter.
    pub(super) output_buffer: Vec<f32>,
    /// Output frames already blended since this transition began.
    processed_frames: usize,
    /// Total output frames across which the incoming IR reaches full weight.
    total_frames: usize,
}

impl IrTransition {
    /// Blends equally sized interleaved blocks and reports whether the ramp is complete.
    /// `output_channels` must match the stream's mono or stereo layout.
    pub(super) fn blend_into(&mut self, incoming: &mut [f32], output_channels: usize) -> bool {
        blend_ir_transition(
            incoming,
            &self.output_buffer,
            output_channels,
            self.processed_frames,
            self.total_frames,
        );
        self.processed_frames = self
            .processed_frames
            .saturating_add(incoming.len() / output_channels);
        self.processed_frames >= self.total_frames
    }
}

impl FilterState {
    /// Allocates zeroed overlap and staging buffers for a prepared IR.
    pub(super) fn new(
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
    fn with_carried_tail(impulse_response: Arc<ImpulseResponse>, previous: &Self) -> Self {
        Self {
            impulse_response,
            tail: previous.tail.clone(),
            tail_position: previous.tail_position,
            complex_cpu_buffer: vec![[0.0, 0.0]; previous.complex_cpu_buffer.len()],
        }
    }
}

/// Blends an outgoing block into an incoming block with a sample-continuous ramp.
fn blend_ir_transition(
    incoming: &mut [f32],
    outgoing: &[f32],
    output_channels: usize,
    processed_frames: usize,
    total_frames: usize,
) {
    debug_assert!(output_channels > 0);
    debug_assert_eq!(incoming.len(), outgoing.len());
    for (frame_index, (incoming_frame, outgoing_frame)) in incoming
        .chunks_exact_mut(output_channels)
        .zip(outgoing.chunks_exact(output_channels))
        .enumerate()
    {
        let incoming_weight =
            ((processed_frames + frame_index + 1) as f32 / total_frames.max(1) as f32).min(1.0);
        let outgoing_weight = 1.0 - incoming_weight;
        for (incoming_sample, outgoing_sample) in incoming_frame.iter_mut().zip(outgoing_frame) {
            *incoming_sample =
                *outgoing_sample * outgoing_weight + *incoming_sample * incoming_weight;
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/auralization/convolver/stream.rs"]
mod tests;
