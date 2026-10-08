//! Per-stream overlap history and impulse-response transitions.

use super::ImpulseResponse;
use super::gpu::GpuConvolution;
use crate::core::config::OutputChannels;
use crate::core::error::{SoundError, SoundResult};
use std::sync::Arc;

/// Number of input blocks used to ramp between live impulse responses.
const IR_TRANSITION_BLOCKS: usize = 4;

/// Independent convolution history for one mono input stream.
///
/// Created by [`super::FftConvolver::create_stream`]. The stream retains its IR
/// spectra and overlap samples independently of sound assets or playback cursors.
/// Use it with a convolver on the same backend and with matching dimensions and
/// output channels. Mutable borrowing excludes concurrent updates and rendering.
pub struct ConvolutionStream {
    /// Current filter and its overlap history.
    current: FilterState,
    /// Previous filter retained until its replacement ramp finishes.
    ir_transition: Option<IrTransition>,
    /// Latest IR queued while another transition is active.
    pending_impulse_response: Option<Arc<ImpulseResponse>>,
    /// Interleaved output layout fixed when the stream is created.
    output_channels: OutputChannels,
}

impl ConvolutionStream {
    /// Allocates zeroed history for a validated IR and convolver layout.
    pub(super) fn new(
        impulse_response: Arc<ImpulseResponse>,
        tail_size: usize,
        fft_size: usize,
        output_channels: OutputChannels,
    ) -> Self {
        Self {
            current: FilterState::new(impulse_response, tail_size, fft_size),
            ir_transition: None,
            pending_impulse_response: None,
            output_channels,
        }
    }

    /// Borrows the current filter for backend, FFT-size, and sample-rate validation.
    pub(super) fn impulse_response(&self) -> &ImpulseResponse {
        &self.current.impulse_response
    }

    /// Checks history dimensions and channel count before any stream mutation.
    pub(super) fn validate_layout(
        &self,
        fft_size: usize,
        tail_size: usize,
        output_channels: OutputChannels,
    ) -> SoundResult<()> {
        if self.current.complex_cpu_buffer.len() != fft_size
            || self.current.tail.len() != tail_size
            || self.output_channels.count() != output_channels.count()
        {
            return Err(SoundError::invalid_argument(
                "Stream dimensions and output channels must match this convolver",
            ));
        }
        Ok(())
    }

    /// Starts a sample-ramped transition from the current IR to `impulse_response`.
    fn transition_impulse_response(
        &mut self,
        impulse_response: Arc<ImpulseResponse>,
        block_size: usize,
    ) {
        // The overlap tail summarizes every dry sample emitted before this IR
        // update. Both transition branches must retain it; only contributions
        // from new dry samples should diverge between the old and new filters.
        let next = FilterState::with_carried_tail(impulse_response, &self.current);
        let previous = std::mem::replace(&mut self.current, next);
        self.ir_transition = Some(IrTransition {
            previous,
            output_buffer: vec![0.0; block_size * self.output_channels.count()],
            processed_frames: 0,
            total_frames: block_size * IR_TRANSITION_BLOCKS,
        });
    }

    /// Starts an IR transition or coalesces it behind the active transition.
    pub(super) fn queue_impulse_response_transition(
        &mut self,
        impulse_response: Arc<ImpulseResponse>,
        block_size: usize,
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
            self.transition_impulse_response(impulse_response, block_size);
        }
    }

    /// Processes a validated mono block and advances any active crossfade.
    pub(super) fn process_block(
        &mut self,
        gpu: &mut GpuConvolution,
        input: &[f32],
        output: &mut [f32],
    ) -> SoundResult<()> {
        // Tail draining closes input even when the FFT has no overlap samples.
        // Reject it before consuming a pending filter or touching GPU scratch.
        if self.current.tail_position.is_some() {
            return Err(SoundError::invalid_argument(
                "convolution stream has already started flushing its tail and cannot process more input blocks",
            ));
        }
        self.start_pending_transition(input.len());
        self.current
            .process_block(gpu, input, output, self.output_channels)?;

        let transition_complete = if let Some(transition) = self.ir_transition.as_mut() {
            transition.previous.process_block(
                gpu,
                input,
                &mut transition.output_buffer,
                self.output_channels,
            )?;
            transition.blend_into(output, self.output_channels.count())
        } else {
            false
        };
        if transition_complete {
            self.ir_transition = None;
        }
        Ok(())
    }

    /// Drains already-computed tails into a validated output block, without GPU work.
    pub(super) fn flush_tail_block(&mut self, output: &mut [f32]) -> bool {
        let mut any_tail = self.current.flush_tail_block(output, self.output_channels);
        let transition_complete = if let Some(transition) = self.ir_transition.as_mut() {
            let previous_has_tail = transition
                .previous
                .flush_tail_block(&mut transition.output_buffer, self.output_channels);
            any_tail |= previous_has_tail;
            transition.blend_into(output, self.output_channels.count()) || !previous_has_tail
        } else {
            false
        };
        if transition_complete {
            self.ir_transition = None;
        }
        // No dry samples remain to seed a newly queued filter. Retaining a
        // pending update here would extend the stream with a silent IR tail.
        self.pending_impulse_response = None;
        any_tail
    }

    /// Begins the newest coalesced IR update after the preceding ramp finishes.
    fn start_pending_transition(&mut self, block_size: usize) {
        if self.ir_transition.is_some() {
            return;
        }
        if let Some(impulse_response) = self.pending_impulse_response.take() {
            self.transition_impulse_response(impulse_response, block_size);
        }
    }
}

/// One filter's shared GPU spectrum, overlap history, and CPU staging buffers.
struct FilterState {
    /// Frequency-domain IR used by this filter.
    impulse_response: Arc<ImpulseResponse>,
    /// Overlap tail carried between processed blocks.
    tail: Vec<[f32; 2]>,
    /// `None` accepts input; `Some` records the cursor after tail draining begins.
    tail_position: Option<usize>,
    /// Complex CPU staging buffer used for FFT input and output.
    complex_cpu_buffer: Vec<[f32; 2]>,
}

/// Previous filter state and ramp progress for one live IR replacement.
struct IrTransition {
    /// Convolution state using the outgoing IR, including its overlap tail.
    previous: FilterState,
    /// Interleaved scratch output rendered by the outgoing filter.
    output_buffer: Vec<f32>,
    /// Output frames already blended since this transition began.
    processed_frames: usize,
    /// Total output frames across which the incoming IR reaches full weight.
    total_frames: usize,
}

impl IrTransition {
    /// Blends equally sized interleaved blocks and reports whether the ramp is complete.
    /// `output_channels` must match the stream's mono or stereo layout.
    fn blend_into(&mut self, incoming: &mut [f32], output_channels: usize) -> bool {
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
    fn new(impulse_response: Arc<ImpulseResponse>, tail_size: usize, fft_size: usize) -> Self {
        Self {
            impulse_response,
            tail: vec![[0.0, 0.0]; tail_size],
            tail_position: None,
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

    /// Packs mono input, convolves it, and retains overlap for the next block.
    fn process_block(
        &mut self,
        gpu: &mut GpuConvolution,
        input: &[f32],
        output: &mut [f32],
        output_channels: OutputChannels,
    ) -> SoundResult<()> {
        // Packing mono as the real component preserves both ears when multiplied
        // by a complex IR whose real/imaginary components encode left/right.
        self.complex_cpu_buffer.fill([0.0, 0.0]);
        for (frame, &sample) in self.complex_cpu_buffer.iter_mut().zip(input) {
            *frame = [sample, 0.0];
        }
        gpu.convolve(&mut self.complex_cpu_buffer, &self.impulse_response.buffer)?;

        // Add all prior contributions before extracting this block's new tail.
        for (frame, previous) in self.complex_cpu_buffer.iter_mut().zip(&self.tail) {
            frame[0] += previous[0];
            frame[1] += previous[1];
        }
        self.tail
            .copy_from_slice(&self.complex_cpu_buffer[input.len()..]);
        copy_samples(
            &self.complex_cpu_buffer[..input.len()],
            output,
            output_channels,
        );
        Ok(())
    }

    /// Copies one tail block and zeroes any unfilled output frames.
    fn flush_tail_block(&mut self, output: &mut [f32], output_channels: OutputChannels) -> bool {
        output.fill(0.0);
        let position = self.tail_position.get_or_insert(0);
        if *position >= self.tail.len() {
            return false;
        }
        let block_size = output.len() / output_channels.count();
        let frames_to_copy = block_size.min(self.tail.len() - *position);
        copy_samples(
            &self.tail[*position..*position + frames_to_copy],
            output,
            output_channels,
        );
        *position += frames_to_copy;
        true
    }
}

/// Unpacks complex time-domain frames into mono or interleaved stereo output.
fn copy_samples(frames: &[[f32; 2]], output: &mut [f32], output_channels: OutputChannels) {
    match output_channels {
        OutputChannels::Mono => {
            for (sample, frame) in output[..frames.len()].iter_mut().zip(frames) {
                *sample = frame[0];
            }
        }
        OutputChannels::Stereo | OutputChannels::Binaural => {
            // The real and imaginary components carry the left and right ears.
            output[..frames.len() * 2].copy_from_slice(bytemuck::cast_slice(frames));
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
