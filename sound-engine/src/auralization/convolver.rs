//! Filter preparation and validated access to GPU convolution streams.

mod gpu;
mod stream;

pub use stream::ConvolutionStream;

use crate::acoustics::{GpuIrSnapshot, IrSample};
use crate::core::config::{AuralizationConfig, SoundConfig};
use crate::core::error::{ErrorCode, SoundError, SoundResult};
use crate::gpu::backend::VkBackend;
use crate::gpu::memory::BufferHandle;
use gpu::GpuConvolution;
use std::sync::Arc;

/// GPU-backed overlap-add FFT convolver for mono streams and stereo impulse responses.
///
/// Stereo IRs are packed as complex time-domain samples (`left + i * right`)
/// before the IR FFT. A mono dry signal can then produce both ears with the
/// same FFT, multiply, and inverse FFT used for mono convolution.
///
/// Each block is padded to fit the full IR. Mutable operations reserve exclusive
/// access to the reusable signal and IR scratch buffers across GPU submissions.
pub struct FftConvolver {
    /// Audio layout and sample rate shared by dry sounds and impulse responses.
    sound: SoundConfig,
    /// Number of input frames consumed in one convolution block.
    block_size: usize,
    /// FFT size: next power of two of (block_size + ir_samples - 1).
    fft_size: usize,
    /// Overlap tail length: fft_size - block_size samples preserved between blocks.
    tail_size: usize,
    /// Owns Vulkan resources and serializes use of reusable FFT scratch storage.
    gpu: GpuConvolution,
}

/// Uploaded frequency-domain impulse response produced by the convolver.
///
/// The wrapper owns the transformed GPU buffer and its sample-rate contract.
/// Shared ownership lets streams retain a filter after the application or its
/// preparing convolver releases it.
pub struct ImpulseResponse {
    /// GPU storage buffer containing the FFT-transformed stereo IR.
    buffer: BufferHandle,
    /// Sample rate used to interpret the original time-domain samples.
    sample_rate: u32,
}

impl FftConvolver {
    /// Creates reusable signal/IR FFT plans and the complex multiply pipeline.
    pub fn new(
        gpu: Arc<VkBackend>,
        sound: SoundConfig,
        config: AuralizationConfig,
    ) -> SoundResult<Self> {
        let fft_size = config.fft_size(sound)?;
        let tail_size = fft_size - config.block_size;
        let gpu = GpuConvolution::new(gpu, fft_size)?;

        Ok(Self {
            sound,
            block_size: config.block_size,
            fft_size,
            tail_size,
            gpu,
        })
    }

    /// Uploads and transforms an IR, reserving padding for one full input block.
    pub fn create_impulse_response(
        &mut self,
        samples: &[IrSample],
    ) -> SoundResult<Arc<ImpulseResponse>> {
        validate_ir_length(samples.len(), self.ir_capacity())?;

        let buffer = self.gpu.prepare_ir(samples)?;

        Ok(Arc::new(ImpulseResponse {
            buffer,
            sample_rate: self.sound.sample_rate,
        }))
    }

    /// Copies a GPU time-domain IR into convolver storage and transforms it into frequency-domain form.
    pub fn create_impulse_response_from_gpu_ir(
        &mut self,
        snapshot: &GpuIrSnapshot,
    ) -> SoundResult<Arc<ImpulseResponse>> {
        let source_buffer =
            validate_gpu_ir_snapshot(snapshot, self.ir_capacity(), self.sound.sample_rate)?;

        let buffer = self
            .gpu
            .prepare_gpu_ir(source_buffer, snapshot.sample_count())?;

        Ok(Arc::new(ImpulseResponse {
            buffer,
            sample_rate: self.sound.sample_rate,
        }))
    }

    /// Returns the FFT size used for signal and IR buffers.
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Creates independent zeroed history for an IR uploaded by a compatible convolver.
    pub fn create_stream(
        &self,
        impulse_response: Arc<ImpulseResponse>,
    ) -> SoundResult<ConvolutionStream> {
        self.validate_impulse_response(&impulse_response)?;
        Ok(ConvolutionStream::new(
            impulse_response,
            self.tail_size,
            self.fft_size,
            self.sound.output_channels,
        ))
    }

    /// Crossfades this stream to a new IR without discarding its overlap history.
    /// Rapid updates coalesce until the active transition finishes.
    pub fn update_impulse_response(
        &self,
        stream: &mut ConvolutionStream,
        impulse_response: Arc<ImpulseResponse>,
    ) -> SoundResult<()> {
        self.validate_stream(stream)?;
        self.validate_impulse_response(&impulse_response)?;
        stream.queue_impulse_response_transition(impulse_response, self.block_size);
        Ok(())
    }

    fn validate_impulse_response(&self, impulse_response: &ImpulseResponse) -> SoundResult<()> {
        self.gpu.validate_spectrum(&impulse_response.buffer)?;
        if impulse_response.sample_rate != self.sound.sample_rate {
            return Err(SoundError::invalid_argument(
                "Impulse response sample rate must match this convolver",
            ));
        }
        Ok(())
    }

    fn validate_stream(&self, stream: &ConvolutionStream) -> SoundResult<()> {
        self.validate_impulse_response(stream.impulse_response())?;
        stream.validate_layout(self.fft_size, self.tail_size, self.sound.output_channels)
    }

    fn ir_capacity(&self) -> usize {
        // Linear convolution needs N + M - 1 samples; the remaining FFT bins
        // must stay zero so the last input samples cannot wrap into the head.
        self.fft_size - self.block_size + 1
    }

    /// Processes one mono input block using this stream's filter and overlap history.
    /// Returns an error after tail flushing begins, including when the tail is empty.
    pub fn process_block(
        &mut self,
        stream: &mut ConvolutionStream,
        input_block: &[f32],
        output_block: &mut [f32],
    ) -> SoundResult<()> {
        self.validate_block_lengths(input_block, output_block)?;
        self.validate_stream(stream)?;
        stream.process_block(&mut self.gpu, input_block, output_block)
    }

    /// Drains the next block of already-computed overlap tail samples.
    ///
    /// Returns `true` when `output_block` contains tail samples. Returns `false`
    /// once the stream has no pending overlap tail. No additional convolution is
    /// performed. Once flushing begins, create a new stream to process more input.
    pub fn flush_tail_block(
        &self,
        stream: &mut ConvolutionStream,
        output_block: &mut [f32],
    ) -> SoundResult<bool> {
        let output_channels = self.sound.output_channels.count();
        crate::validate!(
            output_block.len() == self.block_size * output_channels,
            ErrorCode::InvalidArgument,
            "output_block length {} must equal block size {} times channel count {}",
            output_block.len(),
            self.block_size,
            output_channels,
        );

        self.validate_stream(stream)?;
        Ok(stream.flush_tail_block(output_block))
    }

    /// Validates that input and output block sizes match expected channel layout and FFT capacity.
    fn validate_block_lengths(&self, input_block: &[f32], output_block: &[f32]) -> SoundResult<()> {
        crate::validate!(
            input_block.len() == self.block_size,
            ErrorCode::InvalidArgument,
            "input_block length {} must equal block size {}",
            input_block.len(),
            self.block_size
        );

        let output_channels = self.sound.output_channels.count();
        crate::validate!(
            output_block.len() == input_block.len() * output_channels,
            ErrorCode::InvalidArgument,
            "output_block length {} must equal {} times input_block length {}",
            output_block.len(),
            output_channels,
            input_block.len(),
        );

        Ok(())
    }
}

fn validate_ir_length(sample_count: usize, capacity: usize) -> SoundResult<()> {
    crate::validate!(
        sample_count > 0 && sample_count <= capacity,
        ErrorCode::InvalidArgument,
        "Impulse response length {} must be between 1 and {} samples for linear convolution",
        sample_count,
        capacity
    );
    Ok(())
}

fn validate_gpu_ir_snapshot(
    snapshot: &GpuIrSnapshot,
    capacity: usize,
    sample_rate: u32,
) -> SoundResult<&BufferHandle> {
    crate::validate!(
        snapshot.sample_rate() == sample_rate,
        ErrorCode::InvalidArgument,
        "GPU impulse response sample rate {} must match convolver sample rate {}",
        snapshot.sample_rate(),
        sample_rate
    );
    validate_ir_length(snapshot.sample_count() as usize, capacity)?;
    Ok(snapshot.buffer())
}

#[cfg(test)]
#[path = "../../tests/unit/auralization/convolver.rs"]
mod tests;
