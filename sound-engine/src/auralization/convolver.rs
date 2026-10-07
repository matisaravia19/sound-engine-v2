mod stream;

pub use stream::ConvolutionStream;

use crate::acoustics::{GpuIrSnapshot, IrSample};
use crate::core::config::{AuralizationConfig, OutputChannels, SoundConfig};
use crate::core::error::{ErrorCode, SoundError, SoundResult};
use crate::gpu::StorageBufferWrite;
use crate::gpu::backend::VkBackend;
use crate::gpu::compute::{
    BufferBarrierSpec, ComputePipelineSpec, DispatchSpec, PipelineId, PushConstantSpec,
};
use crate::gpu::memory::BufferHandle;
use crate::gpu::shader::ShaderStage;
use ash::vk;
use std::sync::Arc;
use stream::FilterState;
use vkfft_rs::plan::{DeviceHandles, FFTPlan, FFTPlanBuilder};

const MULTIPLY_SHADER_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/auralization/shaders/multiply.comp.glsl"
);

/// Number of input blocks used to ramp between live impulse responses.
const IR_TRANSITION_BLOCKS: usize = 4;

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
    /// Shared FFT plan for transforming mono input signals to frequency-domain.
    signal_plan: FFTPlan,
    /// Reusable GPU storage buffer for signal FFT input and output.
    signal_buffer: BufferHandle,
    /// Reusable forward plan for transforming newly supplied impulse responses.
    ir_plan: FFTPlan,
    /// Retained after both plans because VkFFT borrows it for initialization uploads.
    _initialization_fence: crate::gpu::resource::OwnedFence,
    /// Scratch buffer bound to `ir_plan`; transformed spectra are copied out.
    ir_staging_buffer: BufferHandle,
    /// Compiled compute pipeline for complex-domain frequency multiplication.
    multiply_pipeline_id: PipelineId,
    /// FFT size: next power of two of (block_size + ir_samples - 1).
    fft_size: usize,
    /// Overlap tail length: fft_size - block_size samples preserved between blocks.
    tail_size: usize,
    /// Shared backend kept alive until all plans and buffers above are dropped.
    ///
    /// Rust drops fields in declaration order, so this owner must remain last.
    gpu: Arc<VkBackend>,
}

/// Uploaded frequency-domain impulse response produced by the convolver.
///
/// The wrapper owns the GPU storage buffer for the transformed IR. Registries
/// should store it behind `Arc` so active voices can keep using an IR after its
/// application-facing id is removed.
pub struct ImpulseResponse {
    /// GPU storage buffer containing the FFT-transformed stereo IR.
    buffer: BufferHandle,
}

impl Drop for FftConvolver {
    fn drop(&mut self) {
        // Convolver submissions are synchronous; releasing a convolver must not
        // retain its private multiply pipeline in a shared backend indefinitely.
        self.gpu
            .compute()
            .release_pipeline(self.multiply_pipeline_id);
    }
}

impl FftConvolver {
    /// Creates reusable signal/IR FFT plans and the complex multiply pipeline.
    pub fn new(
        gpu: Arc<VkBackend>,
        sound: SoundConfig,
        config: AuralizationConfig,
    ) -> SoundResult<Self> {
        let fft_size = validated_fft_size(sound, config)?;
        let tail_size = fft_size - config.block_size;
        let initialization_fence = crate::gpu::resource::OwnedFence::new(gpu.device_context())?;

        let (signal_plan, signal_buffer) = Self::build_fft_plan(
            gpu.as_ref(),
            fft_size as u64,
            true,
            initialization_fence.handle,
        )?;
        // VkFFT plan construction is expensive (~165 ms on the benchmark GPU).
        // Bind one reusable scratch buffer at engine setup instead of creating a
        // new plan synchronously every time a moving voice receives a new IR.
        let (ir_plan, ir_staging_buffer) = Self::build_fft_plan(
            gpu.as_ref(),
            fft_size as u64,
            false,
            initialization_fence.handle,
        )?;
        let multiply_pipeline_id = Self::create_multiply_pipeline(gpu.as_ref())?;

        Ok(Self {
            sound,
            block_size: config.block_size,
            signal_plan,
            signal_buffer,
            ir_plan,
            _initialization_fence: initialization_fence,
            ir_staging_buffer,
            multiply_pipeline_id,
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

        let buffer = self
            .gpu
            .memory()
            .create_storage_buffer(self.ir_staging_buffer.size)?;

        self.gpu.memory().clear_buffer(&self.ir_staging_buffer)?;
        self.gpu
            .memory()
            .upload_typed(&self.ir_staging_buffer, samples)?;
        self.transform_staged_ir_into(&buffer)?;

        Ok(Arc::new(ImpulseResponse {
            buffer,
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
            .memory()
            .create_storage_buffer(self.ir_staging_buffer.size)?;
        let copy_bytes =
            u64::from(snapshot.sample_count()) * std::mem::size_of::<IrSample>() as u64;

        self.gpu
            .compute()
            .submit_compute_and_wait(|command_buffer| unsafe {
                // Initialize the padding and copy the IR prefix together. The fill
                // overlaps the copy, so order those transfer writes explicitly.
                self.gpu
                    .memory()
                    .record_clear_buffer(command_buffer, &self.ir_staging_buffer)?;
                self.gpu.compute().record_buffer_barrier(
                    command_buffer,
                    &BufferBarrierSpec::transfer_write_to_transfer_write(
                        self.ir_staging_buffer.buffer,
                    ),
                );
                self.gpu.memory().record_copy_buffer(
                    command_buffer,
                    source_buffer,
                    &self.ir_staging_buffer,
                    copy_bytes,
                    Some(
                        BufferBarrierSpec::transfer_write_to_compute_shader_read_write(
                            self.ir_staging_buffer.buffer,
                        ),
                    ),
                )
            })?;
        self.transform_staged_ir_into(&buffer)?;

        Ok(Arc::new(ImpulseResponse {
            buffer,
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
            self.sound.output_channels.count(),
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
        stream.queue_impulse_response_transition(
            impulse_response,
            self.block_size * stream.output_channels,
            self.block_size * IR_TRANSITION_BLOCKS,
        );
        Ok(())
    }

    fn validate_impulse_response(&self, impulse_response: &ImpulseResponse) -> SoundResult<()> {
        self.gpu.memory().validate_owner(&impulse_response.buffer)?;
        if impulse_response.buffer.size != self.signal_buffer.size {
            return Err(SoundError::invalid_argument(
                "Impulse response dimensions must match this convolver",
            ));
        }
        Ok(())
    }

    fn validate_stream(&self, stream: &ConvolutionStream) -> SoundResult<()> {
        self.validate_impulse_response(&stream.current.impulse_response)?;
        if stream.current.complex_cpu_buffer.len() != self.fft_size
            || stream.current.tail.len() != self.tail_size
            || stream.output_channels != self.sound.output_channels.count()
        {
            return Err(SoundError::invalid_argument(
                "Stream dimensions and output channels must match this convolver",
            ));
        }
        Ok(())
    }

    fn ir_capacity(&self) -> usize {
        // Linear convolution needs N + M - 1 samples; the remaining FFT bins
        // must stay zero so the last input samples cannot wrap into the head.
        self.fft_size - self.block_size + 1
    }

    /// Transforms the scratch time-domain IR and copies its reusable spectrum.
    fn transform_staged_ir_into(&self, destination: &BufferHandle) -> SoundResult<()> {
        self.gpu
            .compute()
            .submit_compute_and_wait(|command_buffer| unsafe {
                // The queue lock serializes recording; this owner retains plans and buffers through the wait.
                self.ir_plan
                    .append(command_buffer)
                    .map_err(|e| SoundError::external(format!("IR forward launch failed: {e}")))?;
                // Make FFT writes available to the copy in this same submission.
                self.gpu.compute().record_buffer_barrier(
                    command_buffer,
                    &BufferBarrierSpec::compute_shader_write_to_transfer_read(
                        self.ir_staging_buffer.buffer,
                    ),
                );
                self.gpu.memory().record_copy_buffer(
                    command_buffer,
                    &self.ir_staging_buffer,
                    destination,
                    self.ir_staging_buffer.size,
                    Some(
                        BufferBarrierSpec::transfer_write_to_compute_shader_read_write(
                            destination.buffer,
                        ),
                    ),
                )
            })
    }

    /// Processes one mono input block using this stream's filter and overlap history.
    pub fn process_block(
        &mut self,
        stream: &mut ConvolutionStream,
        input_block: &[f32],
        output_block: &mut [f32],
    ) -> SoundResult<()> {
        self.validate_block_lengths(input_block, output_block)?;
        self.validate_stream(stream)?;
        self.start_pending_transition(stream);

        self.process_state_block(&mut stream.current, input_block, output_block)?;

        let transition_complete = if let Some(transition) = stream.ir_transition.as_mut() {
            transition.output_buffer.fill(0.0);
            self.process_state_block(
                &mut transition.previous,
                input_block,
                &mut transition.output_buffer,
            )?;
            transition.blend_into(output_block, self.sound.output_channels.count())
        } else {
            false
        };
        if transition_complete {
            stream.ir_transition = None;
        }

        Ok(())
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
        let mut any_tail = self.flush_state_block(&mut stream.current, output_block);
        let transition_complete = if let Some(transition) = stream.ir_transition.as_mut() {
            transition.output_buffer.fill(0.0);
            let previous_has_tail =
                self.flush_state_block(&mut transition.previous, &mut transition.output_buffer);
            any_tail |= previous_has_tail;
            transition.blend_into(output_block, output_channels) || !previous_has_tail
        } else {
            false
        };
        if transition_complete {
            stream.ir_transition = None;
        }
        // No dry samples remain to seed a newly queued filter. Retaining a
        // pending update here would extend the stream with a silent IR tail.
        stream.pending_impulse_response = None;

        Ok(any_tail)
    }

    /// Begins the newest coalesced IR update after the preceding ramp finishes.
    fn start_pending_transition(&self, stream: &mut ConvolutionStream) {
        if stream.ir_transition.is_some() {
            return;
        }
        let Some(impulse_response) = stream.pending_impulse_response.take() else {
            return;
        };
        stream.transition_impulse_response(
            impulse_response,
            self.block_size * self.sound.output_channels.count(),
            self.block_size * IR_TRANSITION_BLOCKS,
        );
    }

    /// Processes one block through a single IR and overlap state.
    fn process_state_block(
        &self,
        convolution: &mut FilterState,
        input_block: &[f32],
        output_block: &mut [f32],
    ) -> SoundResult<()> {
        crate::validate!(
            convolution.tail_position == 0,
            ErrorCode::InvalidArgument,
            "convolution stream has already started flushing its tail and cannot process more input blocks"
        );

        self.convolve(convolution, input_block)?;
        self.overlap_add(&mut convolution.complex_cpu_buffer, &mut convolution.tail);
        self.copy_samples(
            &convolution.complex_cpu_buffer[..self.block_size],
            output_block,
        );
        Ok(())
    }

    /// Copies and advances one block of a single convolution state's tail.
    fn flush_state_block(&self, convolution: &mut FilterState, output_block: &mut [f32]) -> bool {
        output_block.fill(0.0);
        if convolution.tail_position >= convolution.tail.len() {
            return false;
        }

        let tail_remaining = convolution.tail.len() - convolution.tail_position;
        let frames_to_copy = self.block_size.min(tail_remaining);
        self.copy_samples(
            &convolution.tail
                [convolution.tail_position..convolution.tail_position + frames_to_copy],
            output_block,
        );
        convolution.tail_position = convolution.tail_position.saturating_add(self.block_size);
        true
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

    /// Performs FFT, frequency-domain multiply with IR spectrum, and inverse FFT for one block.
    ///
    /// Updates the stream's CPU complex buffer in-place with the convolved result.
    fn convolve(&self, convolution: &mut FilterState, input_block: &[f32]) -> SoundResult<()> {
        // Pack real mono input as complex for the FFT, zeroing the imaginary part.
        // This allows us to use the same FFT pipeline for stereo IR (packed as left + i*right).
        convolution.complex_cpu_buffer.fill([0.0, 0.0]);
        for (i, &sample) in input_block.iter().enumerate() {
            convolution.complex_cpu_buffer[i] = [sample, 0.0];
        }

        self.upload_signal(&convolution.complex_cpu_buffer)?;

        // Single submission preserves GPU command ordering: forward FFT -> multiply -> inverse FFT.
        self.gpu
            .compute()
            .submit_compute_and_wait(|command_buffer| unsafe {
                // Plans and their buffers remain owned until this synchronous submission completes.
                self.signal_plan.append(command_buffer).map_err(|e| {
                    SoundError::external(format!("Signal forward launch failed: {e}"))
                })?;
                self.gpu.compute().record_buffer_barrier(
                    command_buffer,
                    &BufferBarrierSpec::compute_shader_write_to_compute_shader_read_write(
                        self.signal_buffer.buffer,
                    ),
                );

                self.append_multiply(command_buffer, &convolution.impulse_response.buffer)
                    .map_err(|e| SoundError::external(format!("Multiply dispatch failed: {e}")))?;
                self.gpu.compute().record_buffer_barrier(
                    command_buffer,
                    &BufferBarrierSpec::compute_shader_write_to_compute_shader_read_write(
                        self.signal_buffer.buffer,
                    ),
                );

                self.signal_plan
                    .append_inverse(command_buffer)
                    .map_err(|e| {
                        SoundError::external(format!("Signal inverse launch failed: {e}"))
                    })?;

                Ok(())
            })?;

        self.gpu.memory().download_typed(
            &self.signal_buffer,
            self.fft_size,
            &mut convolution.complex_cpu_buffer,
        )
    }

    /// Uploads a complex signal block to the GPU, overwriting the previous contents.
    fn upload_signal(&self, signal: &[[f32; 2]]) -> SoundResult<()> {
        // The CPU staging block already includes the zero-padded FFT tail and
        // overwrites the complete allocation, so a preceding GPU fill is redundant.
        debug_assert_eq!(
            std::mem::size_of_val(signal) as u64,
            self.signal_buffer.size
        );
        self.gpu.memory().upload_typed(&self.signal_buffer, signal)
    }

    /// Records a complex frequency-domain multiply dispatch.
    ///
    /// Multiplies the signal by the IR spectrum in place across all FFT bins.
    fn append_multiply(
        &self,
        command_buffer: vk::CommandBuffer,
        ir_buffer: &BufferHandle,
    ) -> SoundResult<()> {
        let count = self.fft_size as u32;
        let dispatch_spec = DispatchSpec {
            pipeline_id: self.multiply_pipeline_id,
            storage_buffers: vec![
                StorageBufferWrite::new(0, self.signal_buffer.buffer),
                StorageBufferWrite::new(1, ir_buffer.buffer),
            ],
            push_constants: count.to_ne_bytes().to_vec(),
            push_constant_offset: 0,
            groups: [count.div_ceil(256), 1, 1],
        };

        // Private caller records on this backend and retains both spectra through the wait.
        unsafe {
            self.gpu
                .compute()
                .record_dispatch(command_buffer, &dispatch_spec)
        }
    }

    /// Applies overlap-add to combine FFT output with carried tail for seamless convolution.
    ///
    /// Adds the previous block's tail into the output head, then extracts and saves
    /// the new tail for the next block. This ensures output continuity across blocks.
    fn overlap_add(&self, output: &mut [[f32; 2]], tail: &mut [[f32; 2]]) {
        for (frame, previous) in output.iter_mut().zip(tail.iter()) {
            frame[0] += previous[0];
            frame[1] += previous[1];
        }

        let tail_start = self.block_size;
        let tail_end = tail_start + tail.len();
        tail.copy_from_slice(&output[tail_start..tail_end]);
    }

    /// Unpacks time-domain complex frames into the configured channel layout.
    fn copy_samples(&self, frames: &[[f32; 2]], output: &mut [f32]) {
        match self.sound.output_channels {
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

    /// Constructs an FFT plan and allocates GPU storage for one FFT transformation.
    ///
    /// Safety: The raw Vulkan handles passed to VkFFT are borrowed from the GPU backend
    /// and remain valid for the lifetime of the returned FFTPlan. The plan does not take
    /// ownership. Set `normalize` true to divide FFT output by fft_size (for signal FFT).
    fn build_fft_plan(
        gpu: &VkBackend,
        fft_size: u64,
        normalize: bool,
        fence: vk::Fence,
    ) -> SoundResult<(FFTPlan, BufferHandle)> {
        gpu.compute().with_queue(|queue, command_pool| {
            // VkFFT consumes raw Vulkan handles but does not take ownership.
            // Handles remain valid for the lifetime of the plan and the GPU backend.
            let handles = DeviceHandles {
                physical_device: gpu.device().physical_device,
                device: gpu.device().device.handle(),
                queue,
                command_pool,
                fence,
            };

            // Allocate storage for complex-domain FFT: fft_size complex numbers = fft_size * 2 f32 values.
            let buffer = gpu
                .memory()
                .create_storage_buffer(fft_size * 2 * std::mem::size_of::<f32>() as u64)?;

            let builder = FFTPlanBuilder::new(&handles)
                .with_length(fft_size)
                .with_buffer(buffer.buffer, buffer.size);
            let builder = if normalize {
                builder.with_normalization()
            } else {
                builder
            };
            // with_queue serializes VkFFT's initialization submissions and pool access.
            let plan = unsafe { builder.build() }
                .map_err(|e| SoundError::external(format!("FFT plan build failed: {e}")))?;

            Ok((plan, buffer))
        })
    }

    /// Compiles the frequency-domain multiply compute pipeline.
    ///
    /// The pipeline performs element-wise complex multiplication:
    /// (a + i*b) * (c + i*d) = (ac - bd) + i(ad + bc).
    /// Descriptor 0: signal FFT buffer (read-write), Descriptor 1: IR spectrum buffer (read-only).
    fn create_multiply_pipeline(gpu: &VkBackend) -> SoundResult<PipelineId> {
        let multiply_shader_id = gpu
            .shaders()
            .load_glsl_file(ShaderStage::Compute, MULTIPLY_SHADER_PATH)?;

        let pipeline_spec = ComputePipelineSpec {
            shader_id: multiply_shader_id,
            storage_buffer_bindings: vec![0, 1],
            push_constant_ranges: vec![PushConstantSpec::new(0, std::mem::size_of::<u32>() as u32)], // Element count
        };

        gpu.compute().create_pipeline(pipeline_spec)
    }
}

fn validated_fft_size(sound: SoundConfig, config: AuralizationConfig) -> SoundResult<usize> {
    if config.block_size == 0 || sound.ir_num_samples == 0 || sound.sample_rate == 0 {
        return Err(SoundError::invalid_argument(
            "FFT block size, IR length and sample rate must be non-zero",
        ));
    }
    config
        .block_size
        .checked_add(sound.ir_num_samples as usize - 1)
        .and_then(usize::checked_next_power_of_two)
        // The multiply shader uses a 32-bit element count; allocations must also
        // fit Rust's addressable slice size before any GPU work is initialized.
        .filter(|size| *size <= u32::MAX as usize && *size <= isize::MAX as usize / 8)
        .ok_or_else(|| SoundError::invalid_argument("FFT size exceeds addressable capacity"))
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
