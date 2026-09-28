use crate::acoustics::{GpuIrSnapshot, IrSample};
use crate::auralization::IR_TRANSITION_BLOCKS;
use crate::auralization::voice::{Voice, VoiceConvolutionState};
use crate::core::config::EngineConfig;
use crate::core::config::OutputChannels;
use crate::core::error::ErrorCode;
use crate::core::error::{SoundError, SoundResult};
use crate::gpu::backend::VkBackend;
use crate::gpu::compute::{
    BufferBarrierSpec, ComputePipelineSpec, DescriptorBindingSpec, DescriptorWrite, DispatchSpec,
    PipelineId, PushConstantSpec,
};
use crate::gpu::memory::BufferHandle;
use crate::gpu::shader::ShaderStage;
use ash::vk;
use std::sync::Arc;
use vkfft_rs::plan::{DeviceHandles, FFTPlan, FFTPlanBuilder};

const MULTIPLY_SHADER_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/auralization/shaders/multiply.comp.glsl"
);

/// GPU-backed partitioned convolver for mono voices and stereo impulse responses.
///
/// Stereo IRs are packed as complex time-domain samples (`left + i * right`)
/// before the IR FFT. A mono dry signal can then produce both ears with the
/// same FFT, multiply, and inverse FFT used for mono convolution.
pub struct PartitionedConvolver {
    /// Engine configuration containing block size, channel layout, and IR length.
    config: EngineConfig,
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

impl Drop for PartitionedConvolver {
    fn drop(&mut self) {
        // Convolver submissions are synchronous; releasing a convolver must not
        // retain its private multiply pipeline in a shared backend indefinitely.
        self.gpu
            .compute()
            .release_pipeline(self.multiply_pipeline_id);
    }
}

impl PartitionedConvolver {
    /// Creates the shared FFT plan and multiply compute pipeline.
    pub fn new(gpu: Arc<VkBackend>, config: EngineConfig) -> SoundResult<Self> {
        let fft_size = validated_fft_size(config)?;
        let tail_size = fft_size - config.auralization.block_size as usize;
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
            config,
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

    /// Uploads an impulse response and transforms it into frequency-domain form.
    pub fn create_impulse_response(
        &self,
        samples: &[IrSample],
    ) -> SoundResult<Arc<ImpulseResponse>> {
        crate::validate!(
            !samples.is_empty(),
            ErrorCode::InvalidArgument,
            "Impulse response must not be empty"
        );

        crate::validate!(
            samples.len() <= self.fft_size,
            ErrorCode::InvalidArgument,
            "Impulse response length {} exceeds FFT size {}",
            samples.len(),
            self.fft_size
        );

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
        &self,
        snapshot: &GpuIrSnapshot,
    ) -> SoundResult<Arc<ImpulseResponse>> {
        validate_gpu_ir_snapshot(snapshot, self.fft_size, self.config.sound.sample_rate)?;
        let source_buffer = snapshot
            .buffer
            .as_ref()
            .ok_or_else(|| SoundError::invalid_argument("GPU IR snapshot has no buffer"))?;

        let buffer = self
            .gpu
            .memory()
            .create_storage_buffer(self.ir_staging_buffer.size)?;
        let copy_bytes = snapshot
            .sample_count
            .checked_mul(std::mem::size_of::<IrSample>() as u32)
            .ok_or_else(|| SoundError::invalid_argument("GPU IR copy byte size overflow"))?
            as u64;

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

    /// Returns the number of overlap frames kept per voice.
    pub fn tail_size(&self) -> usize {
        self.tail_size
    }

    /// Transforms the scratch time-domain IR and copies its reusable spectrum.
    fn transform_staged_ir_into(&self, destination: &BufferHandle) -> SoundResult<()> {
        self.gpu
            .compute()
            .submit_compute_and_wait(|command_buffer| unsafe {
                // The queue lock serializes recording; this owner retains plans and buffers through the wait.
                self.ir_plan.append(command_buffer).map_err(|e| {
                    SoundError::external(format!("IR forward launch failed: {e:?}"))
                })?;
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

    /// Processes one mono input block for a voice into the configured output layout.
    pub fn process_block(
        &self,
        voice: &mut Voice,
        input_block: &[f32],
        output_block: &mut [f32],
    ) -> SoundResult<()> {
        self.validate_block_lengths(input_block, output_block)?;
        self.start_pending_transition(voice);

        let voice_id = voice.id();
        self.process_state_block(voice_id, &mut voice.convolution, input_block, output_block)?;

        let transition_complete = if let Some(transition) = voice.ir_transition.as_mut() {
            transition.output_buffer.fill(0.0);
            self.process_state_block(
                voice_id,
                &mut transition.previous,
                input_block,
                &mut transition.output_buffer,
            )?;
            blend_ir_transition(
                output_block,
                &transition.output_buffer,
                self.config.sound.output_channels.count(),
                transition.processed_frames,
                transition.total_frames,
            );
            transition.processed_frames = transition
                .processed_frames
                .saturating_add(self.config.auralization.block_size);
            transition.processed_frames >= transition.total_frames
        } else {
            false
        };
        if transition_complete {
            voice.ir_transition = None;
        }

        Ok(())
    }

    /// Drains the next block of already-computed overlap tail samples.
    ///
    /// Returns `true` when `output_block` contains tail samples. Returns `false`
    /// once the voice has no pending overlap tail; No additional convolution is
    /// performed; the method only drains the voice overlap state. Once a voice
    /// starts flushing its tail, it cannot process more input blocks until restarted.
    pub fn flush_tail_block(
        &self,
        voice: &mut Voice,
        output_block: &mut [f32],
    ) -> SoundResult<bool> {
        let output_channels = self.config.sound.output_channels.count();
        crate::validate!(
            output_block.len() == self.config.auralization.block_size * output_channels,
            ErrorCode::InvalidArgument,
            "output_block length {} must equal block size {} times channel count {}",
            output_block.len(),
            self.config.auralization.block_size,
            output_channels,
        );

        let mut any_tail = self.flush_state_block(&mut voice.convolution, output_block);
        let transition_complete = if let Some(transition) = voice.ir_transition.as_mut() {
            transition.output_buffer.fill(0.0);
            let previous_has_tail =
                self.flush_state_block(&mut transition.previous, &mut transition.output_buffer);
            any_tail |= previous_has_tail;
            blend_ir_transition(
                output_block,
                &transition.output_buffer,
                output_channels,
                transition.processed_frames,
                transition.total_frames,
            );
            transition.processed_frames = transition
                .processed_frames
                .saturating_add(self.config.auralization.block_size);
            transition.processed_frames >= transition.total_frames || !previous_has_tail
        } else {
            false
        };
        if transition_complete {
            voice.ir_transition = None;
        }
        // No dry samples remain to seed a newly queued filter. Retaining a
        // pending update here would extend the voice with a silent IR tail.
        voice.pending_impulse_response = None;

        Ok(any_tail)
    }

    /// Begins the newest coalesced IR update after the preceding ramp finishes.
    fn start_pending_transition(&self, voice: &mut Voice) {
        if voice.ir_transition.is_some() {
            return;
        }
        let Some(impulse_response) = voice.pending_impulse_response.take() else {
            return;
        };
        voice.transition_impulse_response(
            impulse_response,
            self.tail_size,
            self.fft_size,
            self.config.auralization.block_size * self.config.sound.output_channels.count(),
            self.config.auralization.block_size * IR_TRANSITION_BLOCKS,
        );
    }

    /// Processes one block through a single IR and overlap state.
    fn process_state_block(
        &self,
        voice_id: u64,
        convolution: &mut VoiceConvolutionState,
        input_block: &[f32],
        output_block: &mut [f32],
    ) -> SoundResult<()> {
        self.gpu
            .memory()
            .validate_owner(&convolution.impulse_response.buffer)?;
        if convolution.complex_cpu_buffer.len() != self.fft_size
            || convolution.tail.len() != self.tail_size
            || convolution.impulse_response.buffer.size != self.signal_buffer.size
        {
            return Err(SoundError::invalid_argument(
                "Voice and impulse response dimensions must match this convolver",
            ));
        }
        crate::validate!(
            convolution.tail_position == 0,
            ErrorCode::InvalidArgument,
            "voice {} has already started flushing its tail and cannot process more input blocks",
            voice_id
        );

        self.convolve(convolution, input_block)?;
        self.overlap_add(&mut convolution.complex_cpu_buffer, &mut convolution.tail);
        self.copy_samples(
            &convolution.complex_cpu_buffer,
            0,
            self.config.auralization.block_size,
            output_block,
        );
        Ok(())
    }

    /// Copies and advances one block of a single convolution state's tail.
    fn flush_state_block(
        &self,
        convolution: &mut VoiceConvolutionState,
        output_block: &mut [f32],
    ) -> bool {
        output_block.fill(0.0);
        if convolution.tail_position >= convolution.tail.len() {
            return false;
        }

        let tail_remaining = convolution.tail.len() - convolution.tail_position;
        let frames_to_copy = self.config.auralization.block_size.min(tail_remaining);
        self.copy_samples(
            &convolution.tail,
            convolution.tail_position,
            frames_to_copy,
            output_block,
        );
        convolution.tail_position = convolution
            .tail_position
            .saturating_add(self.config.auralization.block_size);
        true
    }

    /// Validates that input and output block sizes match expected channel layout and FFT capacity.
    fn validate_block_lengths(&self, input_block: &[f32], output_block: &[f32]) -> SoundResult<()> {
        crate::validate!(
            input_block.len() == self.config.auralization.block_size,
            ErrorCode::InvalidArgument,
            "input_block length {} must equal block size {}",
            input_block.len(),
            self.config.auralization.block_size
        );

        let output_channels = self.config.sound.output_channels.count();
        crate::validate!(
            output_block.len() == input_block.len() * output_channels,
            ErrorCode::InvalidArgument,
            "output_block length {} must equal {} times input_block length {}",
            output_block.len(),
            output_channels,
            input_block.len(),
        );
        crate::validate!(
            output_block.len() <= self.fft_size * output_channels,
            ErrorCode::InvalidArgument,
            "output_block length {} exceeds fft output sample capacity {}",
            output_block.len(),
            self.fft_size * output_channels
        );

        Ok(())
    }

    /// Performs FFT, frequency-domain multiply with IR spectrum, and inverse FFT for one block.
    ///
    /// Updates the voice's CPU complex buffer in-place with the convolved result.
    fn convolve(
        &self,
        convolution: &mut VoiceConvolutionState,
        input_block: &[f32],
    ) -> SoundResult<()> {
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
                    SoundError::external(format!("Signal forward launch failed: {e:?}"))
                })?;
                self.gpu.compute().record_buffer_barrier(
                    command_buffer,
                    &BufferBarrierSpec::compute_shader_write_to_compute_shader_read_write(
                        self.signal_buffer.buffer,
                    ),
                );

                self.append_multiply(
                    command_buffer,
                    &convolution.impulse_response.buffer,
                    self.fft_size as u32,
                )
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
                        SoundError::external(format!("Signal inverse launch failed: {e:?}"))
                    })?;

                Ok(())
            })?;

        self.download_convolution_output(&mut convolution.complex_cpu_buffer)?;

        Ok(())
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

    /// Downloads the convolved frequency-domain result from GPU to CPU buffer.
    fn download_convolution_output(&self, output: &mut [[f32; 2]]) -> SoundResult<()> {
        self.gpu
            .memory()
            .download_typed::<[f32; 2]>(&self.signal_buffer, output.len(), output)
    }

    /// Records a complex frequency-domain multiply dispatch.
    ///
    /// Multiplies `count` complex elements from the signal buffer with the IR spectrum buffer,
    /// storing results back into the signal buffer.
    fn append_multiply(
        &self,
        command_buffer: vk::CommandBuffer,
        ir_buffer: &BufferHandle,
        count: u32,
    ) -> SoundResult<()> {
        let dispatch_spec = DispatchSpec {
            pipeline_id: self.multiply_pipeline_id,
            descriptor_writes: vec![
                DescriptorWrite::storage_buffer(0, self.signal_buffer.buffer),
                DescriptorWrite::storage_buffer(1, ir_buffer.buffer),
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
        // Add previous block tail into the current block head.
        for i in 0..tail.len() {
            output[i][0] += tail[i][0];
            output[i][1] += tail[i][1];
        }

        // Store new overlap tail from current convolution result for the next block.
        let tail_start = self.config.auralization.block_size;
        let tail_end = tail_start + tail.len();
        tail.copy_from_slice(&output[tail_start..tail_end]);
    }

    /// Extracts interleaved output samples from complex frequency-domain data.
    ///
    /// Reads `count` complex frames starting at `start_index` and formats them according
    /// to the configured output channel layout (mono, stereo, or binaural).
    fn copy_samples(&self, from: &[[f32; 2]], start_index: usize, count: usize, to: &mut [f32]) {
        match self.config.sound.output_channels {
            OutputChannels::Mono => {
                // Mono output uses only the real (left channel) part of complex samples.
                for (dest_frame, source_frame) in (start_index..start_index + count).enumerate() {
                    to[dest_frame] = from[source_frame][0];
                }
            }
            OutputChannels::Stereo | OutputChannels::Binaural => {
                // Stereo output uses real (left) and imaginary (right) as separate channels.
                // Real and imaginary were packed during IR upload as stereo IR data.
                for (dest_frame, source_frame) in (start_index..start_index + count).enumerate() {
                    to[2 * dest_frame] = from[source_frame][0];
                    to[2 * dest_frame + 1] = from[source_frame][1];
                }
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
                .with_single_dimension(fft_size)
                .with_buffer(buffer.buffer, buffer.size);
            let builder = if normalize {
                builder.with_normalization()
            } else {
                builder
            };
            // with_queue serializes VkFFT's initialization submissions and pool access.
            let plan = unsafe { builder.build() }.map_err(|e| {
                SoundError::external(format!("Global signal plan build failed: {e:?}"))
            })?;

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
            descriptor_bindings: vec![
                DescriptorBindingSpec::storage_buffer(0), // Signal FFT buffer
                DescriptorBindingSpec::storage_buffer(1), // IR spectrum buffer
            ],
            push_constant_ranges: vec![PushConstantSpec::new(0, std::mem::size_of::<u32>() as u32)], // Element count
        };

        gpu.compute().create_pipeline(pipeline_spec)
    }
}

fn validated_fft_size(config: EngineConfig) -> SoundResult<usize> {
    if config.auralization.block_size == 0
        || config.sound.ir_num_samples == 0
        || config.sound.sample_rate == 0
    {
        return Err(SoundError::invalid_argument(
            "FFT block size, IR length and sample rate must be non-zero",
        ));
    }
    config
        .auralization
        .block_size
        .checked_add(config.sound.ir_num_samples as usize - 1)
        .and_then(usize::checked_next_power_of_two)
        // The multiply shader uses a 32-bit element count; allocations must also
        // fit Rust's addressable slice size before any GPU work is initialized.
        .filter(|size| *size <= u32::MAX as usize && *size <= isize::MAX as usize / 8)
        .ok_or_else(|| SoundError::invalid_argument("FFT size exceeds addressable capacity"))
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

fn validate_gpu_ir_snapshot(
    snapshot: &GpuIrSnapshot,
    fft_size: usize,
    sample_rate: u32,
) -> SoundResult<()> {
    crate::validate!(
        snapshot.sample_rate == sample_rate,
        ErrorCode::InvalidArgument,
        "GPU impulse response sample rate {} must match convolver sample rate {}",
        snapshot.sample_rate,
        sample_rate
    );
    crate::validate!(
        snapshot.sample_count > 0,
        ErrorCode::InvalidArgument,
        "GPU impulse response must not be empty"
    );
    crate::validate!(
        snapshot.sample_count as usize <= fft_size,
        ErrorCode::InvalidArgument,
        "GPU impulse response length {} exceeds FFT size {}",
        snapshot.sample_count,
        fft_size
    );
    if snapshot.buffer.is_none() {
        return Err(SoundError::invalid_argument(
            "GPU IR snapshot has no buffer",
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "convolver_tests.rs"]
mod tests;
