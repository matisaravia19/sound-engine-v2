use crate::acoustics::IrSample;
use crate::auralization::voice::Voice;
use crate::core::config::EngineConfig;
use crate::core::config::OutputChannels;
use crate::core::error::{ErrorCode, SoundError, SoundResult};
use crate::gpu::backend::VkBackend;
use crate::gpu::compute::{
    BufferBarrierSpec, ComputePipelineSpec, DescriptorBindingSpec, DescriptorWrite, DispatchSpec, PipelineId,
    PushConstantSpec,
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
    /// Shared GPU backend for FFT and compute dispatch operations.
    gpu: Arc<VkBackend>,
    /// Engine configuration containing block size, channel layout, and IR length.
    config: EngineConfig,
    /// Shared FFT plan for transforming mono input signals to frequency-domain.
    signal_plan: FFTPlan,
    /// Reusable GPU storage buffer for signal FFT input and output.
    signal_buffer: BufferHandle,
    /// Compiled compute pipeline for complex-domain frequency multiplication.
    multiply_pipeline_id: PipelineId,
    /// FFT size: next power of two of (block_size + ir_samples - 1).
    fft_size: usize,
    /// Overlap tail length: fft_size - block_size samples preserved between blocks.
    tail_size: usize,
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

impl PartitionedConvolver {
    /// Creates the shared FFT plan and multiply compute pipeline.
    pub fn new(gpu: Arc<VkBackend>, config: EngineConfig) -> SoundResult<Self> {
        let fft_size = (config.auralization.block_size + config.sound.ir_num_samples as usize - 1).next_power_of_two();
        let tail_size = fft_size - config.auralization.block_size as usize;

        let (signal_plan, signal_buffer) = Self::build_fft_plan(gpu.as_ref(), fft_size as u64, true)?;
        let multiply_pipeline_id = Self::create_multiply_pipeline(gpu.as_ref())?;

        Ok(Self {
            gpu,
            config,
            signal_plan,
            signal_buffer,
            multiply_pipeline_id,
            fft_size,
            tail_size,
        })
    }

    /// Uploads an impulse response and transforms it into frequency-domain form.
    pub fn create_impulse_response(&self, samples: &[IrSample]) -> SoundResult<Arc<ImpulseResponse>> {
        crate::debug_validate!(
            !samples.is_empty(),
            ErrorCode::InvalidArgument,
            "Impulse response must not be empty"
        );

        crate::debug_validate!(
            samples.len() <= self.fft_size,
            ErrorCode::InvalidArgument,
            "Impulse response length {} exceeds FFT size {}",
            samples.len(),
            self.fft_size
        );

        let (plan, buffer) = Self::build_fft_plan(self.gpu.as_ref(), self.fft_size as u64, false)?;

        self.gpu.memory().clear_buffer(&buffer)?;
        self.gpu.memory().upload_typed(&buffer, samples)?;

        // Transform the IR once so voices can reuse the spectrum per block.
        self.gpu.compute().submit_compute_and_wait(|command_buffer| {
            plan.append(command_buffer)
                .map_err(|e| SoundError::external(format!("IR forward launch failed: {e:?}")))?;

            Ok(())
        })?;

        Ok(Arc::new(ImpulseResponse { buffer }))
    }

    /// Returns the FFT size used for signal and IR buffers.
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Returns the number of overlap frames kept per voice.
    pub fn tail_size(&self) -> usize {
        self.tail_size
    }

    /// Processes one mono input block for a voice into the configured output layout.
    pub fn process_block(&self, voice: &mut Voice, input_block: &[f32], output_block: &mut [f32]) -> SoundResult<()> {
        self.validate_block_lengths(input_block, output_block)?;

        crate::debug_validate!(
            voice.convolution.tail_position == 0,
            ErrorCode::InvalidArgument,
            "voice {} has already started flushing its tail and cannot process more input blocks",
            voice.id()
        );

        self.convolve(voice, input_block)?;
        self.overlap_add(&mut voice.convolution.complex_cpu_buffer, &mut voice.convolution.tail);

        self.copy_samples(
            &voice.convolution.complex_cpu_buffer,
            0,
            self.config.auralization.block_size,
            output_block,
        );

        Ok(())
    }

    /// Drains the next block of already-computed overlap tail samples.
    ///
    /// Returns `true` when `output_block` contains tail samples. Returns `false`
    /// once the voice has no pending overlap tail; No additional convolution is
    /// performed; the method only drains the voice overlap state. Once a voice
    /// starts flushing its tail, it cannot process more input blocks until restarted.
    pub fn flush_tail_block(&self, voice: &mut Voice, output_block: &mut [f32]) -> SoundResult<bool> {
        let output_channels = self.config.sound.output_channels.count();
        crate::debug_validate!(
            output_block.len() == self.config.auralization.block_size * output_channels,
            ErrorCode::InvalidArgument,
            "output_block length {} must equal block size {} times channel count {}",
            output_block.len(),
            self.config.auralization.block_size,
            output_channels,
        );

        if voice.convolution.tail_position >= voice.convolution.tail.len() {
            return Ok(false);
        }

        output_block.fill(0.0);
        let tail_remaining = voice.convolution.tail.len() - voice.convolution.tail_position;
        let frames_to_copy = self.config.auralization.block_size.min(tail_remaining);

        self.copy_samples(
            &voice.convolution.tail,
            voice.convolution.tail_position,
            frames_to_copy,
            output_block,
        );
        voice.convolution.tail_position = voice
            .convolution
            .tail_position
            .saturating_add(self.config.auralization.block_size);

        Ok(true)
    }

    /// Validates that input and output block sizes match expected channel layout and FFT capacity.
    fn validate_block_lengths(&self, input_block: &[f32], output_block: &[f32]) -> SoundResult<()> {
        crate::debug_validate!(
            input_block.len() == self.config.auralization.block_size,
            ErrorCode::InvalidArgument,
            "input_block length {} must equal block size {}",
            input_block.len(),
            self.config.auralization.block_size
        );

        let output_channels = self.config.sound.output_channels.count();
        crate::debug_validate!(
            output_block.len() == input_block.len() * output_channels,
            ErrorCode::InvalidArgument,
            "output_block length {} must equal {} times input_block length {}",
            output_block.len(),
            output_channels,
            input_block.len(),
        );
        crate::debug_validate!(
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
    fn convolve(&self, voice: &mut Voice, input_block: &[f32]) -> SoundResult<()> {
        // Pack real mono input as complex for the FFT, zeroing the imaginary part.
        // This allows us to use the same FFT pipeline for stereo IR (packed as left + i*right).
        voice.convolution.complex_cpu_buffer.fill([0.0, 0.0]);
        for (i, &sample) in input_block.iter().enumerate() {
            voice.convolution.complex_cpu_buffer[i] = [sample, 0.0];
        }

        self.upload_signal(&voice.convolution.complex_cpu_buffer)?;

        // Single submission preserves GPU command ordering: forward FFT -> multiply -> inverse FFT.
        self.gpu.compute().submit_compute_and_wait(|command_buffer| {
            self.signal_plan
                .append(command_buffer)
                .map_err(|e| SoundError::external(format!("Signal forward launch failed: {e:?}")))?;
            self.gpu.compute().record_buffer_barrier(
                command_buffer,
                &BufferBarrierSpec::compute_shader_write_to_compute_shader_read_write(self.signal_buffer.buffer),
            );

            self.append_multiply(
                command_buffer,
                &voice.convolution.impulse_response.buffer,
                self.fft_size as u32,
            )
            .map_err(|e| SoundError::external(format!("Multiply dispatch failed: {e}")))?;
            self.gpu.compute().record_buffer_barrier(
                command_buffer,
                &BufferBarrierSpec::compute_shader_write_to_compute_shader_read_write(self.signal_buffer.buffer),
            );

            self.signal_plan
                .append_inverse(command_buffer)
                .map_err(|e| SoundError::external(format!("Signal inverse launch failed: {e:?}")))?;

            Ok(())
        })?;

        self.download_convolution_output(&mut voice.convolution.complex_cpu_buffer)?;

        Ok(())
    }

    /// Uploads a complex signal block to the GPU, overwriting the previous contents.
    fn upload_signal(&self, signal: &[[f32; 2]]) -> SoundResult<()> {
        self.gpu.memory().clear_buffer(&self.signal_buffer)?;
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

        self.gpu.compute().record_dispatch(command_buffer, &dispatch_spec)
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
    fn build_fft_plan(gpu: &VkBackend, fft_size: u64, normalize: bool) -> SoundResult<(FFTPlan, BufferHandle)> {
        unsafe {
            // VkFFT consumes raw Vulkan handles but does not take ownership.
            // Handles remain valid for the lifetime of the plan and the GPU backend.
            let handles = DeviceHandles {
                physical_device: gpu.device().physical_device,
                device: gpu.device().device.handle(),
                queue: gpu.compute().queue_handle()?,
                command_pool: gpu.compute().command_pool_handle()?,
                fence: vk::Fence::null(), // Not used with the plan append pattern
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
            let plan = builder
                .build()
                .map_err(|e| SoundError::external(format!("Global signal plan build failed: {e:?}")))?;

            Ok((plan, buffer))
        }
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
