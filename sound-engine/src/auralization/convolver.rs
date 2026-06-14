use crate::acoustics::IrSample;
use crate::auralization::voice::Voice;
use crate::core::config::EngineConfig;
use crate::core::config::OutputChannels;
use crate::error::{ErrorCode, SoundError, SoundResult};
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
    gpu: Arc<VkBackend>,
    config: EngineConfig,
    signal_plan: FFTPlan,
    signal_buffer: BufferHandle,
    multiply_pipeline_id: PipelineId,
    fft_size: usize,
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

    fn convolve(&self, voice: &mut Voice, input_block: &[f32]) -> SoundResult<()> {
        // Pack real mono input as complex for the FFT, zeroing the imaginary part.
        voice.convolution.complex_cpu_buffer.fill([0.0, 0.0]);
        for (i, &sample) in input_block.iter().enumerate() {
            voice.convolution.complex_cpu_buffer[i] = [sample, 0.0];
        }

        self.upload_signal(&voice.convolution.complex_cpu_buffer)?;

        // One submission keeps FFT, multiply, and inverse FFT ordered on the GPU.
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

    fn upload_signal(&self, signal: &[[f32; 2]]) -> SoundResult<()> {
        self.gpu.memory().clear_buffer(&self.signal_buffer)?;
        self.gpu.memory().upload_typed(&self.signal_buffer, signal)
    }

    fn download_convolution_output(&self, output: &mut [[f32; 2]]) -> SoundResult<()> {
        self.gpu
            .memory()
            .download_typed::<[f32; 2]>(&self.signal_buffer, output.len(), output)
    }

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

    fn overlap_add(&self, output: &mut [[f32; 2]], tail: &mut [[f32; 2]]) {
        // Add previous block tail into the current block head.
        for i in 0..tail.len() {
            output[i][0] += tail[i][0];
            output[i][1] += tail[i][1];
        }

        // Store new overlap tail from current convolution result.
        let tail_start = self.config.auralization.block_size;
        let tail_end = tail_start + tail.len();
        tail.copy_from_slice(&output[tail_start..tail_end]);
    }

    fn copy_samples(&self, from: &[[f32; 2]], start_index: usize, count: usize, to: &mut [f32]) {
        match self.config.sound.output_channels {
            OutputChannels::Mono => {
                for (dest_frame, source_frame) in (start_index..start_index + count).enumerate() {
                    to[dest_frame] = from[source_frame][0]; // Take left channel as mono output
                }
            }
            OutputChannels::Stereo | OutputChannels::Binaural => {
                for (dest_frame, source_frame) in (start_index..start_index + count).enumerate() {
                    to[2 * dest_frame] = from[source_frame][0]; // Left channel
                    to[2 * dest_frame + 1] = from[source_frame][1]; // Right channel
                }
            }
        }
    }

    fn build_fft_plan(gpu: &VkBackend, fft_size: u64, normalize: bool) -> SoundResult<(FFTPlan, BufferHandle)> {
        unsafe {
            // VkFFT consumes raw Vulkan handles but does not own them.
            let handles = DeviceHandles {
                physical_device: gpu.device().physical_device,
                device: gpu.device().device.handle(),
                queue: gpu.compute().queue_handle()?,
                command_pool: gpu.compute().command_pool_handle()?,
                fence: vk::Fence::null(), // Not used for the global signal plan
            };

            let buffer = gpu
                .memory()
                .create_storage_buffer(fft_size * 2 * std::mem::size_of::<f32>() as u64)?; // Placeholder size, will be updated per voice

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

    fn create_multiply_pipeline(gpu: &VkBackend) -> SoundResult<PipelineId> {
        let multiply_shader_id = gpu
            .shaders()
            .load_glsl_file(ShaderStage::Compute, MULTIPLY_SHADER_PATH)?;

        let pipeline_spec = ComputePipelineSpec {
            shader_id: multiply_shader_id,
            descriptor_bindings: vec![
                DescriptorBindingSpec::storage_buffer(0),
                DescriptorBindingSpec::storage_buffer(1),
            ],
            push_constant_ranges: vec![PushConstantSpec::new(0, std::mem::size_of::<u32>() as u32)],
        };

        gpu.compute().create_pipeline(pipeline_spec)
    }
}
