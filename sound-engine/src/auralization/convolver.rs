use crate::acoustics::IrSample;
use crate::auralization::{ImpulseResponseId, VoiceId};
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
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
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
    id_counter: AtomicU64,
    voices: Mutex<HashMap<VoiceId, Voice>>,
    impulse_responses: HashMap<ImpulseResponseId, Arc<ImpulseResponse>>,
    signal_plan: FFTPlan,
    signal_buffer: BufferHandle,
    multiply_pipeline_id: PipelineId,
    fft_size: usize,
    tail_size: usize,
}

struct Voice {
    impulse_response: Arc<ImpulseResponse>,
    tail: Vec<[f32; 2]>,
    complex_cpu_buffer: Vec<[f32; 2]>,
}

struct ImpulseResponse {
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
            id_counter: AtomicU64::new(0),
            voices: Mutex::new(HashMap::new()),
            impulse_responses: HashMap::new(),
            signal_plan,
            signal_buffer,
            multiply_pipeline_id,
            fft_size,
            tail_size,
        })
    }

    /// Registers an impulse response and uploads its frequency-domain form.
    pub fn register_impulse_response(&mut self, id: ImpulseResponseId, samples: &[IrSample]) -> SoundResult<()> {
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

        let ir = Arc::new(ImpulseResponse { buffer });
        self.impulse_responses.insert(id, ir);

        Ok(())
    }

    /// Removes an impulse response from the convolver.
    pub fn forget_impulse_response(&mut self, id: ImpulseResponseId) {
        self.impulse_responses.remove(&id);
    }

    /// Starts a voice that uses a registered impulse response.
    pub fn start_sound(&mut self, ir_id: ImpulseResponseId) -> SoundResult<VoiceId> {
        let ir = self
            .impulse_responses
            .get(&ir_id)
            .ok_or_else(|| SoundError::not_found(format!("Unknown impulse response id {ir_id}")))?;

        let voice_id = self.id_counter.fetch_add(1, Ordering::Relaxed);

        let mut voices = self
            .voices
            .lock()
            .map_err(|_| SoundError::poisoned_lock("Voices lock is poisoned"))?;

        voices.insert(
            voice_id,
            Voice {
                impulse_response: ir.clone(),
                tail: vec![[0.0, 0.0]; self.tail_size],
                complex_cpu_buffer: vec![[0.0, 0.0]; self.fft_size],
            },
        );

        Ok(voice_id)
    }

    /// Stops a voice and releases its overlap state.
    pub fn stop_sound(&mut self, voice_id: VoiceId) -> SoundResult<()> {
        let mut voices = self
            .voices
            .lock()
            .map_err(|_| SoundError::poisoned_lock("Voices lock is poisoned"))?;
        voices.remove(&voice_id);
        Ok(())
    }

    /// Processes one mono input block for a voice into the configured output layout.
    pub fn process_block(&self, voice_id: VoiceId, input_block: &[f32], output_block: &mut [f32]) -> SoundResult<()> {
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

        let mut voices = self
            .voices
            .lock()
            .map_err(|_| SoundError::poisoned_lock("Voices lock is poisoned"))?;

        let voice = voices
            .get_mut(&voice_id)
            .ok_or_else(|| SoundError::not_found(format!("Unknown voice id {voice_id}")))?;

        self.convolve(voice, input_block)?;
        self.overlap_add(&mut voice.complex_cpu_buffer, &mut voice.tail);
        self.copy_samples(&voice.complex_cpu_buffer, output_block);

        Ok(())
    }

    fn convolve(&self, voice: &mut Voice, input_block: &[f32]) -> SoundResult<()> {
        // Pack real mono input as complex for the FFT, zeroing the imaginary part.
        voice.complex_cpu_buffer.fill([0.0, 0.0]);
        for (i, &sample) in input_block.iter().enumerate() {
            voice.complex_cpu_buffer[i] = [sample, 0.0];
        }

        self.upload_signal(&voice.complex_cpu_buffer)?;

        // One submission keeps FFT, multiply, and inverse FFT ordered on the GPU.
        self.gpu.compute().submit_compute_and_wait(|command_buffer| {
            self.signal_plan
                .append(command_buffer)
                .map_err(|e| SoundError::external(format!("Signal forward launch failed: {e:?}")))?;
            self.gpu.compute().record_buffer_barrier(
                command_buffer,
                &BufferBarrierSpec::compute_shader_write_to_compute_shader_read_write(self.signal_buffer.buffer),
            );

            self.append_multiply(command_buffer, &voice.impulse_response.buffer, self.fft_size as u32)
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

        self.download_convolution_output(&mut voice.complex_cpu_buffer)?;

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

    fn copy_samples(&self, complex_output: &[[f32; 2]], output_block: &mut [f32]) {
        match self.config.sound.output_channels {
            OutputChannels::Mono => {
                for i in 0..self.config.auralization.block_size {
                    output_block[i] = complex_output[i][0]; // Take left channel as mono output
                }
            }
            OutputChannels::Stereo | OutputChannels::Binaural => {
                for i in 0..self.config.auralization.block_size {
                    output_block[2 * i] = complex_output[i][0]; // Left channel
                    output_block[2 * i + 1] = complex_output[i][1]; // Right channel
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
