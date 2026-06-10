use crate::auralization::{BLOCK_SIZE, FFT_SIZE, ImpulseResponseId, TAIL_SIZE, VoiceId};
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
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};
use vkfft_rs::plan::{DeviceHandles, FFTPlan, FFTPlanBuilder};

const MULTIPLY_SHADER_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/auralization/shaders/multiply.comp.glsl"
);

/// GPU-backed partitioned convolver for mono voices and impulse responses.
///
/// Each registered IR is transformed to the frequency domain once. Playback
/// transforms each input block, multiplies it by the IR spectrum, runs the
/// inverse transform, and applies overlap-add.
pub struct PartitionedConvolver {
    gpu: Arc<VkBackend>,
    id_counter: AtomicU64,
    voices: Mutex<HashMap<VoiceId, Voice>>,
    impulse_responses: HashMap<ImpulseResponseId, Arc<ImpulseResponse>>,
    signal_plan: FFTPlan,
    signal_buffer: BufferHandle,
    multiply_pipeline_id: PipelineId,
}

struct Voice {
    impulse_response: Arc<ImpulseResponse>,
    tail: Vec<f32>,
    complex_cpu_buffer: Vec<f32>,
    real_cpu_buffer: Vec<f32>,
}

struct ImpulseResponse {
    buffer: BufferHandle,
}

impl PartitionedConvolver {
    /// Creates the shared FFT plan and multiply compute pipeline.
    pub fn new(gpu: Arc<VkBackend>) -> SoundResult<Self> {
        let (signal_plan, signal_buffer) = Self::build_fft_plan(gpu.as_ref(), true)?;
        let multiply_pipeline_id = Self::create_multiply_pipeline(gpu.as_ref())?;

        Ok(Self {
            gpu,
            id_counter: AtomicU64::new(0),
            voices: Mutex::new(HashMap::new()),
            impulse_responses: HashMap::new(),
            signal_plan,
            signal_buffer,
            multiply_pipeline_id,
        })
    }

    /// Registers an impulse response and uploads its frequency-domain form.
    pub fn register_impulse_response(&mut self, id: ImpulseResponseId, data: &[f32]) -> SoundResult<()> {
        crate::debug_validate!(
            !data.is_empty(),
            ErrorCode::InvalidArgument,
            "Impulse response must not be empty"
        );
        crate::debug_validate!(
            data.len() <= FFT_SIZE as usize,
            ErrorCode::InvalidArgument,
            "Impulse response length {} exceeds FFT size {}",
            data.len(),
            FFT_SIZE
        );

        let (plan, buffer) = Self::build_fft_plan(self.gpu.as_ref(), false)?;

        let mut complex_data = vec![0.0_f32; data.len() * 2];
        pack_real_as_complex(data, &mut complex_data);

        self.gpu.memory().clear_buffer(&buffer)?;

        self.gpu.memory().upload_typed(&buffer, complex_data.as_slice())?;

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

        let voice_id = self.id_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let mut voices = self
            .voices
            .lock()
            .map_err(|_| SoundError::poisoned_lock("Voices lock is poisoned"))?;

        voices.insert(
            voice_id,
            Voice {
                impulse_response: ir.clone(),
                tail: vec![0.0; TAIL_SIZE],
                real_cpu_buffer: vec![0.0; FFT_SIZE as usize],
                complex_cpu_buffer: vec![0.0; (FFT_SIZE * 2) as usize],
            },
        );

        Ok(voice_id)
    }

    /// Processes one mono input block for a voice into `output_block`.
    pub fn process_block(&self, voice_id: VoiceId, input_block: &[f32], output_block: &mut [f32]) -> SoundResult<()> {
        crate::debug_validate!(
            input_block.len() == BLOCK_SIZE as usize,
            ErrorCode::InvalidArgument,
            "input_block length {} must equal BLOCK_SIZE {}",
            input_block.len(),
            BLOCK_SIZE
        );
        crate::debug_validate!(
            output_block.len() <= FFT_SIZE as usize,
            ErrorCode::InvalidArgument,
            "output_block length {} exceeds FFT_SIZE {}",
            output_block.len(),
            FFT_SIZE
        );

        let mut voices = self
            .voices
            .lock()
            .map_err(|_| SoundError::poisoned_lock("Voices lock is poisoned"))?;

        let voice = voices
            .get_mut(&voice_id)
            .ok_or_else(|| SoundError::not_found(format!("Unknown voice id {voice_id}")))?;

        self.convolve(voice, input_block)?;
        self.overlap_add(&mut voice.real_cpu_buffer, &mut voice.tail);

        output_block.copy_from_slice(&voice.real_cpu_buffer[..output_block.len()]);

        Ok(())
    }

    fn convolve(&self, voice: &mut Voice, input_block: &[f32]) -> SoundResult<()> {
        pack_real_as_complex(input_block, &mut voice.complex_cpu_buffer);
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

            self.append_multiply(command_buffer, &voice.impulse_response.buffer, FFT_SIZE)
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
        pack_complex_as_real(&voice.complex_cpu_buffer, &mut voice.real_cpu_buffer);

        Ok(())
    }

    fn upload_signal(&self, signal: &[f32]) -> SoundResult<()> {
        self.gpu.memory().clear_buffer(&self.signal_buffer)?;

        self.gpu.memory().upload_typed(&self.signal_buffer, signal)
    }

    fn download_convolution_output(&self, output: &mut [f32]) -> SoundResult<()> {
        self.gpu
            .memory()
            .download_typed::<f32>(&self.signal_buffer, output.len(), output)
    }

    fn append_multiply(
        &self,
        command_buffer: vk::CommandBuffer,
        ir_buffer: &BufferHandle,
        count: u64,
    ) -> SoundResult<()> {
        let dispatch_spec = DispatchSpec {
            pipeline_id: self.multiply_pipeline_id,
            descriptor_writes: vec![
                DescriptorWrite::storage_buffer(0, self.signal_buffer.buffer),
                DescriptorWrite::storage_buffer(1, ir_buffer.buffer),
            ],
            push_constants: (count as u32).to_ne_bytes().to_vec(),
            push_constant_offset: 0,
            groups: [count.div_ceil(256) as u32, 1, 1],
        };

        self.gpu.compute().record_dispatch(command_buffer, &dispatch_spec)
    }

    fn overlap_add(&self, output: &mut [f32], tail: &mut [f32]) {
        // Add previous block tail into the current block head.
        for i in 0..tail.len() {
            output[i] += tail[i];
        }

        // Store new overlap tail from current convolution result.
        let tail_start = BLOCK_SIZE as usize;
        let tail_end = tail_start + tail.len();
        tail.copy_from_slice(&output[tail_start..tail_end]);
    }

    fn build_fft_plan(gpu: &VkBackend, normalize: bool) -> SoundResult<(FFTPlan, BufferHandle)> {
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
                .create_storage_buffer(FFT_SIZE * 2 * std::mem::size_of::<f32>() as u64)?; // Placeholder size, will be updated per voice

            let builder = FFTPlanBuilder::new(&handles)
                .with_single_dimension(FFT_SIZE)
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

fn pack_real_as_complex(input: &[f32], output: &mut [f32]) {
    debug_assert!(output.len() >= input.len() * 2);
    output.fill(0.0);
    for (i, &sample) in input.iter().enumerate() {
        output[2 * i] = sample;
    }
}

fn pack_complex_as_real(input: &[f32], output: &mut [f32]) {
    debug_assert!(input.len() >= output.len() * 2);
    for (i, chunk) in input.chunks_exact(2).enumerate() {
        if i == output.len() {
            break;
        }
        output[i] = chunk[0];
    }
}
