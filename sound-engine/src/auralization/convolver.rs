use crate::auralization::{AudioError, BLOCK_SIZE, FFT_SIZE, ImpulseResponseId, TAIL_SIZE, VoiceId};
use crate::gpu::GpuError;
use crate::gpu::backend::VkBackend;
use crate::gpu::compute::{
    ComputePipelineSpec, DescriptorBindingSpec, DescriptorWrite, DispatchSpec, PipelineId,
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
    pub fn new(gpu: Arc<VkBackend>) -> Result<Self, AudioError> {
        let (signal_plan, signal_buffer) = Self::build_fft_plan(gpu.as_ref())?;
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

    pub fn register_impulse_response(&mut self, id: ImpulseResponseId, data: &[f32]) -> Result<(), AudioError> {
        let (plan, buffer) = Self::build_fft_plan(self.gpu.as_ref())?;

        let mut complex_data = vec![0.0_f32; data.len() * 2];
        pack_real_as_complex(data, &mut complex_data);

        self.gpu
            .memory()
            .upload_typed(&buffer, complex_data.as_slice())
            .map_err(map_gpu_error)?;

        self.gpu
            .compute()
            .submit_compute_and_wait(|command_buffer| {
                plan.append(command_buffer)
                    .map_err(|_| std::io::Error::other("IR forward launch failed"))?;
                Ok(())
            })
            .map_err(map_gpu_error)?;

        let ir = Arc::new(ImpulseResponse { buffer });
        self.impulse_responses.insert(id, ir);

        Ok(())
    }

    pub fn forget_impulse_response(&mut self, id: ImpulseResponseId) {
        self.impulse_responses.remove(&id);
    }

    pub fn start_sound(&mut self, ir_id: ImpulseResponseId) -> Result<VoiceId, AudioError> {
        let ir = self
            .impulse_responses
            .get(&ir_id)
            .ok_or_else(|| std::io::Error::other(format!("Unknown impulse response id {ir_id}")))?;

        let voice_id = self.id_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let mut voices = self
            .voices
            .lock()
            .map_err(|_| std::io::Error::other("Voices lock is poisoned"))?;

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

    pub fn process_block(
        &self,
        voice_id: VoiceId,
        input_block: &[f32],
        output_block: &mut [f32],
    ) -> Result<(), AudioError> {
        let mut voices = self
            .voices
            .lock()
            .map_err(|_| std::io::Error::other("Voices lock is poisoned"))?;

        let voice = voices
            .get_mut(&voice_id)
            .ok_or_else(|| std::io::Error::other(format!("Unknown voice id {voice_id}")))?;

        self.convolve(voice, input_block)?;
        self.overlap_add(&mut voice.real_cpu_buffer, &mut voice.tail);

        output_block.copy_from_slice(&voice.real_cpu_buffer[..output_block.len()]);

        Ok(())
    }

    fn convolve(&self, voice: &mut Voice, input_block: &[f32]) -> Result<(), AudioError> {
        pack_real_as_complex(input_block, &mut voice.complex_cpu_buffer);
        self.upload_signal(&voice.complex_cpu_buffer)?;

        self.gpu
            .compute()
            .submit_compute_and_wait(|command_buffer| {
                self.signal_plan
                    .append(command_buffer)
                    .map_err(|e| std::io::Error::other(format!("Signal forward launch failed: {e:?}")))?;

                self.append_multiply(command_buffer, &voice.impulse_response.buffer, FFT_SIZE / 256)
                    .map_err(|e| std::io::Error::other(format!("Multiply dispatch failed: {e:?}")))?;

                self.signal_plan
                    .append_inverse(command_buffer)
                    .map_err(|e| std::io::Error::other(format!("Signal inverse launch failed: {e:?}")))?;

                Ok(())
            })
            .map_err(map_gpu_error)?;

        self.download_convolution_output(&mut voice.complex_cpu_buffer)?;
        pack_complex_as_real(&voice.complex_cpu_buffer, &mut voice.real_cpu_buffer);

        Ok(())
    }

    fn upload_signal(&self, signal: &[f32]) -> Result<(), AudioError> {
        self.gpu
            .memory()
            .clear_buffer(&self.signal_buffer)
            .map_err(map_gpu_error)?;

        self.gpu
            .memory()
            .upload_typed(&self.signal_buffer, signal)
            .map_err(map_gpu_error)
    }

    fn download_convolution_output(&self, output: &mut [f32]) -> Result<(), AudioError> {
        self.gpu
            .memory()
            .download_typed::<f32>(&self.signal_buffer, self.signal_buffer.size as usize, output)
            .map_err(map_gpu_error)
    }

    fn append_multiply(
        &self,
        command_buffer: vk::CommandBuffer,
        ir_buffer: &BufferHandle,
        count: u64,
    ) -> Result<(), AudioError> {
        let dispatch_spec = DispatchSpec {
            pipeline_id: self.multiply_pipeline_id,
            descriptor_writes: vec![
                DescriptorWrite::storage_buffer(0, self.signal_buffer.buffer),
                DescriptorWrite::storage_buffer(1, ir_buffer.buffer),
            ],
            push_constants: vec![],
            push_constant_offset: 0,
            groups: [count.div_ceil(256) as u32, 1, 1],
        };

        self.gpu
            .compute()
            .record_dispatch(command_buffer, &dispatch_spec)
            .map_err(map_gpu_error)
    }

    fn overlap_add(&self, output: &mut [f32], tail: &mut [f32]) {
        for i in 0..tail.len() {
            output[i] += tail[i];
        }

        for i in BLOCK_SIZE as usize..tail.len() - output.len() {
            tail[i] = tail[i + output.len()];
        }
    }

    fn build_fft_plan(gpu: &VkBackend) -> Result<(FFTPlan, BufferHandle), AudioError> {
        unsafe {
            let handles = DeviceHandles {
                physical_device: gpu.device().physical_device,
                device: gpu.device().device.handle(),
                queue: gpu.compute().queue_handle().map_err(map_gpu_error)?,
                command_pool: gpu.compute().command_pool_handle().map_err(map_gpu_error)?,
                fence: vk::Fence::null(), // Not used for the global signal plan
            };

            let buffer = gpu
                .memory()
                .create_storage_buffer(FFT_SIZE * 2 * std::mem::size_of::<f32>() as u64) // Placeholder size, will be updated per voice
                .map_err(map_gpu_error)?;

            let plan = FFTPlanBuilder::new(&handles)
                .with_single_dimension(buffer.size) // Placeholder, will be updated per voice
                .with_buffer(buffer.buffer, buffer.size) // Placeholder, will be updated per voice
                .with_normalization()
                .build()
                .map_err(|e| std::io::Error::other(format!("Global signal plan build failed: {e:?}")))?;

            Ok((plan, buffer))
        }
    }

    fn create_multiply_pipeline(gpu: &VkBackend) -> Result<PipelineId, AudioError> {
        let multiply_shader_id = gpu
            .shaders()
            .load_glsl_file(ShaderStage::Compute, MULTIPLY_SHADER_PATH)
            .map_err(map_gpu_error)?;

        let pipeline_spec = ComputePipelineSpec {
            shader_id: multiply_shader_id,
            descriptor_bindings: vec![
                DescriptorBindingSpec::storage_buffer(0),
                DescriptorBindingSpec::storage_buffer(1),
            ],
            push_constant_ranges: vec![],
        };

        gpu.compute().create_pipeline(pipeline_spec).map_err(map_gpu_error)
    }
}

fn pack_real_as_complex(input: &[f32], output: &mut [f32]) {
    for (i, &sample) in input.iter().enumerate() {
        output[2 * i] = sample;
    }
}

fn pack_complex_as_real(input: &[f32], output: &mut [f32]) {
    for (i, chunk) in input.chunks_exact(2).enumerate() {
        output[i] = chunk[0];
    }
}

fn map_gpu_error(err: GpuError) -> AudioError {
    std::io::Error::other(err.to_string()).into()
}
