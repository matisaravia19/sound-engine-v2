use crate::auralization::{AudioError, FFT_SIZE, ImpulseResponseId, TAIL_SIZE, VoiceId};
use crate::gpu::GpuError;
use crate::gpu::backend::VkBackend;
use crate::gpu::compute::PipelineId;
use crate::gpu::memory::BufferHandle;
use crate::gpu::shader::{ShaderId, ShaderStage};
use ash::vk;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use vkfft_rs::plan::{DeviceHandles, FFTPlan, FFTPlanBuilder};

const MULTIPLY_SHADER_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/auralization/shaders/multiply.comp.glsl"
);

pub(crate) struct PartitionedConvolver {
    gpu: Arc<VkBackend>,
    id_counter: AtomicU64,
    voices: HashMap<VoiceId, Voice>,
    impulse_responses: HashMap<ImpulseResponseId, Arc<ImpulseResponse>>,
    signal_plan: FFTPlan,
    signal_buffer: BufferHandle,
    multiply_pipeline_id: PipelineId,
}

struct Voice {
    impulse_response: Arc<ImpulseResponse>,
    tail: Vec<f32>,
    convolution_output: Vec<f32>,
}

struct ImpulseResponse {
    buffer: BufferHandle,
}

impl PartitionedConvolver {
    pub(crate) fn new(gpu: Arc<VkBackend>) -> Result<Self, AudioError> {
        let (signal_plan, signal_buffer) = Self::build_fft_plan(gpu.as_ref())?;
        let multiply_pipeline_id = Self::create_multiply_pipeline(gpu.as_ref())?;

        Ok(Self {
            gpu,
            id_counter: AtomicU64::new(0),
            voices: HashMap::new(),
            impulse_responses: HashMap::new(),
            signal_plan,
            signal_buffer,
            multiply_pipeline_id,
        })
    }

    pub(super) fn register_impulse_response(&mut self, id: ImpulseResponseId, data: &[f32]) -> Result<(), AudioError> {
        let (plan, buffer) = Self::build_fft_plan(self.gpu.as_ref())?;

        let complex_data = pack_real_as_complex(data);
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

    pub(super) fn forget_impulse_response(&mut self, id: ImpulseResponseId) {
        self.impulse_responses.remove(&id);
    }

    pub(super) fn start_sound(&mut self, ir_id: ImpulseResponseId) -> Result<VoiceId, AudioError> {
        let ir = self
            .impulse_responses
            .get(&ir_id)
            .ok_or_else(|| std::io::Error::other(format!("Unknown impulse response id {ir_id}")))?;

        let voice_id = self.id_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        self.voices.insert(
            voice_id,
            Voice {
                impulse_response: ir.clone(),
                tail: vec![0.0; TAIL_SIZE],
                convolution_output: vec![0.0; FFT_SIZE as usize],
            },
        );

        Ok(voice_id)
    }

    pub(super) fn process_block(
        &mut self,
        voice_id: VoiceId,
        input_block: &[f32],
        output_block: &mut [f32],
    ) -> Result<(), AudioError> {
        let voice = self
            .voices
            .get_mut(&voice_id)
            .ok_or_else(|| std::io::Error::other(format!("Unknown voice id {voice_id}")))?;

        Ok(())
    }

    pub(super) fn multiply_signal_with_ir(&self, ir_id: ImpulseResponseId, count: u32) -> Result<(), AudioError> {
        let ir = self
            .impulse_responses
            .get(&ir_id)
            .ok_or_else(|| std::io::Error::other(format!("Unknown impulse response id {ir_id}")))?;
        self.launch_multiply(&ir.buffer, count)
    }

    fn launch_multiply(&self, ir_buffer: &BufferHandle, count: u32) -> Result<(), AudioError> {
        self.gpu
            .compute()
            .dispatch_multiply(
                self.multiply_pipeline_id,
                self.signal_buffer.buffer,
                ir_buffer.buffer,
                count,
            )
            .map_err(map_gpu_error)
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
                .create_storage_buffer(FFT_SIZE * std::mem::size_of::<f32>() as u64 * 2) // Placeholder size, will be updated per voice
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

        let descriptor_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        gpu.compute()
            .create_pipeline(multiply_shader_id, &descriptor_bindings)
            .map_err(map_gpu_error)
    }
}

fn pack_real_as_complex(input: &[f32]) -> Vec<f32> {
    let mut packed = vec![0.0_f32; input.len() * 2];
    for (i, &sample) in input.iter().enumerate() {
        packed[2 * i] = sample;
    }
    packed
}

fn map_gpu_error(err: GpuError) -> AudioError {
    std::io::Error::other(err.to_string()).into()
}
