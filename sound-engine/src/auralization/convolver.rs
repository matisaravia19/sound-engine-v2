use crate::auralization::{AudioError, FFT_SIZE, ImpulseResponseId, IrSnapshot, VoiceId};
use crate::gpu::GpuError;
use crate::gpu::backend::VkBackend;
use crate::gpu::memory::BufferHandle;
use ash::vk;
use std::collections::HashMap;
use std::sync::Arc;
use vkfft_rs::plan::{DeviceHandles, FFTPlan, FFTPlanBuilder};

pub(crate) struct PartitionedConvolver {
    gpu: Arc<VkBackend>,
    voices: HashMap<VoiceId, Voice>,
    impulse_responses: HashMap<ImpulseResponseId, Arc<ImpulseResponse>>,
    signal_plan: FFTPlan,
    signal_buffer: BufferHandle,
}

struct Voice {
    impulse_response: Arc<ImpulseResponse>,
}

struct ImpulseResponse {
    buffer: BufferHandle,
}

impl PartitionedConvolver {
    pub(crate) fn new(gpu: Arc<VkBackend>) -> Result<Self, AudioError> {
        let (signal_plan, signal_buffer) = Self::build_fft_plan(gpu.as_ref())?;

        Ok(Self {
            gpu,
            voices: HashMap::new(),
            impulse_responses: HashMap::new(),
            signal_plan,
            signal_buffer,
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

    // pub(crate) fn ensure_ir(&mut self, voice_id: VoiceId, ir: &Arc<IrSnapshot>) -> Result<(), AudioError> {
    //     if ir.sample_rate != self.sample_rate {
    //         return Err(std::io::Error::other(format!(
    //             "IR sample rate mismatch: got {}, expected {}",
    //             ir.sample_rate, self.sample_rate
    //         ))
    //         .into());
    //     }

    //     if ir.samples.is_empty() {
    //         return Err(std::io::Error::other("IR cannot be empty").into());
    //     }

    //     self.voice_ir.entry(voice_id).or_insert_with(|| Arc::clone(ir));

    //     let required_fft_len = (self.block_size + ir.samples.len() - 1).next_power_of_two();
    //     let needs_rebuild = match self.voice_state.get(&voice_id) {
    //         Some(state) => !Arc::ptr_eq(&state.ir, ir) || state.fft_len != required_fft_len,
    //         None => true,
    //     };

    //     if needs_rebuild {
    //         if let Some(old_state) = self.voice_state.remove(&voice_id) {
    //             old_state.destroy(&mut self.gpu)?;
    //         }

    //         let state = VoiceFftState::new(Arc::clone(ir), required_fft_len, &mut self.gpu)?;
    //         state.upload_real_to_ir_buffer(&mut self.gpu, &ir.samples)?;
    //         state.launch_plan(&self.gpu, false, true)?;

    //         self.voice_state.insert(voice_id, state);
    //         self.voice_ir.insert(voice_id, Arc::clone(ir));
    //     }

    //     Ok(())
    // }

    // pub(crate) fn process_voice(
    //     &mut self,
    //     voice_id: VoiceId,
    //     dry_block: &[f32],
    //     wet_block: &mut [f32],
    // ) -> Result<(), AudioError> {
    //     if dry_block.len() != self.block_size || wet_block.len() != self.block_size {
    //         return Err(std::io::Error::other(format!(
    //             "Block size mismatch in convolver: dry {}, wet {}, expected {}",
    //             dry_block.len(),
    //             wet_block.len(),
    //             self.block_size
    //         ))
    //         .into());
    //     }

    //     if !self.voice_ir.contains_key(&voice_id) {
    //         return Err(std::io::Error::other(format!("IR not set for voice {}", voice_id)).into());
    //     }

    //     let state = self
    //         .voice_state
    //         .get(&voice_id)
    //         .ok_or_else(|| std::io::Error::other(format!("VkFFT state not initialized for voice {}", voice_id)))?;

    //     state.upload_real_to_signal_buffer(&mut self.gpu, dry_block)?;
    //     state.launch_plan(&self.gpu, false, false)?;
    //     state.multiply_signal_with_ir_spectrum(&mut self.gpu)?;
    //     state.launch_plan(&self.gpu, true, false)?;
    //     state.download_real_from_signal_buffer(&mut self.gpu, wet_block)?;

    //     Ok(())
    // }

    // pub(crate) fn forget_voice(&mut self, voice_id: VoiceId) {
    //     self.voice_ir.remove(&voice_id);
    //     if let Some(state) = self.voice_state.remove(&voice_id) {
    //         let _ = state.destroy(&mut self.gpu);
    //     }
    // }

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
}

// impl VoiceFftState {
//     fn new(ir: Arc<IrSnapshot>, fft_len: usize, gpu: &mut VkBackend) -> Result<Self, AudioError> {
//         if fft_len == 0 {
//             return Err(std::io::Error::other("fft_len must be > 0").into());
//         }

//         let element_count = fft_len
//             .checked_mul(2)
//             .ok_or_else(|| std::io::Error::other("fft_len overflow"))?;
//         let buffer_bytes = (element_count
//             .checked_mul(std::mem::size_of::<f32>())
//             .ok_or_else(|| std::io::Error::other("buffer size overflow"))?) as u64;

//         let signal_buffer = gpu
//             .memory_mut()
//             .create_storage_buffer(buffer_bytes)
//             .map_err(map_gpu_error)?;
//         let ir_buffer = gpu
//             .memory_mut()
//             .create_storage_buffer(buffer_bytes)
//             .map_err(map_gpu_error)?;

//         let device_ctx = gpu.device();

//         let fence = unsafe { device_ctx.device.create_fence(&vk::FenceCreateInfo::default(), None)? };

//         let command_buffer = {
//             let alloc_info = vk::CommandBufferAllocateInfo::default()
//                 .command_pool(device_ctx.queues.compute_command_pool)
//                 .level(vk::CommandBufferLevel::PRIMARY)
//                 .command_buffer_count(1);
//             unsafe { device_ctx.device.allocate_command_buffers(&alloc_info)?[0] }
//         };

//         let mut signal_builder = FFTPlanBuilder::new(&DeviceHandles {
//             physical_device: device_ctx.physical_device,
//             device: device_ctx.device.handle(),
//             queue: device_ctx.queues.compute_queue,
//             command_pool: device_ctx.queues.compute_command_pool,
//             fence,
//         })
//         .map_err(|e| std::io::Error::other(format!("PlanBuilder::new (signal) failed: {e:?}")))?;

//         signal_builder
//             .with_dimensions(&[fft_len as u64])
//             .with_buffer(signal_buffer.buffer, buffer_bytes);
//         let signal_plan = signal_builder
//             .build()
//             .map_err(|e| std::io::Error::other(format!("PlanBuilder::build (signal) failed: {e:?}")))?;

//         let mut ir_builder = FFTPlanBuilder::new(&DeviceHandles {
//             physical_device: device_ctx.physical_device,
//             device: device_ctx.device.handle(),
//             queue: device_ctx.queues.compute_queue,
//             command_pool: device_ctx.queues.compute_command_pool,
//             fence,
//         })
//         .map_err(|e| std::io::Error::other(format!("PlanBuilder::new (ir) failed: {e:?}")))?;

//         ir_builder
//             .with_dimensions(&[fft_len as u64])
//             .with_buffer(ir_buffer.buffer, buffer_bytes);
//         let ir_plan = ir_builder
//             .build()
//             .map_err(|e| std::io::Error::other(format!("PlanBuilder::build (ir) failed: {e:?}")))?;

//         Ok(Self {
//             ir,
//             fft_len,
//             buffer_bytes,
//             signal_buffer,
//             ir_buffer,
//             command_buffer,
//             fence,
//             signal_plan: Some(signal_plan),
//             ir_plan: Some(ir_plan),
//             signal_builder: Some(signal_builder),
//             ir_builder: Some(ir_builder),
//         })
//     }

//     fn destroy(mut self, gpu: &mut VkBackend) -> Result<(), AudioError> {
//         self.signal_plan.take();
//         self.ir_plan.take();
//         self.signal_builder.take();
//         self.ir_builder.take();

//         {
//             let device_ctx = gpu.device();
//             unsafe {
//                 let _ = device_ctx.device.device_wait_idle();
//                 device_ctx.device.free_command_buffers(
//                     device_ctx.queues.compute_command_pool,
//                     std::slice::from_ref(&self.command_buffer),
//                 );
//                 device_ctx.device.destroy_fence(self.fence, None);
//             }
//         }

//         gpu.memory_mut().destroy_buffer(self.signal_buffer);
//         gpu.memory_mut().destroy_buffer(self.ir_buffer);

//         Ok(())
//     }

//     fn upload_real_to_signal_buffer(&self, gpu: &mut VkBackend, input: &[f32]) -> Result<(), AudioError> {
//         let packed = pack_real_as_complex(input, self.fft_len);
//         gpu.memory_mut()
//             .upload_bytes(&self.signal_buffer, f32_slice_to_bytes(&packed))
//             .map_err(map_gpu_error)?;
//         Ok(())
//     }

//     fn upload_real_to_ir_buffer(&self, gpu: &mut VkBackend, input: &[f32]) -> Result<(), AudioError> {
//         let packed = pack_real_as_complex(input, self.fft_len);
//         gpu.memory_mut()
//             .upload_bytes(&self.ir_buffer, f32_slice_to_bytes(&packed))
//             .map_err(map_gpu_error)?;
//         Ok(())
//     }

//     fn download_real_from_signal_buffer(&self, gpu: &mut VkBackend, output: &mut [f32]) -> Result<(), AudioError> {
//         let raw = gpu
//             .memory_mut()
//             .download_bytes(&self.signal_buffer, self.buffer_bytes as usize)
//             .map_err(map_gpu_error)?;
//         let complex = bytes_to_f32_vec(&raw)?;

//         let scale = 1.0_f32 / (self.fft_len as f32);
//         for (i, out) in output.iter_mut().enumerate() {
//             let idx = 2 * i;
//             *out = if idx < complex.len() { complex[idx] * scale } else { 0.0 };
//         }

//         Ok(())
//     }

//     fn launch_plan(&self, gpu: &VkBackend, inverse: bool, use_ir_buffer: bool) -> Result<(), AudioError> {
//         let device_ctx = gpu.device();

//         unsafe {
//             device_ctx
//                 .device
//                 .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;

//             let begin_info = vk::CommandBufferBeginInfo::default();
//             device_ctx
//                 .device
//                 .begin_command_buffer(self.command_buffer, &begin_info)?;

//             let plan = if use_ir_buffer {
//                 self.ir_plan
//                     .as_ref()
//                     .ok_or_else(|| std::io::Error::other("IR VkFFT plan missing"))?
//             } else {
//                 self.signal_plan
//                     .as_ref()
//                     .ok_or_else(|| std::io::Error::other("Signal VkFFT plan missing"))?
//             };

//             if inverse {
//                 plan.append_inverse(self.command_buffer)
//                     .map_err(|e| std::io::Error::other(format!("VkFFT inverse launch failed: {e:?}")))?;
//             } else {
//                 plan.append(self.command_buffer)
//                     .map_err(|e| std::io::Error::other(format!("VkFFT forward launch failed: {e:?}")))?;
//             }

//             device_ctx.device.end_command_buffer(self.command_buffer)?;
//         }

//         let token = gpu.compute().submit_compute(self.command_buffer).map_err(map_gpu_error)?;
//         gpu.compute().wait_for(token).map_err(map_gpu_error)?;

//         Ok(())
//     }

//     fn multiply_signal_with_ir_spectrum(&self, gpu: &mut VkBackend) -> Result<(), AudioError> {
//         let signal_raw = gpu
//             .memory_mut()
//             .download_bytes(&self.signal_buffer, self.buffer_bytes as usize)
//             .map_err(map_gpu_error)?;
//         let ir_raw = gpu
//             .memory_mut()
//             .download_bytes(&self.ir_buffer, self.buffer_bytes as usize)
//             .map_err(map_gpu_error)?;

//         let mut signal = bytes_to_f32_vec(&signal_raw)?;
//         let ir = bytes_to_f32_vec(&ir_raw)?;

//         for k in 0..(signal.len() / 2) {
//             let idx = 2 * k;
//             let ar = signal[idx];
//             let ai = signal[idx + 1];
//             let br = ir[idx];
//             let bi = ir[idx + 1];

//             signal[idx] = ar * br - ai * bi;
//             signal[idx + 1] = ar * bi + ai * br;
//         }

//         gpu.memory_mut()
//             .upload_bytes(&self.signal_buffer, f32_slice_to_bytes(&signal))
//             .map_err(map_gpu_error)?;

//         Ok(())
//     }
// }

// impl Drop for PartitionedConvolver {
//     fn drop(&mut self) {
//         let mut pending = HashMap::new();
//         std::mem::swap(&mut pending, &mut self.voice_state);
//         for (_, state) in pending {
//             let _ = state.destroy(&mut self.gpu);
//         }
//     }
// }

fn pack_real_as_complex(input: &[f32]) -> Vec<f32> {
    let mut packed = vec![0.0_f32; input.len() * 2];
    for (i, &sample) in input.iter().enumerate() {
        packed[2 * i] = sample;
    }
    packed
}

fn f32_slice_to_bytes(data: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data)) }
}

fn bytes_to_f32_vec(bytes: &[u8]) -> Result<Vec<f32>, AudioError> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<f32>()) {
        return Err(
            std::io::Error::other(format!("Byte buffer length {} is not aligned to f32 size", bytes.len())).into(),
        );
    }

    let count = bytes.len() / std::mem::size_of::<f32>();
    let mut out = vec![0.0_f32; count];
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr().cast::<u8>(), bytes.len());
    }
    Ok(out)
}

fn map_gpu_error(err: GpuError) -> AudioError {
    std::io::Error::other(err.to_string()).into()
}
