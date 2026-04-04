use crate::gpu::buffer::GpuBuffer;
use crate::gpu::context::GpuContext;
use crate::gpu::errors::GpuError;
use ash::vk;
use std::mem::size_of;
use vkfft_rs::plan::{DeviceHandles, Plan, PlanBuilder};

pub struct Auralizer {
    block_size: usize,
    ir_size: usize,
    fft_size: usize,
    buffer_bytes: u64,
    context: GpuContext,
    signal_buffer: Option<GpuBuffer>,
    fence: Option<vk::Fence>,
    command_buffer: Option<vk::CommandBuffer>,
    signal_plan: Option<Plan>,
}

impl Auralizer {
    pub fn new(block_size: usize, ir_size: usize) -> Result<Self, GpuError> {
        if block_size == 0 {
            return Err(std::io::Error::other("block_size must be greater than zero").into());
        }
        if ir_size == 0 {
            return Err(std::io::Error::other("ir_size must be greater than zero").into());
        }

        let fft_size = (block_size + ir_size - 1).next_power_of_two();
        let buffer_bytes = (fft_size * 2 * size_of::<f32>()) as u64;

        let context = GpuContext::init()?;
        let signal_buffer = context.create_buffer(buffer_bytes)?;
        let fence = context.create_fence()?;
        let command_buffer = allocate_command_buffer(&context)?;

        let device_handles = DeviceHandles {
            physical_device: context.physical_device().clone(),
            device: context.device().handle(),
            queue: context.compute_queue().clone(),
            command_pool: context.compute_command_pool().clone(),
            fence,
        };

        let mut signal_builder = PlanBuilder::new(DeviceHandles { ..device_handles })
            .map_err(|e| std::io::Error::other(format!("Signal PlanBuilder::new failed: {e:?}")))?;

        signal_builder
            .with_dimensions(&[fft_size as u64])
            .with_buffer(signal_buffer.buffer(), buffer_bytes)
            .with_normalization();

        let signal_plan = signal_builder
            .build()
            .map_err(|e| std::io::Error::other(format!("Signal PlanBuilder::build failed: {e:?}")))?;

        Ok(Self {
            block_size,
            ir_size,
            fft_size,
            buffer_bytes,
            context,
            signal_buffer: Some(signal_buffer),
            fence: Some(fence),
            command_buffer: Some(command_buffer),
            signal_plan: Some(signal_plan),
        })
    }

    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    pub fn expected_ir_fft_len(&self) -> usize {
        self.fft_size * 2
    }

    pub fn process_full(&mut self, input: &[f32], ir_fft: &[f32]) -> Result<Vec<f32>, GpuError> {
        let expected_ir_fft_len = self.expected_ir_fft_len();
        if ir_fft.len() != expected_ir_fft_len {
            return Err(std::io::Error::other(format!(
                "ir_fft length mismatch: got {}, expected {expected_ir_fft_len}",
                ir_fft.len()
            ))
            .into());
        }

        if input.is_empty() {
            return Ok(Vec::new());
        }

        let signal_buffer = self
            .signal_buffer
            .take()
            .ok_or_else(|| std::io::Error::other("signal buffer not available"))?;

        let result = (|| -> Result<Vec<f32>, GpuError> {
        let output_len = input.len() + self.ir_size - 1;
        let mut output = vec![0.0_f32; output_len];
        let mut overlap = vec![0.0_f32; self.ir_size - 1];

        let mut block_start = 0usize;
        while block_start < input.len() {
            let block_end = (block_start + self.block_size).min(input.len());
            let input_block = &input[block_start..block_end];
            let block_len = input_block.len();

            let packed_block = pack_real_as_complex(input_block, self.fft_size);
            self.context
                .upload_to_buffer(&signal_buffer, f32_as_bytes(&packed_block))?;

            self.submit_fft_forward()?;

            let freq_bytes = self
                .context
                .download_from_buffer(&signal_buffer, self.buffer_bytes as usize)?;
            let mut block_fft = bytes_to_f32_vec(&freq_bytes)?;
            complex_multiply_in_place(&mut block_fft, ir_fft)?;

            self.context
                .upload_to_buffer(&signal_buffer, f32_as_bytes(&block_fft))?;

            self.submit_fft_inverse()?;

            let ifft_bytes = self
                .context
                .download_from_buffer(&signal_buffer, self.buffer_bytes as usize)?;
            let block_ifft = bytes_to_f32_vec(&ifft_bytes)?;

            let conv_block_len = block_len + self.ir_size - 1;
            let mut block_time = extract_real_samples(&block_ifft, conv_block_len)?;
            for i in 0..overlap.len() {
                block_time[i] += overlap[i];
            }

            output[block_start..(block_start + block_len)].copy_from_slice(&block_time[..block_len]);
            for i in 0..overlap.len() {
                overlap[i] = block_time[block_len + i];
            }

            block_start += self.block_size;
        }

        for (i, &sample) in overlap.iter().enumerate() {
            output[input.len() + i] = sample;
        }

        Ok(output)
        })();

        self.signal_buffer = Some(signal_buffer);
        result
    }

    fn signal_plan_ref(&self) -> Result<&Plan, GpuError> {
        self.signal_plan
            .as_ref()
            .ok_or_else(|| std::io::Error::other("signal plan not available").into())
    }

    fn command_buffer(&self) -> Result<vk::CommandBuffer, GpuError> {
        self.command_buffer
            .ok_or_else(|| std::io::Error::other("command buffer not available").into())
    }

    fn fence(&self) -> Result<vk::Fence, GpuError> {
        self.fence
            .ok_or_else(|| std::io::Error::other("fence not available").into())
    }

    fn submit_fft_forward(&self) -> Result<(), GpuError> {
        let command_buffer = self.command_buffer()?;
        let fence = self.fence()?;

        unsafe {
            self.context
                .device()
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;

            let begin_info =
                vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.context.device().begin_command_buffer(command_buffer, &begin_info)?;
        }

        self.signal_plan_ref()?
            .launch(command_buffer)
            .map_err(|e| std::io::Error::other(format!("Signal forward launch failed: {e:?}")))?;

        unsafe {
            self.context.device().end_command_buffer(command_buffer)?;
        }

        self.context.reset_fence(fence)?;
        let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));

        unsafe {
            self.context.device().queue_submit(
                self.context.compute_queue().clone(),
                std::slice::from_ref(&submit_info),
                fence,
            )?;
        }

        self.context.wait_for_fence(fence)?;
        Ok(())
    }

    fn submit_fft_inverse(&self) -> Result<(), GpuError> {
        let command_buffer = self.command_buffer()?;
        let fence = self.fence()?;

        unsafe {
            self.context
                .device()
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;

            let begin_info =
                vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.context.device().begin_command_buffer(command_buffer, &begin_info)?;
        }

        self.signal_plan_ref()?
            .launch_inverse(command_buffer)
            .map_err(|e| std::io::Error::other(format!("Signal inverse launch failed: {e:?}")))?;

        unsafe {
            self.context.device().end_command_buffer(command_buffer)?;
        }

        self.context.reset_fence(fence)?;
        let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));

        unsafe {
            self.context.device().queue_submit(
                self.context.compute_queue().clone(),
                std::slice::from_ref(&submit_info),
                fence,
            )?;
        }

        self.context.wait_for_fence(fence)?;
        Ok(())
    }
}

impl Drop for Auralizer {
    fn drop(&mut self) {
        let _ = unsafe { self.context.device().device_wait_idle() };

        if let Some(plan) = self.signal_plan.take() {
            drop(plan);
        }

        if let Some(command_buffer) = self.command_buffer.take() {
            unsafe {
                self.context.device().free_command_buffers(
                    self.context.compute_command_pool().clone(),
                    std::slice::from_ref(&command_buffer),
                );
            }
        }

        if let Some(buffer) = self.signal_buffer.take() {
            self.context.destroy_buffer(buffer);
        }

        if let Some(fence) = self.fence.take() {
            self.context.destroy_fence(fence);
        }
    }
}

fn allocate_command_buffer(context: &GpuContext) -> Result<vk::CommandBuffer, GpuError> {
    let allocation_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(context.compute_command_pool().clone())
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let command_buffer = unsafe { context.device().allocate_command_buffers(&allocation_info)? }[0];
    Ok(command_buffer)
}

fn complex_multiply_in_place(lhs: &mut [f32], rhs: &[f32]) -> Result<(), GpuError> {
    if lhs.len() != rhs.len() || lhs.len() % 2 != 0 {
        return Err(std::io::Error::other("Complex buffers must have equal even lengths").into());
    }

    for i in (0..lhs.len()).step_by(2) {
        let ar = lhs[i];
        let ai = lhs[i + 1];
        let br = rhs[i];
        let bi = rhs[i + 1];

        lhs[i] = ar * br - ai * bi;
        lhs[i + 1] = ar * bi + ai * br;
    }

    Ok(())
}

fn pack_real_as_complex(input: &[f32], fft_size: usize) -> Vec<f32> {
    let mut packed = vec![0.0_f32; fft_size * 2];
    for (i, &sample) in input.iter().enumerate() {
        packed[2 * i] = sample;
    }
    packed
}

fn f32_as_bytes(samples: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(samples.as_ptr() as *const u8, std::mem::size_of_val(samples)) }
}

fn bytes_to_f32_vec(bytes: &[u8]) -> Result<Vec<f32>, GpuError> {
    if bytes.len() % size_of::<f32>() != 0 {
        return Err(std::io::Error::other("GPU readback size is not aligned to f32 elements").into());
    }

    let mut out = Vec::with_capacity(bytes.len() / size_of::<f32>());
    for chunk in bytes.chunks_exact(size_of::<f32>()) {
        out.push(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }

    Ok(out)
}

fn extract_real_samples(complex_data: &[f32], count: usize) -> Result<Vec<f32>, GpuError> {
    if complex_data.len() < count * 2 {
        return Err(std::io::Error::other("GPU output buffer is smaller than expected").into());
    }

    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        out.push(complex_data[2 * i]);
    }

    Ok(out)
}
