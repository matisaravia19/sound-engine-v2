use ash::vk;
use sound_engine::gpu::context::GpuContext;
use sound_engine::gpu::errors::GpuError;
use std::mem::size_of;
use vkfft_rs::plan::{DeviceHandles, PlanBuilder};

fn main() -> Result<(), GpuError> {
    run()
}

fn run() -> Result<(), GpuError> {
    // Deterministic test signal and IR: y[n] = x[n] + 0.5 * x[n-64]
    let signal: Vec<f32> = (0..512)
        .map(|i| ((i as f32) * 0.03125).sin() + ((i % 13) as f32 * 0.01))
        .collect();
    let mut ir = vec![0.0_f32; 65];
    ir[0] = 1.0;
    ir[64] = 0.5;

    let expected = cpu_linear_convolution(&signal, &ir);

    let conv_len = signal.len() + ir.len() - 1;
    let fft_len = conv_len.next_power_of_two();

    let packed_signal = pack_real_as_complex(&signal, fft_len);
    let packed_kernel = pack_real_as_complex(&ir, fft_len);
    let buffer_bytes = (packed_signal.len() * size_of::<f32>()) as u64;

    let mut context = GpuContext::init()?;
    let signal_buffer = context.create_buffer(buffer_bytes)?;
    let kernel_buffer = context.create_buffer(buffer_bytes)?;

    context.upload_to_buffer(&signal_buffer, f32_as_bytes(&packed_signal))?;
    context.upload_to_buffer(&kernel_buffer, f32_as_bytes(&packed_kernel))?;

    let fence = context.create_fence()?;

    let device_handles = DeviceHandles {
        physical_device: context.physical_device().clone(),
        device: context.device().handle(),
        queue: context.compute_queue().clone(),
        command_pool: context.compute_command_pool().clone(),
        fence,
    };

    let mut kernel_builder = PlanBuilder::new(DeviceHandles { ..device_handles })
        .map_err(|e| std::io::Error::other(format!("Kernel PlanBuilder::new failed: {e:?}")))?;
    kernel_builder
        .with_dimensions(&[fft_len as u64])
        .with_buffer(kernel_buffer.buffer(), buffer_bytes);

    let kernel_plan = kernel_builder
        .build()
        .map_err(|e| std::io::Error::other(format!("Kernel PlanBuilder::build failed: {e:?}")))?;

    let mut signal_builder = PlanBuilder::new(DeviceHandles { ..device_handles })
        .map_err(|e| std::io::Error::other(format!("Signal PlanBuilder::new failed: {e:?}")))?;
    signal_builder
        .with_dimensions(&[fft_len as u64])
        .with_buffer(signal_buffer.buffer(), buffer_bytes)
        .with_normalization();

    let signal_plan = signal_builder
        .build()
        .map_err(|e| std::io::Error::other(format!("Signal PlanBuilder::build failed: {e:?}")))?;

    let command_buffer = allocate_command_buffer(&context)?;

    submit_plan(&context, command_buffer, fence, || {
        kernel_plan
            .launch(command_buffer)
            .map_err(|e| std::io::Error::other(format!("Kernel forward launch failed: {e:?}")))
    })?;

    submit_plan(&context, command_buffer, fence, || {
        signal_plan
            .launch(command_buffer)
            .map_err(|e| std::io::Error::other(format!("Signal forward launch failed: {e:?}")))
    })?;

    let signal_freq_bytes = context.download_from_buffer(&signal_buffer, buffer_bytes as usize)?;
    let kernel_freq_bytes = context.download_from_buffer(&kernel_buffer, buffer_bytes as usize)?;

    let mut signal_freq = bytes_to_f32_vec(&signal_freq_bytes)?;
    let kernel_freq = bytes_to_f32_vec(&kernel_freq_bytes)?;
    complex_multiply_in_place(&mut signal_freq, &kernel_freq)?;

    context.upload_to_buffer(&signal_buffer, f32_as_bytes(&signal_freq))?;

    submit_plan(&context, command_buffer, fence, || {
        signal_plan
            .launch_inverse(command_buffer)
            .map_err(|e| std::io::Error::other(format!("Signal inverse launch failed: {e:?}")))
    })?;

    let output_bytes = context.download_from_buffer(&signal_buffer, buffer_bytes as usize)?;
    let output_complex = bytes_to_f32_vec(&output_bytes)?;
    let output = extract_real_samples(&output_complex, conv_len)?;

    validate_close("gpu_vs_cpu", &output, &expected, 1e-3)?;

    unsafe {
        context.device().device_wait_idle()?;
        context.device().free_command_buffers(
            context.compute_command_pool().clone(),
            std::slice::from_ref(&command_buffer),
        );
    }

    drop(signal_plan);
    drop(kernel_plan);
    drop(signal_builder);
    drop(kernel_builder);

    context.destroy_buffer(kernel_buffer);
    context.destroy_buffer(signal_buffer);
    context.destroy_fence(fence);

    println!("VkFFT array convolution validation passed.");
    println!("Expected: {:?}", expected);
    println!("Output:   {:?}", output);

    Ok(())
}

fn cpu_linear_convolution(signal: &[f32], ir: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0_f32; signal.len() + ir.len() - 1];

    for (n, out_sample) in out.iter_mut().enumerate() {
        let mut acc = 0.0_f32;
        for k in 0..signal.len() {
            if n >= k {
                let j = n - k;
                if j < ir.len() {
                    acc += signal[k] * ir[j];
                }
            }
        }
        *out_sample = acc;
    }

    out
}

fn validate_close(name: &str, actual: &[f32], expected: &[f32], tol: f32) -> Result<(), GpuError> {
    if actual.len() != expected.len() {
        return Err(std::io::Error::other(format!(
            "{name}: length mismatch, actual={} expected={}",
            actual.len(),
            expected.len()
        ))
        .into());
    }

    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        if (a - e).abs() > tol {
            return Err(std::io::Error::other(format!(
                "{name}: mismatch at index {i}, actual={a}, expected={e}, diff={}",
                (a - e).abs()
            ))
            .into());
        }
    }

    Ok(())
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

fn pack_real_as_complex(input: &[f32], fft_len: usize) -> Vec<f32> {
    let mut packed = vec![0.0_f32; fft_len * 2];
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

fn allocate_command_buffer(context: &GpuContext) -> Result<vk::CommandBuffer, GpuError> {
    let allocation_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(context.compute_command_pool().clone())
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let command_buffer = unsafe { context.device().allocate_command_buffers(&allocation_info)? }[0];
    Ok(command_buffer)
}

fn submit_plan<F>(
    context: &GpuContext,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    launch: F,
) -> Result<(), GpuError>
where
    F: FnOnce() -> Result<(), std::io::Error>,
{
    unsafe {
        context
            .device()
            .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;

        let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        context.device().begin_command_buffer(command_buffer, &begin_info)?;
    }

    launch()?;

    unsafe {
        context.device().end_command_buffer(command_buffer)?;
    }

    context.reset_fence(fence)?;

    let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));

    unsafe {
        context.device().queue_submit(
            context.compute_queue().clone(),
            std::slice::from_ref(&submit_info),
            fence,
        )?;
    }

    context.wait_for_fence(fence)?;

    Ok(())
}
