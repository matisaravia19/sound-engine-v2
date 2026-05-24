use sound_engine::auralization::convolver::PartitionedConvolver;
use sound_engine::auralization::{BLOCK_SIZE, ImpulseResponseId};
use sound_engine::gpu::backend::VkBackend;
use std::sync::Arc;

const IR_ID: ImpulseResponseId = 7;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let backend = Arc::new(VkBackend::new().map_err(|e| std::io::Error::other(e.to_string()))?);
    let mut convolver = PartitionedConvolver::new(backend)?;

    // Small deterministic IR so expected output is easy to validate.
    let ir: Vec<f32> = vec![1.0, 0.5, -0.25, 0.125];
    convolver.register_impulse_response(IR_ID, &ir)?;
    let voice_id = convolver.start_sound(IR_ID)?;

    let block_size = BLOCK_SIZE as usize;
    let mut input_block = vec![0.0_f32; block_size];
    input_block[..8].copy_from_slice(&[1.0, -0.5, 0.25, 0.0, 0.5, 0.0, -0.25, 0.125]);

    let mut gpu_output = vec![0.0_f32; block_size];
    convolver.process_block(voice_id, &input_block, &mut gpu_output)?;

    let expected = cpu_linear_convolution(&input_block, &ir);
    let compare_len = 32usize;
    let mut max_abs_err = 0.0_f32;

    println!("index\tinput\texpected\tgpu\tabs_err");
    for i in 0..compare_len {
        let abs_err = (gpu_output[i] - expected[i]).abs();
        max_abs_err = max_abs_err.max(abs_err);
        println!(
            "{i}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            input_block[i], expected[i], gpu_output[i], abs_err
        );
    }

    println!("\nmax_abs_err(first {compare_len} samples) = {max_abs_err:.6}");
    println!(
        "If convolution is working, GPU output should differ from input and be close to expected (small error)."
    );

    Ok(())
}

fn cpu_linear_convolution(signal: &[f32], ir: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0_f32; signal.len() + ir.len() - 1];
    for (n, out_sample) in out.iter_mut().enumerate() {
        let mut acc = 0.0_f32;
        for (k, &x) in signal.iter().enumerate() {
            if n >= k {
                let j = n - k;
                if j < ir.len() {
                    acc += x * ir[j];
                }
            }
        }
        *out_sample = acc;
    }
    out
}
