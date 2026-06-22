use sound_engine::acoustics::IrSample;
use sound_engine::auralization::Voice;
use sound_engine::auralization::convolver::PartitionedConvolver;
use sound_engine::core::config::{
    AcousticsConfig, AuralizationConfig, EngineConfig, IrCacheConfig, OutputChannels, SoundConfig,
};
use sound_engine::core::error::SoundResult;
use sound_engine::gpu::backend::VkBackend;
use std::sync::Arc;

const BLOCK_SIZE: usize = 1024;

fn main() -> SoundResult<()> {
    let backend = Arc::new(VkBackend::new()?);
    let ir: Vec<f32> = vec![1.0, 0.5, -0.25, 0.125];
    let convolver = PartitionedConvolver::new(
        backend,
        engine_config(44_100, BLOCK_SIZE, ir.len() as u32, OutputChannels::Mono),
    )?;

    // Small deterministic IR so expected output is easy to validate.
    let stereo_ir = mono_to_stereo_ir(&ir);
    let impulse_response = convolver.create_impulse_response(&stereo_ir)?;
    let mut voice = Voice::new(1, 0, 1.0, impulse_response, convolver.tail_size(), convolver.fft_size());

    let block_size = BLOCK_SIZE;
    let mut input_block = vec![0.0_f32; block_size];
    input_block[..8].copy_from_slice(&[1.0, -0.5, 0.25, 0.0, 0.5, 0.0, -0.25, 0.125]);

    let mut gpu_output = vec![0.0_f32; block_size];
    convolver.process_block(&mut voice, &input_block, &mut gpu_output)?;

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
    println!("If convolution is working, GPU output should differ from input and be close to expected (small error).");

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

fn mono_to_stereo_ir(ir: &[f32]) -> Vec<IrSample> {
    ir.iter().map(|&sample| IrSample::new(sample, sample)).collect()
}

fn engine_config(
    sample_rate: u32,
    block_size: usize,
    ir_num_samples: u32,
    output_channels: OutputChannels,
) -> EngineConfig {
    EngineConfig {
        sound: SoundConfig {
            output_channels,
            sample_rate,
            ir_num_samples,
        },
        acoustics: AcousticsConfig {
            listener_radius: 0.2,
            rays_per_query: 1,
            max_bounces: 0,
        },
        cache: IrCacheConfig::default(),
        auralization: AuralizationConfig { block_size },
    }
}
