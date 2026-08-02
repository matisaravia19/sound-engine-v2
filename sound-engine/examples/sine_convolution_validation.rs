//! Verifies the production GPU convolver with analytically predictable sines.
//!
//! Two equal pressure arrivals are separated by 24 samples, exactly half a
//! period at 1 kHz/48 kHz. Their steady-state response is
//! `2*cos(omega*12)` in magnitude and cancels at 1 kHz.

use sound_engine::acoustics::IrSample;
use sound_engine::auralization::{AuralizationEngine, PlaySoundRequest, SoundAsset};
use sound_engine::core::config::{
    AcousticsConfig, AuralizationConfig, EngineConfig, IrCacheConfig, OutputChannels, SoundConfig,
};
use sound_engine::core::error::SoundResult;
use sound_engine::gpu::backend::VkBackend;
use std::f32::consts::TAU;
use std::sync::Arc;

const SAMPLE_RATE: u32 = 48_000;
const BLOCK_SIZE: usize = 1_024;
const IR_SAMPLE_COUNT: usize = 16_384;
const BLOCK_COUNT: usize = 8;
const FIRST_DELAY: usize = 128;
const SECOND_DELAY: usize = 152;
const FREQUENCIES_HZ: &[f32] = &[125.0, 250.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0];

fn main() -> SoundResult<()> {
    let gpu = Arc::new(VkBackend::new()?);
    let mut engine = AuralizationEngine::new(gpu.clone(), engine_config())?;
    let mut ir = vec![IrSample::zero(); IR_SAMPLE_COUNT];
    ir[FIRST_DELAY] = IrSample { left: 1.0, right: 1.0 };
    ir[SECOND_DELAY] = IrSample { left: 1.0, right: 1.0 };
    engine.register_impulse_response(1, &ir)?;

    println!("frequency_hz,expected_gain,measured_gain,relative_l2,steady_rms");
    for &frequency in FREQUENCIES_HZ {
        validate_frequency(&mut engine, frequency)?;
    }

    // Backend teardown is excluded for the same lifecycle reason as the timing
    // benchmarks; all numerical results have already been downloaded above.
    std::mem::forget(engine);
    std::mem::forget(gpu);
    Ok(())
}

fn validate_frequency(engine: &mut AuralizationEngine, frequency: f32) -> SoundResult<()> {
    let sample_count = BLOCK_COUNT * BLOCK_SIZE;
    let input = (0..sample_count)
        .map(|index| (TAU * frequency * index as f32 / SAMPLE_RATE as f32).sin())
        .collect::<Vec<_>>();
    let sound_id = engine.insert_sound(SoundAsset::from_mono_samples(SAMPLE_RATE, input.clone())?)?;
    let voice_id = engine.play_sound(PlaySoundRequest::new(sound_id, 1))?;
    let mut measured = Vec::with_capacity(sample_count);
    let mut output = vec![0.0; BLOCK_SIZE * 2];
    for _ in 0..BLOCK_COUNT {
        engine.render_block(&mut output)?;
        measured.extend(output.chunks_exact(2).map(|frame| frame[0]));
    }
    engine.stop_voice(voice_id);

    let exact = (0..sample_count)
        .map(|index| delayed(&input, index, FIRST_DELAY) + delayed(&input, index, SECOND_DELAY))
        .collect::<Vec<_>>();
    // Skip the first two blocks so the RMS gain represents the steady sine, not
    // the deterministic arrival transient at the beginning of the asset.
    let steady_start = 2 * BLOCK_SIZE;
    let input_rms = rms(&input[steady_start..]);
    let measured_rms = rms(&measured[steady_start..]);
    let expected_gain = 2.0
        * (std::f32::consts::PI * frequency * (SECOND_DELAY - FIRST_DELAY) as f32 / SAMPLE_RATE as f32)
            .cos()
            .abs();
    let relative_l2 = relative_l2(&measured, &exact);
    println!(
        "{frequency:.3},{expected_gain:.9},{:.9},{relative_l2:.9},{measured_rms:.9}",
        measured_rms / input_rms.max(1.0e-12),
    );
    Ok(())
}

fn delayed(input: &[f32], index: usize, delay: usize) -> f32 {
    index
        .checked_sub(delay)
        .and_then(|source| input.get(source))
        .copied()
        .unwrap_or(0.0)
}

fn rms(samples: &[f32]) -> f32 {
    (samples.iter().map(|&sample| sample * sample).sum::<f32>() / samples.len() as f32).sqrt()
}

fn relative_l2(measured: &[f32], exact: &[f32]) -> f32 {
    let error = measured
        .iter()
        .zip(exact)
        .map(|(&measured, &exact)| {
            let difference = measured - exact;
            difference * difference
        })
        .sum::<f32>()
        .sqrt();
    let reference = exact.iter().map(|sample| sample * sample).sum::<f32>().sqrt();
    error / reference.max(1.0e-12)
}

fn engine_config() -> EngineConfig {
    EngineConfig {
        sound: SoundConfig {
            output_channels: OutputChannels::Stereo,
            sample_rate: SAMPLE_RATE,
            ir_num_samples: IR_SAMPLE_COUNT as u32,
        },
        acoustics: AcousticsConfig {
            listener_ray_count: 1,
            max_bounces: 0,
            maximum_path_length: 343.0,
            minimum_throughput: 0.0,
            sampling_seed: 0,
        },
        cache: IrCacheConfig::default(),
        binaural: Default::default(),
        auralization: AuralizationConfig { block_size: BLOCK_SIZE },
    }
}
