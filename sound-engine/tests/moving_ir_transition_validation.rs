//! Verifies continuity when live spatial updates replace an IR.
//!
//! Two separately uploaded but numerically identical delayed IRs must render
//! exactly the same steady sine. The second update arrives before the first
//! transition finishes, matching camera poses that arrive faster than the
//! four-block transition window.

use sound_engine::acoustics::IrSample;
use sound_engine::auralization::{AuralizationEngine, PlaySoundRequest, SoundAsset};
use sound_engine::core::config::{AuralizationConfig, OutputChannels, SoundConfig};
use sound_engine::core::error::{SoundError, SoundResult};
use sound_engine::gpu::backend::VkBackend;
use std::f32::consts::TAU;
use std::sync::Arc;

const SAMPLE_RATE: u32 = 48_000;
const BLOCK_SIZE: usize = 1_024;
const IR_SAMPLES: usize = 8_192;
const DELAY_SAMPLES: usize = BLOCK_SIZE * 3;
const FREQUENCY_HZ: f32 = 750.0;
const WARM_BLOCKS: usize = 8;
const VALIDATION_BLOCKS: usize = 8;
const MAXIMUM_RELATIVE_L2: f64 = 1.0e-4;

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn overlapping_ir_updates_preserve_continuity() -> SoundResult<()> {
    let gpu = Arc::new(VkBackend::new()?);
    let mut engine = AuralizationEngine::new(
        gpu,
        SoundConfig {
            output_channels: OutputChannels::Mono,
            sample_rate: SAMPLE_RATE,
            ir_num_samples: IR_SAMPLES as u32,
        },
        AuralizationConfig {
            block_size: BLOCK_SIZE,
        },
    )?;
    let mut ir = vec![IrSample::zero(); IR_SAMPLES];
    ir[DELAY_SAMPLES] = IrSample {
        left: 1.0,
        right: 0.0,
    };
    let original = engine.prepare_impulse_response(&ir)?;
    let replacement = engine.prepare_impulse_response(&ir)?;
    let coalesced = engine.prepare_impulse_response(&ir)?;

    let samples = (0..BLOCK_SIZE * (WARM_BLOCKS + VALIDATION_BLOCKS + 2))
        .map(sine_sample)
        .collect();
    let sound_id = engine.insert_sound(SoundAsset::from_mono_samples(SAMPLE_RATE, samples)?)?;
    let voice_id = engine.play_sound(PlaySoundRequest::new(sound_id, original))?;
    let mut output = vec![0.0; BLOCK_SIZE];
    for _ in 0..WARM_BLOCKS {
        engine.render_block(&mut output)?;
    }

    engine.update_voice_impulse_response(voice_id, replacement)?;
    let mut maximum_relative_l2 = 0.0_f64;
    for block in 0..VALIDATION_BLOCKS {
        if block == 2 {
            engine.update_voice_impulse_response(voice_id, coalesced.clone())?;
        }
        engine.render_block(&mut output)?;
        let output_start = (WARM_BLOCKS + block) * BLOCK_SIZE;
        let expected = (0..BLOCK_SIZE)
            .map(|frame| sine_sample(output_start + frame - DELAY_SAMPLES))
            .collect::<Vec<_>>();
        let error = relative_l2(&output, &expected);
        assert!(
            error.is_finite(),
            "non-finite output after IR update at block {block}"
        );
        maximum_relative_l2 = maximum_relative_l2.max(error);
    }

    println!("moving IR identical-filter maximum relative L2: {maximum_relative_l2:.9}");
    if maximum_relative_l2 > MAXIMUM_RELATIVE_L2 {
        return Err(SoundError::invalid_state(format!(
            "identical IR updates changed a continuous signal: relative L2 {maximum_relative_l2:.9} exceeds {MAXIMUM_RELATIVE_L2}"
        )));
    }
    Ok(())
}

fn sine_sample(index: usize) -> f32 {
    (TAU * FREQUENCY_HZ * index as f32 / SAMPLE_RATE as f32).sin()
}

fn relative_l2(measured: &[f32], expected: &[f32]) -> f64 {
    let error = measured
        .iter()
        .zip(expected)
        .map(|(measured, expected)| (*measured as f64 - *expected as f64).powi(2))
        .sum::<f64>();
    let reference = expected
        .iter()
        .map(|sample| (*sample as f64).powi(2))
        .sum::<f64>();
    (error / reference.max(1.0e-24)).sqrt()
}
