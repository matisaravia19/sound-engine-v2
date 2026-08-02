//! Measures audio-block latency independently from acoustic path construction.
//!
//! The benchmark uses the production VkFFT convolution path, a 341 ms stereo
//! impulse response, and a continuous 1 kHz sine asset. Reported tail latency is
//! compared with the 1024-frame audio deadline at 48 kHz. The same source file
//! exists on both acoustic comparison branches so transport changes cannot alter
//! the workload.

use sound_engine::acoustics::IrSample;
use sound_engine::auralization::{AuralizationEngine, PlaySoundRequest, SoundAsset};
use sound_engine::core::config::{
    AcousticsConfig, AuralizationConfig, EngineConfig, IrCacheConfig, OutputChannels, SoundConfig,
};
use sound_engine::core::error::SoundResult;
use sound_engine::gpu::backend::VkBackend;
use std::f32::consts::TAU;
use std::sync::Arc;
use std::time::{Duration, Instant};

const SAMPLE_RATE: u32 = 48_000;
const BLOCK_SIZE: usize = 1_024;
const IR_SAMPLE_COUNT: usize = 16_384;
const WARM_BLOCKS: usize = 10;
const MEASURED_BLOCKS: usize = 100;
const VOICE_COUNTS: &[usize] = &[1, 8, 32, 64];

fn main() -> SoundResult<()> {
    let gpu_start = Instant::now();
    let gpu = Arc::new(VkBackend::new()?);
    let gpu_setup = gpu_start.elapsed();

    println!(
        "voices,gpu_setup_ms,engine_setup_ms,ir_registration_ms,p50_ms,p95_ms,p99_ms,max_ms,deadline_ms,p99_deadline_fraction"
    );
    for &voice_count in VOICE_COUNTS {
        benchmark_voice_count(gpu.clone(), gpu_setup, voice_count)?;
    }

    // Vulkan backend destruction is measured separately by the acoustic
    // lifecycle diagnostic. Avoid folding that known teardown stall into the
    // real-time audio-block measurements reported by this process.
    std::mem::forget(gpu);
    Ok(())
}

fn benchmark_voice_count(gpu: Arc<VkBackend>, gpu_setup: Duration, voice_count: usize) -> SoundResult<()> {
    let engine_start = Instant::now();
    let mut engine = AuralizationEngine::new(gpu, engine_config())?;
    let engine_setup = engine_start.elapsed();

    // Two fractional taps exercise stereo complex packing without making the
    // result dependent on a particular acoustic estimator.
    let mut ir = vec![IrSample::zero(); IR_SAMPLE_COUNT];
    ir[143] = IrSample {
        left: 0.75,
        right: 0.25,
    };
    ir[144] = IrSample {
        left: 0.25,
        right: 0.75,
    };
    let ir_start = Instant::now();
    engine.register_impulse_response(1, &ir)?;
    let ir_registration = ir_start.elapsed();

    let total_blocks = WARM_BLOCKS + MEASURED_BLOCKS + 2;
    let sample_count = total_blocks * BLOCK_SIZE;
    let sine = (0..sample_count)
        .map(|index| (TAU * 1_000.0 * index as f32 / SAMPLE_RATE as f32).sin())
        .collect();
    let sound_id = engine.insert_sound(SoundAsset::from_mono_samples(SAMPLE_RATE, sine)?)?;
    for _ in 0..voice_count {
        engine.play_sound(PlaySoundRequest::new(sound_id, 1))?;
    }

    let mut output = vec![0.0; BLOCK_SIZE * OutputChannels::Stereo.count()];
    for _ in 0..WARM_BLOCKS {
        engine.render_block(&mut output)?;
    }
    let mut block_times = Vec::with_capacity(MEASURED_BLOCKS);
    for _ in 0..MEASURED_BLOCKS {
        let start = Instant::now();
        engine.render_block(&mut output)?;
        block_times.push(start.elapsed());
    }
    block_times.sort_unstable();

    let deadline_ms = BLOCK_SIZE as f64 / SAMPLE_RATE as f64 * 1_000.0;
    let p50 = percentile(&block_times, 0.50);
    let p95 = percentile(&block_times, 0.95);
    let p99 = percentile(&block_times, 0.99);
    let max = *block_times.last().unwrap();
    println!(
        "{voice_count},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{deadline_ms:.6},{:.6}",
        milliseconds(gpu_setup),
        milliseconds(engine_setup),
        milliseconds(ir_registration),
        milliseconds(p50),
        milliseconds(p95),
        milliseconds(p99),
        milliseconds(max),
        milliseconds(p99) / deadline_ms,
    );
    Ok(())
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

fn percentile(sorted: &[Duration], fraction: f64) -> Duration {
    let index = ((sorted.len() - 1) as f64 * fraction).ceil() as usize;
    sorted[index]
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}
