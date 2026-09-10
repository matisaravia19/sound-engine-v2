//! Compares streaming direct FIR convolution with the production GPU FFT path.
//!
//! The sweep mirrors the experiment in Smith's *FFT Convolution* chapter while
//! retaining this engine's fixed 1,024-frame callback size. Each impulse-response
//! length runs in an isolated process so Vulkan/VkFFT allocator state cannot bias
//! later rows.

#[path = "support/timing.rs"]
mod timing;
use timing::{milliseconds, percentile};

use sound_engine::acoustics::IrSample;
use sound_engine::auralization::{AuralizationEngine, PlaySoundRequest, SoundAsset};
use sound_engine::core::config::{
    AcousticsConfig, AuralizationConfig, EngineConfig, IrCacheConfig, OutputChannels, SoundConfig,
};
use sound_engine::core::error::{SoundError, SoundResult};
use sound_engine::gpu::backend::VkBackend;
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

const SAMPLE_RATE: u32 = 48_000;
const BLOCK_SIZE: usize = 1_024;
const WARM_BLOCKS: usize = 5;
const MEASURED_BLOCKS: usize = 31;
const IR_LENGTHS: &[usize] = &[
    8, 16, 32, 64, 128, 256, 512, 1_024, 2_048, 4_096, 8_192, 16_384,
];

fn main() -> SoundResult<()> {
    if let Some(length) = child_length()? {
        println!("measurement,{}", run_length(length)?);
        return Ok(());
    }

    println!(
        "ir_samples,fft_samples,gpu_setup_ms,engine_setup_ms,ir_registration_ms,\
direct_p50_ms,direct_p95_ms,direct_p99_ms,fft_p50_ms,fft_p95_ms,fft_p99_ms,\
fft_over_direct_p50,relative_l2,max_absolute_error"
    );
    let executable = std::env::current_exe().map_err(|error| {
        SoundError::external(format!("failed to locate benchmark executable: {error}"))
    })?;
    for &length in IR_LENGTHS {
        let output = Command::new(&executable)
            .args(["--length", &length.to_string()])
            .output()
            .map_err(|error| {
                SoundError::external(format!("failed to launch length {length}: {error}"))
            })?;
        if !output.status.success() {
            return Err(SoundError::external(format!(
                "length {length} failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        let row = stdout
            .lines()
            .find_map(|line| line.strip_prefix("measurement,"))
            .ok_or_else(|| {
                SoundError::external(format!("length {length} returned no measurement"))
            })?;
        println!("{row}");
    }
    Ok(())
}

fn child_length() -> SoundResult<Option<usize>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        return Ok(None);
    }
    if args.len() != 2 || args[0] != "--length" {
        return Err(SoundError::invalid_argument("expected --length <samples>"));
    }
    let length = args[1]
        .parse::<usize>()
        .map_err(|error| SoundError::invalid_argument(format!("invalid IR length: {error}")))?;
    if length == 0 {
        return Err(SoundError::invalid_argument("IR length must be non-zero"));
    }
    Ok(Some(length))
}

fn run_length(ir_length: usize) -> SoundResult<String> {
    let gpu_start = Instant::now();
    let gpu = Arc::new(VkBackend::new()?);
    let gpu_setup = gpu_start.elapsed();

    let engine_start = Instant::now();
    let mut engine = AuralizationEngine::new(gpu, engine_config(ir_length))?;
    let engine_setup = engine_start.elapsed();

    let impulse = impulse_response(ir_length);
    let stereo_impulse = impulse
        .iter()
        .map(|&sample| IrSample {
            left: sample,
            right: sample,
        })
        .collect::<Vec<_>>();
    let registration_start = Instant::now();
    engine.register_impulse_response(1, &stereo_impulse)?;
    let ir_registration = registration_start.elapsed();

    let total_blocks = WARM_BLOCKS + MEASURED_BLOCKS;
    let input = deterministic_signal(total_blocks * BLOCK_SIZE);
    let sound_id =
        engine.insert_sound(SoundAsset::from_mono_samples(SAMPLE_RATE, input.clone())?)?;
    engine.play_sound(PlaySoundRequest::new(sound_id, 1))?;

    let mut direct = DirectFir::new(impulse);
    let mut direct_block = vec![0.0; BLOCK_SIZE];
    let mut gpu_block = vec![0.0; BLOCK_SIZE * 2];
    let mut direct_times = Vec::with_capacity(MEASURED_BLOCKS);
    let mut fft_times = Vec::with_capacity(MEASURED_BLOCKS);
    let mut squared_error = 0.0_f64;
    let mut squared_reference = 0.0_f64;
    let mut max_absolute_error = 0.0_f32;

    for block_index in 0..total_blocks {
        let start = block_index * BLOCK_SIZE;
        let input_block = &input[start..start + BLOCK_SIZE];

        let direct_start = Instant::now();
        direct.process_block(input_block, &mut direct_block);
        let direct_duration = direct_start.elapsed();

        let fft_start = Instant::now();
        engine.render_block(&mut gpu_block)?;
        let fft_duration = fft_start.elapsed();

        if block_index >= WARM_BLOCKS {
            direct_times.push(direct_duration);
            fft_times.push(fft_duration);
            for (&expected, frame) in direct_block.iter().zip(gpu_block.chunks_exact(2)) {
                let difference = frame[0] - expected;
                squared_error += f64::from(difference) * f64::from(difference);
                squared_reference += f64::from(expected) * f64::from(expected);
                max_absolute_error = max_absolute_error.max(difference.abs());
            }
        }
    }

    direct_times.sort_unstable();
    fft_times.sort_unstable();
    let direct_p50 = percentile(&direct_times, 0.50);
    let fft_p50 = percentile(&fft_times, 0.50);
    let relative_l2 = (squared_error / squared_reference.max(1.0e-30)).sqrt();
    let fft_size = (BLOCK_SIZE + ir_length - 1).next_power_of_two();
    Ok(format!(
        "{ir_length},{fft_size},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{relative_l2:.9},{max_absolute_error:.9}",
        milliseconds(gpu_setup),
        milliseconds(engine_setup),
        milliseconds(ir_registration),
        milliseconds(direct_p50),
        milliseconds(percentile(&direct_times, 0.95)),
        milliseconds(percentile(&direct_times, 0.99)),
        milliseconds(fft_p50),
        milliseconds(percentile(&fft_times, 0.95)),
        milliseconds(percentile(&fft_times, 0.99)),
        milliseconds(fft_p50) / milliseconds(direct_p50).max(1.0e-12),
    ))
}

struct DirectFir {
    impulse: Vec<f32>,
    delay_line: Vec<f32>,
    cursor: usize,
}

impl DirectFir {
    fn new(impulse: Vec<f32>) -> Self {
        let delay_line = vec![0.0; impulse.len()];
        Self {
            impulse,
            delay_line,
            cursor: 0,
        }
    }

    fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        for (&sample, destination) in input.iter().zip(output) {
            self.delay_line[self.cursor] = sample;
            let mut sum = 0.0_f32;
            for (tap, &coefficient) in self.impulse.iter().enumerate() {
                let index = if self.cursor >= tap {
                    self.cursor - tap
                } else {
                    self.delay_line.len() + self.cursor - tap
                };
                sum = coefficient.mul_add(self.delay_line[index], sum);
            }
            *destination = sum;
            self.cursor += 1;
            if self.cursor == self.delay_line.len() {
                self.cursor = 0;
            }
        }
    }
}

fn impulse_response(length: usize) -> Vec<f32> {
    (0..length)
        .map(|index| {
            let time = index as f32 / length.max(1) as f32;
            let carrier = (index as f32 * 0.754_877_7).sin();
            let sample = (-6.0 * time).exp() * carrier;
            if index == 0 {
                1.0
            } else {
                sample * 0.02
            }
        })
        .collect()
}

fn deterministic_signal(length: usize) -> Vec<f32> {
    let mut state = 0x1234_5678_u32;
    (0..length)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
        })
        .collect()
}

fn engine_config(ir_length: usize) -> EngineConfig {
    EngineConfig {
        sound: SoundConfig {
            output_channels: OutputChannels::Stereo,
            sample_rate: SAMPLE_RATE,
            ir_num_samples: ir_length as u32,
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
        auralization: AuralizationConfig {
            block_size: BLOCK_SIZE,
        },
    }
}
