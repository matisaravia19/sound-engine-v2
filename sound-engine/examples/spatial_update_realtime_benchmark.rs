//! End-to-end latency benchmark for moving spatial voices.
//!
//! Unlike the isolated tracing and convolution benchmarks, this exercises the
//! public `SoundEngine` path used by a game: invalidate a voice, resolve its IR
//! cache entry, transform/register the IR, and render the next audio block.

#[path = "support/timing.rs"]
mod timing;
use timing::percentile;

use glam::{Vec3, vec3};
use sound_engine::auralization::SoundAsset;
use sound_engine::core::config::{
    AcousticsConfig, AuralizationConfig, EngineConfig, IrCacheConfig, OutputChannels, SoundConfig,
};
use sound_engine::core::error::{SoundError, SoundResult};
use sound_engine::scene::SceneDescription;
use sound_engine::{ListenerPose, PointSource, SoundEngine};
use std::f32::consts::TAU;
use std::process::Command;
use std::time::{Duration, Instant};

const SAMPLE_RATE: u32 = 48_000;
const BLOCK_SIZE: usize = 1_024;
const IR_SAMPLE_COUNT: u32 = 16_384;
const WARM_REPEATS: usize = 101;
const IR_UPDATE_BUDGET_MS: f64 = 4.0;

struct Measurements {
    engine_setup_ms: f64,
    initial_play_ms: f64,
    cache_hit_render_ms: f64,
    cache_miss_render_ms: f64,
    source_move_render_ms: f64,
    warm_p50_ms: f64,
    warm_p95_ms: f64,
    warm_p99_ms: f64,
}

fn main() -> SoundResult<()> {
    if let Some(mode) = std::env::args().nth(1) {
        let output_channels = match mode.as_str() {
            "--stereo" => OutputChannels::Stereo,
            "--binaural" => OutputChannels::Binaural,
            _ => {
                return Err(SoundError::invalid_argument(format!(
                    "unknown benchmark mode {mode}"
                )));
            }
        };
        print_child_measurements(&run_mode(output_channels)?);
        return Ok(());
    }

    // Run each mode in a fresh process so device setup and allocator history do
    // not measurably penalize whichever channel layout is constructed second.
    let stereo = run_isolated_mode("--stereo")?;
    let binaural = run_isolated_mode("--binaural")?;
    let deadline_ms = BLOCK_SIZE as f64 / SAMPLE_RATE as f64 * 1_000.0;

    println!(
        "mode,engine_setup_ms,initial_play_ms,cache_hit_render_ms,cache_miss_render_ms,\
source_move_render_ms,warm_p50_ms,warm_p95_ms,warm_p99_ms,audio_deadline_ms"
    );
    print_measurements("stereo", &stereo, deadline_ms);
    print_measurements("binaural", &binaural, deadline_ms);

    require(
        binaural.warm_p95_ms <= IR_UPDATE_BUDGET_MS,
        format!(
            "binaural warm p95 {:.3} ms exceeds the {:.3} ms IR-update budget",
            binaural.warm_p95_ms, IR_UPDATE_BUDGET_MS
        ),
    )?;
    require(
        binaural.warm_p95_ms <= stereo.warm_p95_ms * 1.10,
        format!(
            "binaural warm p95 {:.3} ms exceeds stereo {:.3} ms by more than 10%",
            binaural.warm_p95_ms, stereo.warm_p95_ms
        ),
    )
}

fn run_isolated_mode(mode: &str) -> SoundResult<Measurements> {
    let executable = std::env::current_exe().map_err(|error| {
        SoundError::external(format!("failed to locate benchmark executable: {error}"))
    })?;
    let output = Command::new(executable)
        .arg(mode)
        .output()
        .map_err(|error| {
            SoundError::external(format!("failed to launch {mode} benchmark: {error}"))
        })?;
    if !output.status.success() {
        return Err(SoundError::external(format!(
            "{mode} benchmark failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout
        .lines()
        .find_map(|line| line.strip_prefix("measurements,"))
        .ok_or_else(|| {
            SoundError::external(format!("{mode} benchmark returned no measurements"))
        })?;
    parse_measurements(line)
}

fn print_child_measurements(measurements: &Measurements) {
    println!(
        "measurements,{},{},{},{},{},{},{},{}",
        measurements.engine_setup_ms,
        measurements.initial_play_ms,
        measurements.cache_hit_render_ms,
        measurements.cache_miss_render_ms,
        measurements.source_move_render_ms,
        measurements.warm_p50_ms,
        measurements.warm_p95_ms,
        measurements.warm_p99_ms,
    );
}

fn parse_measurements(line: &str) -> SoundResult<Measurements> {
    let values = line
        .split(',')
        .map(|value| {
            value.parse::<f64>().map_err(|error| {
                SoundError::external(format!("invalid benchmark measurement {value}: {error}"))
            })
        })
        .collect::<SoundResult<Vec<_>>>()?;
    if values.len() != 8 {
        return Err(SoundError::external(format!(
            "expected 8 benchmark measurements, received {}",
            values.len()
        )));
    }
    Ok(Measurements {
        engine_setup_ms: values[0],
        initial_play_ms: values[1],
        cache_hit_render_ms: values[2],
        cache_miss_render_ms: values[3],
        source_move_render_ms: values[4],
        warm_p50_ms: values[5],
        warm_p95_ms: values[6],
        warm_p99_ms: values[7],
    })
}

fn run_mode(output_channels: OutputChannels) -> SoundResult<Measurements> {
    let engine_start = Instant::now();
    let mut engine = SoundEngine::new(engine_config(output_channels))?;
    let engine_setup_ms = milliseconds(engine_start);
    engine.load_scene(empty_scene())?;
    engine.set_listener_pose(listener(vec3(1.0, 0.0, 0.0)))?;

    let samples = (0..BLOCK_SIZE * 64)
        .map(|index| (TAU * 1_000.0 * index as f32 / SAMPLE_RATE as f32).sin())
        .collect();
    let sound_id = engine.insert_sound(SoundAsset::from_mono_samples(SAMPLE_RATE, samples)?)?;
    let source_id = engine.create_sound_source(PointSource::new(Vec3::ZERO))?;

    let play_start = Instant::now();
    engine.play_sound_from_source(sound_id, source_id, 1.0)?;
    let initial_play_ms = milliseconds(play_start);
    let mut output = vec![0.0; BLOCK_SIZE * output_channels.count()];
    engine.render_block(&mut output)?;

    // This movement is within the default 0.2 m cache tolerance. The following
    // render therefore isolates cache lookup plus convolution; no redundant IR
    // transform or registration should occur for the reused GPU buffer.
    engine.set_listener_pose(listener(vec3(1.1, 0.0, 0.0)))?;
    let cache_hit_start = Instant::now();
    engine.render_block(&mut output)?;
    let cache_hit_render_ms = milliseconds(cache_hit_start);

    // This movement crosses both the position cell and reuse-distance limits,
    // forcing a new trace before the IR can be transformed and registered.
    engine.set_listener_pose(listener(vec3(1.6, 0.0, 0.0)))?;
    let cache_miss_start = Instant::now();
    engine.render_block(&mut output)?;
    let cache_miss_render_ms = milliseconds(cache_miss_start);

    engine.update_sound_source(source_id, PointSource::new(vec3(0.5, 0.0, 0.0)))?;
    let source_move_start = Instant::now();
    engine.render_block(&mut output)?;
    let source_move_render_ms = milliseconds(source_move_start);

    // Every warm position moves beyond the cache reuse radius, so each render
    // includes acoustic construction, IR registration, and the next audio block.
    let mut warm_updates = Vec::with_capacity(WARM_REPEATS);
    for index in 0..WARM_REPEATS {
        let distance = 2.0 + index as f32 * 0.25;
        engine.set_listener_pose(listener(vec3(distance, 0.0, 0.0)))?;
        let start = Instant::now();
        engine.render_block(&mut output)?;
        warm_updates.push(start.elapsed());
    }
    warm_updates.sort_unstable();

    let measurements = Measurements {
        engine_setup_ms,
        initial_play_ms,
        cache_hit_render_ms,
        cache_miss_render_ms,
        source_move_render_ms,
        warm_p50_ms: duration_ms(percentile(&warm_updates, 0.50)),
        warm_p95_ms: duration_ms(percentile(&warm_updates, 0.95)),
        warm_p99_ms: duration_ms(percentile(&warm_updates, 0.99)),
    };

    Ok(measurements)
}

fn print_measurements(mode: &str, measurements: &Measurements, deadline_ms: f64) {
    println!(
        "{mode},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{deadline_ms:.6}",
        measurements.engine_setup_ms,
        measurements.initial_play_ms,
        measurements.cache_hit_render_ms,
        measurements.cache_miss_render_ms,
        measurements.source_move_render_ms,
        measurements.warm_p50_ms,
        measurements.warm_p95_ms,
        measurements.warm_p99_ms,
    );
}

fn listener(position: Vec3) -> ListenerPose {
    ListenerPose {
        position,
        forward: -Vec3::X,
        up: Vec3::Z,
    }
}

fn engine_config(output_channels: OutputChannels) -> EngineConfig {
    EngineConfig {
        sound: SoundConfig {
            output_channels,
            sample_rate: SAMPLE_RATE,
            ir_num_samples: IR_SAMPLE_COUNT,
        },
        acoustics: AcousticsConfig {
            listener_ray_count: 65_536,
            max_bounces: 0,
            maximum_path_length: 343.0,
            minimum_throughput: 0.0,
            sampling_seed: 7,
        },
        cache: IrCacheConfig::default(),
        binaural: Default::default(),
        auralization: AuralizationConfig {
            block_size: BLOCK_SIZE,
        },
    }
}

fn empty_scene() -> SceneDescription {
    SceneDescription {
        meshes: Vec::new(),
        materials: Vec::new(),
        objects: Vec::new(),
        diffraction_edges: Vec::new(),
    }
}

fn milliseconds(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1_000.0
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn require(condition: bool, message: String) -> SoundResult<()> {
    if condition {
        Ok(())
    } else {
        Err(SoundError::external(message))
    }
}
