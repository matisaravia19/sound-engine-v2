//! End-to-end latency benchmark for moving spatial voices.
//!
//! Unlike the isolated tracing and convolution benchmarks, this exercises the
//! public `SoundEngine` path used by a game: invalidate a voice, resolve its IR
//! cache entry, transform/register the IR, and render the next audio block.

use glam::{Vec3, vec3};
use sound_engine::auralization::SoundAsset;
use sound_engine::core::config::{
    AcousticsConfig, AuralizationConfig, EngineConfig, IrCacheConfig, OutputChannels, SoundConfig,
};
use sound_engine::core::error::SoundResult;
use sound_engine::scene::SceneDescription;
use sound_engine::{ListenerPose, PointSource, SoundEngine};
use std::f32::consts::TAU;
use std::time::Instant;

const SAMPLE_RATE: u32 = 48_000;
const BLOCK_SIZE: usize = 1_024;
const IR_SAMPLE_COUNT: u32 = 16_384;

fn main() -> SoundResult<()> {
    let engine_start = Instant::now();
    let mut engine = SoundEngine::new(engine_config())?;
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
    let mut output = vec![0.0; BLOCK_SIZE * OutputChannels::Stereo.count()];
    engine.render_block(&mut output)?;

    // This movement is within the default 0.2 m cache tolerance. The following
    // render therefore isolates cache-hit IR registration plus convolution.
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

    let deadline_ms = BLOCK_SIZE as f64 / SAMPLE_RATE as f64 * 1_000.0;
    println!(
        "engine_setup_ms,initial_play_ms,cache_hit_render_ms,cache_miss_render_ms,source_move_render_ms,audio_deadline_ms"
    );
    println!(
        "{engine_setup_ms:.6},{initial_play_ms:.6},{cache_hit_render_ms:.6},{cache_miss_render_ms:.6},\
{source_move_render_ms:.6},{deadline_ms:.6}"
    );

    // Backend teardown is a separately tracked lifecycle defect and currently
    // stalls on this machine. Keep it out of the spatial-update measurements.
    std::mem::forget(engine);
    Ok(())
}

fn listener(position: Vec3) -> ListenerPose {
    ListenerPose {
        position,
        right: Vec3::Y,
    }
}

fn engine_config() -> EngineConfig {
    EngineConfig {
        sound: SoundConfig {
            output_channels: OutputChannels::Stereo,
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
        auralization: AuralizationConfig { block_size: BLOCK_SIZE },
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
