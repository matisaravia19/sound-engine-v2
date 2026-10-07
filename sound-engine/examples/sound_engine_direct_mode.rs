//! Render a generated tone through a room and save the mixed output as a WAV.

#[path = "support/demo.rs"]
mod demo;

use demo::{box_room_scene, engine_config, sine_tone};
use glam::{Vec3, vec3};
use hound::{SampleFormat, WavSpec, WavWriter};
use sound_engine::core::error::SoundResult;
use sound_engine::{ListenerPose, PlaySpatialSoundRequest, PointSource, SoundEngine};
use std::fs;
use std::path::Path;

const OUTPUT_PATH: &str = "target/sound_engine_direct_mode.wav";

fn main() -> SoundResult<()> {
    let mut engine = SoundEngine::new(engine_config())?;
    engine.load_scene(box_room_scene())?;
    engine.set_listener_pose(ListenerPose {
        position: vec3(2.0, 0.0, -1.0),
        forward: -Vec3::X,
        up: Vec3::Z,
    })?;

    let sound_id = engine.insert_sound(sine_tone(440.0, 0.75)?)?;
    let voice_id = engine.play_sound(PlaySpatialSoundRequest {
        sound_id,
        source: PointSource {
            position: vec3(-3.0, 0.5, 2.0),
            amplitude: 1_000.0,
        },
        volume: 0.8,
    })?;
    println!("started direct-mode voice {voice_id}");

    let mut rendered = Vec::new();
    let mut block = vec![0.0; engine.block_size() * engine.output_channels()];
    while engine.has_active_voices() {
        engine.render_block(&mut block)?;
        rendered.extend_from_slice(&block);
    }

    write_wav(
        OUTPUT_PATH,
        engine.sample_rate(),
        engine.output_channels() as u16,
        &rendered,
    )?;
    println!("wrote {OUTPUT_PATH}");
    Ok(())
}

fn write_wav(
    path: impl AsRef<Path>,
    sample_rate: u32,
    channels: u16,
    samples: &[f32],
) -> SoundResult<()> {
    if let Some(parent) = path.as_ref().parent() {
        fs::create_dir_all(parent).map_err(|source| {
            sound_engine::core::error::SoundError::with_source(
                sound_engine::core::error::ErrorCode::Io,
                format!("Failed to create {}", parent.display()),
                source,
            )
        })?;
    }

    let spec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path.as_ref(), spec).map_err(|source| {
        sound_engine::core::error::SoundError::with_source(
            sound_engine::core::error::ErrorCode::Io,
            format!("Failed to create {}", path.as_ref().display()),
            source,
        )
    })?;
    for sample in samples {
        let sample = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer.write_sample(sample).map_err(|source| {
            sound_engine::core::error::SoundError::with_source(
                sound_engine::core::error::ErrorCode::Io,
                "Failed to write WAV sample",
                source,
            )
        })?;
    }
    writer.finalize().map_err(|source| {
        sound_engine::core::error::SoundError::with_source(
            sound_engine::core::error::ErrorCode::Io,
            format!("Failed to finalize {}", path.as_ref().display()),
            source,
        )
    })?;
    Ok(())
}
