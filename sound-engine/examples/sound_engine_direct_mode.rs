use glam::{Mat4, Vec3, vec3};
use hound::{SampleFormat, WavSpec, WavWriter};
use sound_engine::auralization::SoundAsset;
use sound_engine::core::config::{
    AcousticsConfig, AuralizationConfig, EngineConfig, IrCacheConfig, OutputChannels, SoundConfig,
};
use sound_engine::core::error::SoundResult;
use sound_engine::scene::{Material, MeshAsset, SceneDescription, SceneObject};
use sound_engine::{ListenerPose, PlaySpatialSoundRequest, PointSource, SoundEngine};
use std::fs;
use std::path::Path;

const SAMPLE_RATE: u32 = 48_000;
const BLOCK_SIZE: usize = 1024;
const IR_SAMPLES: u32 = SAMPLE_RATE * 2;
const OUTPUT_PATH: &str = "target/sound_engine_direct_mode.wav";

fn main() -> SoundResult<()> {
    let mut engine = SoundEngine::new(engine_config())?;
    engine.load_scene(box_room_scene())?;
    engine.set_listener_pose(ListenerPose {
        position: vec3(2.0, 0.0, -1.0),
        right: Vec3::Y,
    })?;

    let sound_id = engine.insert_sound(sine_tone(440.0, 0.75)?)?;
    let voice_id = engine.play_sound(PlaySpatialSoundRequest {
        sound_id,
        source: PointSource {
            position: vec3(-3.0, 0.5, 2.0),
            energy: 1_000.0,
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

    write_wav(OUTPUT_PATH, engine.sample_rate(), engine.output_channels() as u16, &rendered)?;
    println!("wrote {OUTPUT_PATH}");
    Ok(())
}

fn engine_config() -> EngineConfig {
    EngineConfig {
        sound: SoundConfig {
            output_channels: OutputChannels::Stereo,
            sample_rate: SAMPLE_RATE,
            ir_num_samples: IR_SAMPLES,
        },
        acoustics: AcousticsConfig {
            listener_half_extent: Vec3::splat(0.5),
            rays_per_query: 262_144,
            max_bounces: 8,
        },
        cache: IrCacheConfig::default(),
        auralization: AuralizationConfig { block_size: BLOCK_SIZE },
    }
}

fn box_room_scene() -> SceneDescription {
    SceneDescription {
        meshes: vec![MeshAsset {
            id: 1,
            vertices: box_room_vertices(vec3(10.0, 3.0, 14.0)),
            indices: Vec::new(),
            opaque: true,
        }],
        materials: vec![Material {
            id: 1,
            absorption_bands: vec![0.08],
            scattering: 0.15,
            transmission: None,
        }],
        objects: vec![SceneObject {
            id: 1,
            mesh_id: 1,
            material_id: 1,
            transform: Mat4::IDENTITY,
            active: true,
        }],
    }
}

fn box_room_vertices(half_extent: Vec3) -> Vec<Vec3> {
    let min = -half_extent;
    let max = half_extent;
    let corners = [
        vec3(min.x, min.y, min.z),
        vec3(max.x, min.y, min.z),
        vec3(max.x, max.y, min.z),
        vec3(min.x, max.y, min.z),
        vec3(min.x, min.y, max.z),
        vec3(max.x, min.y, max.z),
        vec3(max.x, max.y, max.z),
        vec3(min.x, max.y, max.z),
    ];
    let faces = [
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [3, 7, 6, 2],
        [0, 4, 7, 3],
        [1, 2, 6, 5],
    ];

    let mut vertices = Vec::with_capacity(faces.len() * 6);
    for [a, b, c, d] in faces {
        vertices.extend_from_slice(&[corners[a], corners[b], corners[c], corners[a], corners[c], corners[d]]);
    }
    vertices
}

fn sine_tone(frequency: f32, seconds: f32) -> SoundResult<SoundAsset> {
    let frames = (SAMPLE_RATE as f32 * seconds) as usize;
    let samples = (0..frames)
        .map(|frame| {
            let t = frame as f32 / SAMPLE_RATE as f32;
            let envelope = fade_envelope(frame, frames);
            (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.35 * envelope
        })
        .collect();
    SoundAsset::from_mono_samples(SAMPLE_RATE, samples)
}

fn fade_envelope(frame: usize, frames: usize) -> f32 {
    let fade_frames = (SAMPLE_RATE / 100) as usize;
    let fade_in = (frame as f32 / fade_frames as f32).min(1.0);
    let fade_out = ((frames.saturating_sub(frame)) as f32 / fade_frames as f32).min(1.0);
    fade_in.min(fade_out)
}

fn write_wav(path: impl AsRef<Path>, sample_rate: u32, channels: u16, samples: &[f32]) -> SoundResult<()> {
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
