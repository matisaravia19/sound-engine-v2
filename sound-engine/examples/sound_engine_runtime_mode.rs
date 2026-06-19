use glam::{Mat4, Vec3, vec3};
use sound_engine::auralization::SoundAsset;
use sound_engine::core::config::{
    AcousticsConfig, AuralizationConfig, EngineConfig, IrCacheConfig, OutputChannels, SoundConfig,
};
use sound_engine::core::error::SoundResult;
use sound_engine::playback::PlaybackConfig;
use sound_engine::scene::{Material, MeshAsset, SceneDescription, SceneObject};
use sound_engine::{ListenerPose, PlaySpatialSoundRequest, PointSource, SoundEngine};
use std::thread;
use std::time::Duration;

const SAMPLE_RATE: u32 = 48_000;
const BLOCK_SIZE: usize = 1024;
const IR_SAMPLES: u32 = SAMPLE_RATE * 2;

fn main() -> SoundResult<()> {
    let mut engine = SoundEngine::new(engine_config())?;
    engine.load_scene(box_room_scene())?;
    engine.set_listener_pose(ListenerPose {
        position: vec3(2.0, 0.0, -1.0),
        right: Vec3::Y,
    })?;

    let sound_a = engine.insert_sound(sine_tone(440.0, 0.75)?)?;
    let sound_b = engine.insert_sound(sine_tone(660.0, 0.75)?)?;
    let knocks = engine.insert_sound(tac_tac_tac()?)?;
    let music = engine.load_wav_sound("C:/Users/matis/OneDrive/Documentos/Fing/Tesis/explosion.wav")?;
    engine.start_sound_player(PlaybackConfig::default())?;

    let voice_a = engine.play_sound(PlaySpatialSoundRequest {
        sound_id: sound_a,
        source: PointSource {
            position: vec3(-3.0, 0.5, 2.0),
            energy: 1_000.0,
        },
        volume: 0.8,
    })?;
    println!("started runtime voice {voice_a}");

    thread::sleep(Duration::from_millis(450));

    let voice_b = engine.play_sound(PlaySpatialSoundRequest {
        sound_id: sound_b,
        source: PointSource {
            position: vec3(-2.5, -0.75, -3.0),
            energy: 1_000.0,
        },
        volume: 0.6,
    })?;
    println!("started runtime voice {voice_b}");

    thread::sleep(Duration::from_millis(1000));

    let knock_voice = engine.play_sound(PlaySpatialSoundRequest {
        sound_id: knocks,
        source: PointSource {
            position: vec3(3.5, 0.25, 4.0),
            energy: 2_500.0,
        },
        volume: 1.0,
    })?;
    println!("started runtime knock voice {knock_voice}");

    thread::sleep(Duration::from_secs(2));

    let music_voice = engine.play_sound(PlaySpatialSoundRequest {
        sound_id: music,
        source: PointSource {
            position: vec3(3.5, 0.0, 0.0),
            energy: 10_000.0,
        },
        volume: 1.0,
    })?;
    println!("started runtime music voice {music_voice}");

    thread::sleep(Duration::from_secs(180));

    engine.stop_sound_player()?;
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

fn tac_tac_tac() -> SoundResult<SoundAsset> {
    let seconds = 0.9;
    let frames = (SAMPLE_RATE as f32 * seconds) as usize;
    let mut samples = vec![0.0; frames];
    for start_seconds in [0.0, 0.18, 0.36] {
        add_knock(&mut samples, start_seconds);
    }
    SoundAsset::from_mono_samples(SAMPLE_RATE, samples)
}

fn add_knock(samples: &mut [f32], start_seconds: f32) {
    let start = (start_seconds * SAMPLE_RATE as f32) as usize;
    let length = (0.035 * SAMPLE_RATE as f32) as usize;
    for frame in 0..length {
        let index = start + frame;
        if index >= samples.len() {
            break;
        }

        let t = frame as f32 / SAMPLE_RATE as f32;
        let envelope = (-95.0 * t).exp();
        let body = (2.0 * std::f32::consts::PI * 185.0 * t).sin();
        let click = (2.0 * std::f32::consts::PI * 2_400.0 * t).sin();
        samples[index] += envelope * (0.75 * body + 0.25 * click);
    }
}

fn fade_envelope(frame: usize, frames: usize) -> f32 {
    let fade_frames = (SAMPLE_RATE / 100) as usize;
    let fade_in = (frame as f32 / fade_frames as f32).min(1.0);
    let fade_out = ((frames.saturating_sub(frame)) as f32 / fade_frames as f32).min(1.0);
    fade_in.min(fade_out)
}
