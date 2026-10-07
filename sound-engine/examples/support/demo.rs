//! Self-contained acoustic scene and tone shared by the engine usage examples.
use glam::{Mat4, Vec3, vec3};
use sound_engine::auralization::SoundAsset;
use sound_engine::core::config::{
    AcousticsConfig, AuralizationConfig, EngineConfig, IrCacheConfig, OutputChannels, SoundConfig,
};
use sound_engine::core::error::SoundResult;
use sound_engine::scene::{Material, MeshAsset, SceneDescription, SceneObject};

const SAMPLE_RATE: u32 = 48_000;
const BLOCK_SIZE: usize = 1024;
const IR_SAMPLES: u32 = SAMPLE_RATE * 2;

/// Audio and ray budgets shared by the engine usage examples.
pub fn engine_config() -> EngineConfig {
    EngineConfig {
        sound: SoundConfig {
            output_channels: OutputChannels::Stereo,
            sample_rate: SAMPLE_RATE,
            ir_num_samples: IR_SAMPLES,
        },
        acoustics: AcousticsConfig {
            listener_ray_count: 262_144,
            max_bounces: 8,
            maximum_path_length: 343.0,
            minimum_throughput: 1.0e-6,
            sampling_seed: 0,
        },
        cache: IrCacheConfig::default(),
        binaural: Default::default(),
        auralization: AuralizationConfig {
            block_size: BLOCK_SIZE,
        },
    }
}

/// A closed room with one broadband material.
pub fn box_room_scene() -> SceneDescription {
    SceneDescription {
        meshes: vec![MeshAsset {
            id: 1,
            vertices: box_room_vertices(vec3(10.0, 3.0, 14.0)),
            indices: Vec::new(),
            opaque: true,
        }],
        materials: vec![Material {
            id: 1,
            absorption: 0.08,
            scattering: 0.15,
            roughness: 0.15,
        }],
        objects: vec![SceneObject {
            id: 1,
            mesh_id: 1,
            material_id: 1,
            transform: Mat4::IDENTITY,
            active: true,
        }],
        diffraction_edges: Vec::new(),
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
        vertices.extend_from_slice(&[
            corners[a], corners[b], corners[c], corners[a], corners[c], corners[d],
        ]);
    }
    vertices
}

/// A short, faded mono tone that requires no external audio asset.
pub fn sine_tone(frequency: f32, seconds: f32) -> SoundResult<SoundAsset> {
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
