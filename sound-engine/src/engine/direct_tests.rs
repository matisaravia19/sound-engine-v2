//! Hardware regression for coherent spatial updates across rendered blocks.

use super::*;
use crate::acoustics::IrSample;
use crate::core::config::*;
use crate::scene::{DiffractionEdge, Material, MeshAsset, SceneObject};
use glam::{Mat4, Vec3, vec3};

fn box_scene() -> SceneDescription {
    let mut vertices = Vec::new();
    for corners in [
        [
            vec3(-0.36, -10.0, -10.0),
            vec3(-0.36, 1.0, -10.0),
            vec3(-0.36, 1.0, 10.0),
            vec3(-0.36, -10.0, 10.0),
        ],
        [
            vec3(0.36, -10.0, -10.0),
            vec3(0.36, 1.0, -10.0),
            vec3(0.36, 1.0, 10.0),
            vec3(0.36, -10.0, 10.0),
        ],
        [
            vec3(-0.36, 1.0, -10.0),
            vec3(0.36, 1.0, -10.0),
            vec3(0.36, 1.0, 10.0),
            vec3(-0.36, 1.0, 10.0),
        ],
    ] {
        for index in [0, 1, 2, 0, 2, 3] {
            vertices.push(corners[index]);
        }
    }
    SceneDescription {
        meshes: vec![MeshAsset {
            id: 1,
            vertices,
            indices: Vec::new(),
            opaque: true,
        }],
        materials: vec![Material {
            id: 1,
            absorption_bands: vec![1.0],
            scattering: 0.0,
            transmission: None,
            roughness: 0.0,
        }],
        objects: vec![SceneObject {
            id: 1,
            mesh_id: 1,
            material_id: 1,
            transform: Mat4::IDENTITY,
            active: true,
        }],
        diffraction_edges: [-1.0, 1.0]
            .into_iter()
            .enumerate()
            .map(|(i, side)| DiffractionEdge {
                id: i as u32 + 1,
                start: vec3(side * 0.36, 1.0, -10.0),
                end: vec3(side * 0.36, 1.0, 10.0),
                bisector_dir: vec3(side, 1.0, 0.0).normalize(),
                edge_angle_radians: 1.5 * std::f32::consts::PI,
                diffraction_radius: 0.1,
                base_strength: 1.0,
            })
            .collect(),
    }
}

fn samples(core: &EngineCore, voice: VoiceId) -> Vec<IrSample> {
    let buffer = &core.spatial_voices[&voice].acoustic_ir_buffer;
    let mut result = vec![IrSample::zero(); buffer.size as usize / 8];
    core.gpu
        .memory()
        .download_typed(buffer, result.len(), &mut result)
        .unwrap();
    result
}

fn assert_same_pressure(actual: &[IrSample], reference: &IrSnapshot) {
    let error: f64 = actual
        .iter()
        .zip(&reference.samples)
        .map(|(a, b)| {
            (a.left as f64 - b.left as f64).powi(2) + (a.right as f64 - b.right as f64).powi(2)
        })
        .sum();
    assert_eq!(actual.len(), reference.samples.len());
    assert!(
        reference.energy > 1e-8,
        "fixture must carry diffracted pressure"
    );
    assert!(
        (error / reference.energy as f64).sqrt() < 1e-4,
        "staging changed stereo ray pressure"
    );
}

#[test]
#[ignore = "requires Vulkan ray tracing; run hardware tests serially"]
fn scalar_updates_preserve_pressure_and_refresh_all_voices() {
    let config = EngineConfig {
        sound: SoundConfig {
            output_channels: OutputChannels::Stereo,
            sample_rate: 48000,
            ir_num_samples: 4096,
        },
        acoustics: AcousticsConfig {
            listener_ray_count: 65536,
            max_bounces: 2,
            maximum_path_length: 15.0,
            minimum_throughput: 0.0,
            sampling_seed: 7,
        },
        cache: IrCacheConfig {
            max_entries: 0,
            ..IrCacheConfig::default()
        },
        binaural: BinauralConfig::default(),
        auralization: AuralizationConfig {
            block_size: 1024,
        },
    };
    let mut core = EngineCore::new(config).unwrap();
    let scene = box_scene();
    core.load_scene(scene.clone()).unwrap();

    let source = PointSource::new(vec3(-2.0, 0.0, 0.0));
    let first = ListenerPose::new(vec3(2.0, 0.0, 0.0), Vec3::Z, Vec3::Y);
    let second = ListenerPose::new(vec3(2.5, 0.0, 0.0), Vec3::X, Vec3::Y);
    core.set_listener_pose(first).unwrap();

    let asset = core
        .insert_sound(
            SoundAsset::from_mono_samples(
                48000,
                (0..96000).map(|i| (i as f32 * 0.1).sin() * 0.1).collect(),
            )
            .unwrap(),
        )
        .unwrap();
    let source_id = core.create_sound_source(source).unwrap();
    let voice = core.play_sound_from_source(asset, source_id, 1.0).unwrap();
    let original = core.spatial_voices[&voice].acoustic_ir_buffer.clone();
    let mut output = vec![0.0; 2048];

    core.set_listener_pose(second).unwrap();
    core.render_realtime_block(&mut output, false).unwrap();
    assert!(core.spatial_voices[&voice].dirty);
    assert!(Arc::ptr_eq(
        &original,
        &core.spatial_voices[&voice].acoustic_ir_buffer
    ));

    let mut moved_source = source;
    moved_source.position.z = 0.3;
    moved_source.amplitude = 0.7;
    core.update_sound_source(source_id, moved_source).unwrap();
    let reference = core.build_impulse_response(moved_source, second).unwrap();
    core.render_realtime_block(&mut output, true).unwrap();
    assert!(!core.spatial_voices[&voice].dirty);
    assert_same_pressure(&samples(&core, voice), &reference);

    let other = core.play_sound_from_source(asset, source_id, 1.0).unwrap();
    let previous = [
        core.spatial_voices[&voice].impulse_response_id,
        core.spatial_voices[&other].impulse_response_id,
    ];

    for step in 0..2 {
        let listener = ListenerPose::new(vec3(2.2 + step as f32 * 0.1, 0.0, 0.0), Vec3::Z, Vec3::Y);
        core.set_listener_pose(listener).unwrap();
        core.render_realtime_block(&mut output, true).unwrap();
        assert!(output.iter().all(|value| value.is_finite()));
    }

    assert_ne!(core.spatial_voices[&voice].impulse_response_id, previous[0]);
    assert_ne!(core.spatial_voices[&other].impulse_response_id, previous[1]);
    core.stop_voice(other);

    let mut blocked = scene;
    blocked.diffraction_edges.clear();
    core.load_scene(blocked).unwrap();
    core.render_realtime_block(&mut output, true).unwrap();
    assert!(
        samples(&core, voice)
            .iter()
            .all(|sample| sample.left == 0.0 && sample.right == 0.0)
    );
    core.stop_voice(voice);
    assert!(!core.has_active_voices());
}
