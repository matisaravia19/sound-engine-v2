use super::backend::VkBackend;
use crate::acoustics::AcousticsEngine;
use crate::auralization::{AuralizationEngine, PlaySoundRequest, SoundAsset};
use crate::core::config::*;
use crate::engine::{ListenerPose, PointSource};
use crate::scene::*;
use glam::{Mat4, Vec3};
use std::sync::Arc;

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn audit_repeated_engine_teardown_returns_live_allocations_to_baseline() {
    let gpu = Arc::new(VkBackend::new().unwrap());
    let device = Arc::downgrade(&gpu.device_context());
    exercise_engine_lifecycle(&gpu);
    let baseline = gpu.memory().live_allocation_stats().unwrap();
    // A retained allocation must be observable; reserved VMA pool blocks are
    // deliberately excluded because the allocator may reuse them across cycles.
    let retained = gpu.memory().create_storage_buffer(4096).unwrap();
    let with_retained = gpu.memory().live_allocation_stats().unwrap();
    assert_eq!(with_retained.0, baseline.0 + 1);
    assert!(with_retained.1 >= baseline.1 + 4096);
    drop(retained);
    for cycle in 0..24 {
        exercise_engine_lifecycle(&gpu);
        assert_eq!(
            gpu.memory().live_allocation_stats().unwrap(),
            baseline,
            "cycle {cycle}"
        );
        assert_eq!(gpu.rt().resource_counts(), (0, 0), "cycle {cycle}");
        assert_eq!(gpu.rt().pipeline_count(), 0, "cycle {cycle}");
        assert_eq!(gpu.compute().pipeline_count(), 0, "cycle {cycle}");
    }
    eprintln!(
        "24 teardown cycles returned to {} live VMA allocations / {} bytes",
        baseline.0, baseline.1
    );
    drop(gpu);
    assert!(device.upgrade().is_none());
}

fn exercise_engine_lifecycle(gpu: &Arc<VkBackend>) {
    let config = EngineConfig {
        sound: SoundConfig {
            sample_rate: 48_000,
            output_channels: OutputChannels::Stereo,
            ir_num_samples: 1024,
        },
        acoustics: AcousticsConfig {
            listener_ray_count: 1,
            max_bounces: 0,
            maximum_path_length: 100.0,
            minimum_throughput: 0.0,
            sampling_seed: 0,
        },
        cache: Default::default(),
        binaural: Default::default(),
        auralization: AuralizationConfig {
            block_size: 64,
        },
    };
    let mut acoustic = AcousticsEngine::new(gpu, config).unwrap();
    let mesh = MeshAsset {
        id: 1,
        vertices: vec![
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(1.0, 5.0, 0.0),
            Vec3::new(0.0, 5.0, 1.0),
        ],
        indices: vec![],
        opaque: true,
    };
    acoustic
        .load_scene(SceneDescription {
            meshes: vec![mesh.clone()],
            materials: vec![Material {
                id: 1,
                absorption: 0.2,
                scattering: 0.0,
                roughness: 0.0,
            }],
            objects: vec![SceneObject {
                id: 1,
                mesh_id: 1,
                material_id: 1,
                transform: Mat4::IDENTITY,
                active: true,
            }],
            diffraction_edges: vec![DiffractionEdge {
                id: 1,
                start: Vec3::new(0.0, 10.0, 0.0),
                end: Vec3::new(0.0, 11.0, 0.0),
                bisector_dir: Vec3::X,
                edge_angle_radians: std::f32::consts::TAU,
                diffraction_radius: 0.1,
            }],
        })
        .unwrap();
    let mut audio =
        AuralizationEngine::new(gpu.clone(), config.sound, config.auralization).unwrap();
    let sound = audio
        .insert_sound(SoundAsset::from_mono_samples(48_000, vec![0.01; 1024]).unwrap())
        .unwrap();
    let mut voice = None;
    let mut output = vec![0.0; 128];
    for step in 0..3 {
        if step != 0 {
            let mut replacement = mesh.clone();
            replacement.vertices[1].x += step as f32 * 0.1;
            acoustic
                .apply_scene_updates(SceneUpdates {
                    updates: vec![
                        SceneUpdate::ReplaceMesh(replacement),
                        SceneUpdate::SetTransform {
                            id: 1,
                            transform: Mat4::from_translation(Vec3::Y * step as f32),
                        },
                    ],
                })
                .unwrap();
        }
        let snapshot = acoustic
            .build_impulse_response_gpu(gpu, PointSource::new(Vec3::X), ListenerPose::default())
            .unwrap();
        audio
            .register_impulse_response_from_gpu_ir(1, &snapshot)
            .unwrap();
        match voice {
            Some(id) => audio.update_voice_impulse_response(id, 1).unwrap(),
            None => voice = Some(audio.play_sound(PlaySoundRequest::new(sound, 1)).unwrap()),
        }
        // Voices retain their filters after registry removal and through swaps.
        audio.forget_impulse_response(1);
        for _ in 0..4 {
            audio.render_block(&mut output).unwrap();
            assert!(output.iter().all(|value| value.is_finite()));
        }
    }
    audio.stop_voice(voice.unwrap());
    assert!(!audio.has_active_voices());
}
