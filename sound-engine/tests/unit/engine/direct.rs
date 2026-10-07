//! Hardware regression for coherent spatial updates across rendered blocks.

use super::*;
use crate::acoustics::IrSample;
use crate::core::config::*;
use crate::scene::{DiffractionEdge, Material, MeshAsset, SceneObject};
use glam::{Mat4, Vec3, vec3};

fn preflight_config() -> EngineConfig {
    EngineConfig {
        sound: SoundConfig {
            output_channels: OutputChannels::Stereo,
            sample_rate: 48_000,
            ir_num_samples: 1024,
        },
        acoustics: AcousticsConfig {
            listener_ray_count: 1,
            max_bounces: 0,
            maximum_path_length: 15.0,
            minimum_throughput: 0.0,
            sampling_seed: 7,
        },
        cache: IrCacheConfig::default(),
        binaural: BinauralConfig::default(),
        auralization: AuralizationConfig {
            block_size: 64,
        },
    }
}

#[test]
#[ignore = "requires Vulkan ray tracing; run hardware tests serially"]
fn fixed_and_tracked_sources_have_independent_lifetimes() {
    let mut core = EngineCore::new(preflight_config()).unwrap();
    let source = PointSource::new(Vec3::X);
    let source_id = core.create_sound_source(source).unwrap();
    let sound = core
        .insert_sound(SoundAsset::from_mono_samples(48_000, vec![0.1; 64]).unwrap())
        .unwrap();
    let tracked = core.play_sound_from_source(sound, source_id, 1.0).unwrap();

    // A one-shot owns its source directly, even when tracked handles are exhausted.
    core.next_sound_source_id = u64::MAX;
    let fixed = core
        .play_sound(PlaySpatialSoundRequest::new(sound, source.position))
        .unwrap();
    assert_eq!(core.sound_sources.len(), 1);
    assert_eq!(core.next_sound_source_id, u64::MAX);
    assert!(core.remove_sound_source(source_id).is_err());

    core.update_sound_source(source_id, source).unwrap();
    assert!(core.spatial_voices[&tracked].refresh_requested_at.is_none());
    let moved = PointSource::new(Vec3::Y);
    core.update_sound_source(source_id, moved).unwrap();
    assert!(core.spatial_voices[&tracked].refresh_requested_at.is_some());
    assert!(core.spatial_voices[&fixed].refresh_requested_at.is_none());
    assert_eq!(
        core.resolve_source(core.spatial_voices[&fixed].source)
            .unwrap(),
        source
    );
    assert_eq!(
        core.resolve_source(core.spatial_voices[&tracked].source)
            .unwrap(),
        moved
    );

    core.set_listener_pose(ListenerPose::new(Vec3::Z, Vec3::Z, Vec3::Y))
        .unwrap();
    assert!(core.spatial_voices[&fixed].refresh_requested_at.is_some());
    let mut output = vec![0.0; 128];
    core.render_block(&mut output).unwrap();
    assert!(
        core.spatial_voices
            .values()
            .all(|voice| voice.refresh_requested_at.is_none())
    );
    core.stop_voice(fixed);
    assert_eq!(core.sound_sources.len(), 1);

    for _ in 0..64 {
        core.render_block(&mut output).unwrap();
        if !core.has_active_voices() {
            break;
        }
    }
    assert!(!core.has_active_voices());
    assert!(core.spatial_voices.is_empty());
    assert_eq!(
        core.sound_sources.len(),
        1,
        "finishing a voice must retain its tracked source"
    );
    core.remove_sound_source(source_id).unwrap();
    assert!(core.sound_sources.is_empty());
}

#[test]
#[ignore = "requires Vulkan ray tracing; run hardware tests serially"]
fn audit_spatial_preflight_rejects_invalid_sounds_without_gpu_resources() {
    let mut core = EngineCore::new(preflight_config()).unwrap();
    core.load_scene(box_scene()).unwrap();
    let source = PointSource::new(vec3(-2.0, 0.0, 0.0));
    let source_id = core.create_sound_source(source).unwrap();
    let empty = core
        .insert_sound(SoundAsset::from_mono_samples(48_000, Vec::new()).unwrap())
        .unwrap();
    // Exercise the lower-level registry API, which intentionally permits other rates.
    let wrong_rate = core
        .auralization
        .sound_bank_mut()
        .insert(SoundAsset::from_mono_samples(44_100, vec![0.1; 64]).unwrap())
        .unwrap();
    let baseline = core.gpu.memory().live_allocation_stats().unwrap();
    let next_source = core.next_sound_source_id;
    for (sound, code) in [
        (SoundId(u64::MAX), crate::core::error::ErrorCode::NotFound),
        (empty, crate::core::error::ErrorCode::InvalidArgument),
        (wrong_rate, crate::core::error::ErrorCode::InvalidArgument),
    ] {
        for tracked in [true, false] {
            let error = if tracked {
                core.play_sound_from_source(sound, source_id, 1.0)
            } else {
                core.play_sound(PlaySpatialSoundRequest::new(sound, source.position))
            }
            .unwrap_err();
            assert_eq!(error.code(), code);
            assert_eq!(
                core.next_sound_source_id, next_source,
                "failed play consumed a source ID"
            );
            assert_eq!(core.sound_sources.len(), 1);
            assert!(core.spatial_voices.is_empty());
            assert!(!core.has_active_voices());
            assert_eq!(core.gpu.memory().live_allocation_stats().unwrap(), baseline);
        }
    }
    let valid = core
        .insert_sound(SoundAsset::from_mono_samples(48_000, vec![0.1; 64]).unwrap())
        .unwrap();
    let voice = core.play_sound_from_source(valid, source_id, 1.0).unwrap();
    assert!(core.has_active_voices());
    core.stop_voice(voice);
    core.remove_sound_source(source_id).unwrap();
}

#[test]
#[ignore = "requires Vulkan ray tracing; run hardware tests serially"]
fn audit_spatial_preflight_rejects_invalid_output_without_advancing_voices() {
    let mut core = EngineCore::new(preflight_config()).unwrap();
    core.load_scene(box_scene()).unwrap();
    let sound = core
        .insert_sound(SoundAsset::from_mono_samples(48_000, vec![0.1; 1024]).unwrap())
        .unwrap();
    let voice = core
        .play_sound(PlaySpatialSoundRequest::new(sound, vec3(-2.0, 0.0, 0.0)))
        .unwrap();
    let mut step = 0;
    for realtime in [false, true] {
        for invalid_len in [0, 127, 129] {
            step += 1;
            core.set_listener_pose(ListenerPose::new(
                vec3(step as f32, 0.0, 0.0),
                Vec3::Z,
                Vec3::Y,
            ))
            .unwrap();
            let old_ir = core.spatial_voices[&voice].acoustic_ir_buffer.clone();
            let updates = core.spatial_updates;
            let cursor = core.last_spatial_voice;
            let baseline = core.gpu.memory().live_allocation_stats().unwrap();
            let mut invalid = vec![77.0; invalid_len];
            let error = if realtime {
                core.render_realtime_block(&mut invalid, true).map(|_| ())
            } else {
                core.render_block(&mut invalid)
            }
            .unwrap_err();
            assert_eq!(error.code(), crate::core::error::ErrorCode::InvalidArgument);
            assert!(invalid.iter().all(|&sample| sample == 77.0));
            assert!(
                core.spatial_voices[&voice].refresh_requested_at.is_some(),
                "failed render consumed a spatial update"
            );
            assert!(Arc::ptr_eq(
                &old_ir,
                &core.spatial_voices[&voice].acoustic_ir_buffer
            ));
            assert_eq!(core.spatial_updates, updates);
            assert_eq!(core.last_spatial_voice, cursor);
            assert_eq!(core.gpu.memory().live_allocation_stats().unwrap(), baseline);

            let mut valid = vec![0.0; 128];
            if realtime {
                core.render_realtime_block(&mut valid, true).unwrap();
            } else {
                core.render_block(&mut valid).unwrap();
            }
            assert!(valid.iter().all(|sample| sample.is_finite()));
            assert!(core.spatial_voices[&voice].refresh_requested_at.is_none());
            assert!(!Arc::ptr_eq(
                &core.spatial_voices[&voice].acoustic_ir_buffer,
                &old_ir
            ));
        }
    }
    core.stop_voice(voice);
    assert!(core.sound_sources.is_empty());
}

#[test]
#[ignore = "requires Vulkan ray tracing; run hardware tests serially"]
fn returning_to_cached_pose_completes_refresh_without_replacing_the_voice_filter() {
    let mut core = EngineCore::new(preflight_config()).unwrap();
    let sound = core
        .insert_sound(SoundAsset::from_mono_samples(48_000, vec![0.1; 1024]).unwrap())
        .unwrap();
    let voice = core
        .play_sound(PlaySpatialSoundRequest::new(sound, Vec3::X))
        .unwrap();
    let original_pose = core.listener;
    let original_buffer = core.spatial_voices[&voice].acoustic_ir_buffer.clone();
    let mut output = vec![0.0; 128];

    for realtime in [false, true] {
        // Several accepted poses may coalesce before either rendering mode
        // refreshes the voice. Returning to its cached response still consumes
        // the pending request, without starting an unnecessary crossfade.
        let moved = ListenerPose::new(Vec3::Y, Vec3::Z, Vec3::Y);
        core.set_listener_pose(moved).unwrap();
        core.set_listener_pose(original_pose).unwrap();
        assert!(core.spatial_voices[&voice].refresh_requested_at.is_some());
        if realtime {
            core.render_realtime_block(&mut output, true).unwrap();
        } else {
            core.render_block(&mut output).unwrap();
        }
        assert!(core.spatial_voices[&voice].refresh_requested_at.is_none());
        assert!(Arc::ptr_eq(
            &core.spatial_voices[&voice].acoustic_ir_buffer,
            &original_buffer
        ));
        assert_eq!(core.spatial_updates, 0);
        assert!(output.iter().all(|sample| sample.is_finite()));
    }
}

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
            absorption: 1.0,
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
    let metrics = core.render_realtime_block(&mut output, false).unwrap();
    assert!(metrics.spatial_deferred);
    assert_eq!(metrics.trace_ms, 0.0);
    assert_eq!(metrics.ir_prepare_ms, 0.0);
    assert!(metrics.convolution_ms.is_finite() && metrics.convolution_ms >= 0.0);
    assert_eq!(metrics.spatial_updates, 0);
    assert_eq!(metrics.spatial_update_age_ms, 0.0);
    assert!(core.spatial_voices[&voice].refresh_requested_at.is_some());
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
    assert!(core.spatial_voices[&voice].refresh_requested_at.is_none());
    assert_same_pressure(&samples(&core, voice), &reference);

    let other = core.play_sound_from_source(asset, source_id, 1.0).unwrap();
    let previous = [
        core.spatial_voices[&voice].acoustic_ir_buffer.clone(),
        core.spatial_voices[&other].acoustic_ir_buffer.clone(),
    ];

    for step in 0..2 {
        let listener = ListenerPose::new(vec3(2.2 + step as f32 * 0.1, 0.0, 0.0), Vec3::Z, Vec3::Y);
        core.set_listener_pose(listener).unwrap();
        core.render_realtime_block(&mut output, true).unwrap();
        assert!(output.iter().all(|value| value.is_finite()));
    }

    assert!(!Arc::ptr_eq(
        &core.spatial_voices[&voice].acoustic_ir_buffer,
        &previous[0]
    ));
    assert!(!Arc::ptr_eq(
        &core.spatial_voices[&other].acoustic_ir_buffer,
        &previous[1]
    ));
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

impl EngineCore {
    /// Observes final device release without extending its lifetime in lifecycle tests.
    #[cfg(feature = "playback")]
    pub(in crate::engine) fn device_lifetime(
        &self,
    ) -> std::sync::Weak<crate::gpu::backend::VkDeviceContext> {
        Arc::downgrade(&self.gpu.device_context())
    }
}
