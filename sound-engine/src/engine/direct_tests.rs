//! Hardware regression for coherent spatial updates across rendered blocks.

use super::*;
use crate::acoustics::{AcousticModel, IrSample};
use crate::core::config::*;
use crate::scene::{DiffractionEdge, Material, MeshAsset, SceneObject};
use glam::{Mat4, Vec3, vec3};

#[test]
#[ignore = "requires Vulkan ray tracing; run hardware tests serially"]
fn source_directivity_keeps_complex_pressure_and_source_orientation() {
    check_source_directivity(16);
    check_source_directivity(64);
}

fn check_source_directivity(band_count: u32) {
    let config = EngineConfig {
        sound: SoundConfig {
            sample_rate: 48000,
            ir_num_samples: 4096,
            output_channels: OutputChannels::Stereo,
        },
        acoustics: AcousticsConfig {
            listener_ray_count: 1024,
            max_bounces: 0,
            maximum_path_length: 20.0,
            minimum_throughput: 0.0,
            sampling_seed: 7,
        },
        cache: IrCacheConfig::default(),
        binaural: BinauralConfig::default(),
        auralization: AuralizationConfig { block_size: 1024 },
    };
    let mut core = EngineCore::new(
        config,
        AcousticModel::Utd {
            cone_width: 0.04,
            maximum_frequency_hz: 20000.0,
            diffraction_orders: 2,
            band_count,
            surface_two_edge_paths: false,
        },
    )
    .unwrap();
    let mut values = Vec::new();
    for p in 0..8 {
        for t in 0..5 {
            let theta = t as f32 * std::f32::consts::PI / 4.0;
            let phi = p as f32 * std::f32::consts::PI / 4.0;
            let y = if t == 0 || t == 4 { 0.0 } else { theta.sin() * phi.cos() };
            values.extend([[0.625 + 0.375 * theta.cos(), 0.1 * y]; 2]);
        }
    }
    let profile = crate::acoustics::SourceDirectivity::new(8, 5, vec![20.0, 20000.0], values).unwrap();
    let id = core.register_source_directivity(&profile).unwrap();
    let mut source = PointSource::new(Vec3::ZERO);
    source.radiation.profile = id;
    let listener = ListenerPose::new(vec3(3.0, 0.0, 0.0), -Vec3::X, Vec3::Y);
    let fft = realfft::RealFftPlanner::<f32>::new().plan_fft_forward(4096);
    for (rotation, gain) in [
        (glam::Quat::IDENTITY, [1.0, 0.0]),
        (glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_2), [0.625, -0.1]),
        (glam::Quat::from_rotation_y(std::f32::consts::PI), [0.25, 0.0]),
    ] {
        source.radiation.orientation = rotation;
        let ir = core.build_impulse_response(source, listener).unwrap();
        assert!(ir.samples.iter().all(|v| v.left.is_finite() && v.right.is_finite()));
        for ear in 0..2 {
            let mut spectrum = fft.make_output_vec();
            let mut pressure: Vec<_> = ir
                .samples
                .iter()
                .map(|v| if ear == 0 { v.left } else { v.right })
                .collect();
            fft.process(&mut pressure, &mut spectrum).unwrap();
            for bin in [23, 93, 163] {
                let frequency = bin as f32 * 48000.0 / 4096.0;
                let phase = -std::f32::consts::TAU * frequency * 3.0 / 343.0;
                let expected = realfft::num_complex::Complex32::new(gain[0], gain[1])
                    * realfft::num_complex::Complex32::from_polar(std::f32::consts::FRAC_1_SQRT_2 / 3.0, phase);
                let error = (spectrum[bin] - expected).norm() / expected.norm();
                assert!(
                    error < 0.01,
                    "source orientation/complex pressure mismatch ear={ear}, bin={bin}, error={error}"
                );
            }
        }
    }
    source.radiation.profile = id + 1;
    assert!(
        core.create_sound_source(source).is_err(),
        "unknown source asset must fail before mutation"
    );
    source.radiation.profile = id;
    source.radiation.orientation = glam::Quat::from_array([0.0; 4]);
    assert!(core.build_impulse_response(source, listener).is_err());
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
        .map(|(a, b)| (a.left as f64 - b.left as f64).powi(2) + (a.right as f64 - b.right as f64).powi(2))
        .sum();
    assert_eq!(actual.len(), reference.samples.len());
    assert!(reference.energy > 1e-8, "fixture must carry diffracted pressure");
    assert!(
        (error / reference.energy as f64).sqrt() < 1e-4,
        "staging changed stereo ray pressure"
    );
}

#[test]
#[ignore = "requires Vulkan ray tracing; run hardware tests serially"]
fn staged_updates_preserve_pressure_and_newer_poses() {
    check_staged_source_updates(false, false);
}

#[test]
#[ignore = "requires Vulkan ray tracing; run hardware tests serially"]
fn reflected_ray_uses_complex_gain_at_physical_source_departure() {
    let config = EngineConfig {
        sound: SoundConfig {
            sample_rate: 48000,
            ir_num_samples: 4096,
            output_channels: OutputChannels::Mono,
        },
        acoustics: AcousticsConfig {
            listener_ray_count: 1_048_576,
            max_bounces: 1,
            maximum_path_length: 20.0,
            minimum_throughput: 0.0,
            sampling_seed: 7,
        },
        cache: IrCacheConfig::default(),
        binaural: BinauralConfig::default(),
        auralization: AuralizationConfig { block_size: 1024 },
    };
    let mut core = EngineCore::new(
        config,
        AcousticModel::Utd {
            cone_width: 0.04,
            maximum_frequency_hz: 20000.0,
            diffraction_orders: 2,
            band_count: 16,
            surface_two_edge_paths: false,
        },
    )
    .unwrap();
    // Linear polar-angle response has an exact interpolation reference at the
    // planar specular direction. Its imaginary component checks source phase.
    let profile = crate::acoustics::SourceDirectivity::new(
        1,
        19,
        vec![20.0, 20000.0],
        (0..19)
            .flat_map(|t| [[1.0 - 0.5 * t as f32 / 18.0, 0.1 * t as f32 / 18.0]; 2])
            .collect(),
    )
    .unwrap();
    let mut source = PointSource::new(vec3(-2.0, 1.0, 0.0));
    source.radiation.profile = core.register_source_directivity(&profile).unwrap();
    let listener = ListenerPose::new(vec3(3.0, 1.0, 0.0), -Vec3::X, Vec3::Y);
    let floor = SceneDescription {
        meshes: vec![MeshAsset {
            id: 1,
            opaque: true,
            vertices: vec![
                vec3(-20.0, 0.0, -20.0),
                vec3(20.0, 0.0, -20.0),
                vec3(20.0, 0.0, 20.0),
                vec3(-20.0, 0.0, 20.0),
            ],
            indices: vec![0, 1, 2, 0, 2, 3],
        }],
        materials: vec![Material {
            id: 1,
            absorption_bands: vec![0.0],
            scattering: 0.0,
            roughness: 0.0,
            transmission: None,
        }],
        objects: vec![SceneObject {
            id: 1,
            mesh_id: 1,
            material_id: 1,
            transform: Mat4::IDENTITY,
            active: true,
        }],
        diffraction_edges: Vec::new(),
    };
    core.load_scene(floor.clone()).unwrap();
    let reflected = core.build_impulse_response(source, listener).unwrap();
    let mut absorbing = floor;
    absorbing.materials[0].absorption_bands = vec![1.0];
    core.load_scene(absorbing).unwrap();
    let control = core.build_impulse_response(source, listener).unwrap();
    let mut isolated: Vec<_> = reflected
        .samples
        .iter()
        .zip(control.samples)
        .map(|(a, b)| a.left - b.left)
        .collect();
    let fft = realfft::RealFftPlanner::<f32>::new().plan_fft_forward(4096);
    let mut spectrum = fft.make_output_vec();
    fft.process(&mut isolated, &mut spectrum).unwrap();
    let theta = 2.0_f32.atan2(5.0) / std::f32::consts::PI;
    let gain = realfft::num_complex::Complex32::new(1.0 - 0.5 * theta, 0.1 * theta);
    let length = 29.0_f32.sqrt();
    for bin in [23, 93, 163] {
        let f = bin as f32 * 48000.0 / 4096.0;
        let expected = gain
            * realfft::num_complex::Complex32::from_polar(1.0 / length, -std::f32::consts::TAU * f * length / 343.0);
        let error = (spectrum[bin] - expected).norm() / expected.norm();
        eprintln!("reflected source pressure bin={bin}, relative={error}");
        assert!(
            error < 0.05,
            "reflected pressure does not match independent planar source response"
        );
    }
}

#[test]
#[ignore = "requires Vulkan ray tracing; run hardware tests serially"]
fn staged_directional_sources_preserve_captured_and_newer_radiation() {
    check_staged_source_updates(true, false);
}

#[test]
#[ignore = "requires Vulkan ray tracing; run hardware tests serially"]
fn staged_surface_two_edges_preserve_stereo_and_source_updates() {
    check_staged_source_updates(true, true);
}

fn check_staged_source_updates(directional: bool, surface_two_edge_paths: bool) {
    let config = EngineConfig {
        sound: SoundConfig {
            output_channels: OutputChannels::Stereo,
            sample_rate: 48000,
            ir_num_samples: 4096,
        },
        acoustics: AcousticsConfig {
            listener_ray_count: 65537,
            max_bounces: if surface_two_edge_paths { 3 } else { 2 },
            maximum_path_length: 15.0,
            minimum_throughput: 0.0,
            sampling_seed: 7,
        },
        cache: IrCacheConfig {
            max_entries: 0,
            ..IrCacheConfig::default()
        },
        binaural: BinauralConfig::default(),
        auralization: AuralizationConfig { block_size: 1024 },
    };
    let model = AcousticModel::Utd {
        cone_width: 0.04,
        maximum_frequency_hz: 20000.0,
        diffraction_orders: 2,
        band_count: 16,
        surface_two_edge_paths,
    };
    let mut core = EngineCore::new(config, model).unwrap();
    let mut scene = box_scene();
    if surface_two_edge_paths {
        scene.meshes.push(MeshAsset {
            id: 2,
            vertices: vec![
                vec3(-20.0, 0.0, -20.0),
                vec3(20.0, 0.0, -20.0),
                vec3(20.0, 0.0, 20.0),
                vec3(-20.0, 0.0, 20.0),
            ],
            indices: vec![0, 1, 2, 0, 2, 3],
            opaque: true,
        });
        scene.materials.push(Material {
            id: 2,
            absorption_bands: vec![0.0],
            scattering: 0.0,
            roughness: 0.0,
            transmission: None,
        });
        scene.objects.push(SceneObject {
            id: 2,
            mesh_id: 2,
            material_id: 2,
            transform: Mat4::IDENTITY,
            active: true,
        });
    }
    core.load_scene(scene.clone()).unwrap();
    let mut source = PointSource::new(vec3(-3.0, 0.2, 0.0));
    let profiles: Vec<_> = if directional {
        [1.0, 0.5]
            .into_iter()
            .map(|scale| {
                crate::acoustics::SourceDirectivity::new(
                    1,
                    19,
                    vec![20.0, 20000.0],
                    (0..19)
                        .flat_map(|t| [[scale * (1.0 - 0.5 * t as f32 / 18.0), 0.0]; 2])
                        .collect(),
                )
                .unwrap()
            })
            .collect()
    } else {
        Vec::new()
    };
    for profile in &profiles {
        core.register_source_directivity(profile).unwrap();
    }
    if directional {
        source.radiation.profile = 1;
    }
    let mut updated_source = source;
    if directional {
        updated_source.radiation.profile = 2;
        updated_source.radiation.orientation = glam::Quat::from_rotation_y(0.5);
        updated_source.position.x -= 0.2;
    }
    let pose = |x, forward| ListenerPose::new(vec3(x, 0.4, 0.0), forward, Vec3::Y);
    let first = pose(1.5, -Vec3::Z);
    let second = pose(2.0, Vec3::Z);
    // Compare against a single unsplit launch, including an odd ray budget.
    let mut reference_pipeline =
        crate::acoustics::ExperimentalUtdPipeline::with_band_orders(core.gpu.as_ref(), config, 0.04, 20000.0, 16, 2)
            .unwrap();
    reference_pipeline.set_surface_two_edge_paths(surface_two_edge_paths);
    for profile in &profiles {
        reference_pipeline
            .register_source_directivity(core.gpu.as_ref(), profile)
            .unwrap();
    }
    let mut reference_scene = crate::scene::SceneManager::new();
    reference_scene.load(scene.clone()).unwrap();
    let mut reference = |source: PointSource, listener: ListenerPose| {
        let query = crate::acoustics::AcousticQuery {
            query_id: 0,
            source_position: source.position,
            source_amplitude: source.amplitude,
            source_radiation: source.radiation,
            listener_position: listener.position,
            listener_forward: listener.forward,
            listener_up: listener.up,
        };
        let snapshot = reference_pipeline
            .build_ir_gpu(core.gpu.as_ref(), &mut reference_scene, query)
            .unwrap();
        reference_pipeline
            .download_gpu_ir(core.gpu.as_ref(), &snapshot)
            .unwrap()
    };
    let reference_first = reference(source, first);
    let reference_second = reference(updated_source, second);
    core.set_listener_pose(pose(3.0, Vec3::X)).unwrap();
    let asset = core
        .insert_sound(
            SoundAsset::from_mono_samples(48000, (0..96000).map(|i| (i as f32 * 0.1).sin() * 0.1).collect()).unwrap(),
        )
        .unwrap();
    let source_id = core.create_sound_source(source).unwrap();
    let voice = core.play_sound_from_source(asset, source_id, 1.0).unwrap();
    let original = core.spatial_voices[&voice].acoustic_ir_buffer.clone();
    let mut output = vec![0.0; 2048];

    core.set_listener_pose(first).unwrap();
    core.render_realtime_block(&mut output, true).unwrap();
    assert!(core.pending_ir.is_some());
    assert!(Arc::ptr_eq(&original, &core.spatial_voices[&voice].acoustic_ir_buffer));
    core.set_listener_pose(second).unwrap();
    core.update_sound_source(source_id, updated_source).unwrap();
    core.render_realtime_block(&mut output, true).unwrap();
    assert!(core.pending_ir.is_some());
    assert!(Arc::ptr_eq(&original, &core.spatial_voices[&voice].acoustic_ir_buffer));
    finish_pending_ir(&mut core, &mut output);
    assert!(core.pending_ir.is_none());
    assert!(core.spatial_voices[&voice].dirty, "newer pose was lost");
    assert_same_pressure(&samples(&core, voice), &reference_first);
    core.render_realtime_block(&mut output, true).unwrap();
    finish_pending_ir(&mut core, &mut output);
    assert!(!core.spatial_voices[&voice].dirty);
    assert_same_pressure(&samples(&core, voice), &reference_second);

    // Continuous pose changes must not starve either of two active voices.
    let other = core.play_sound_from_source(asset, source_id, 1.0).unwrap();
    let previous = [
        core.spatial_voices[&voice].impulse_response_id,
        core.spatial_voices[&other].impulse_response_id,
    ];
    for step in 0..34 {
        core.set_listener_pose(pose(2.2 + step as f32 * 0.1, Vec3::Z)).unwrap();
        core.render_realtime_block(&mut output, true).unwrap();
        assert!(output.iter().all(|v| v.is_finite()));
        if core.spatial_voices[&voice].impulse_response_id != previous[0]
            && core.spatial_voices[&other].impulse_response_id != previous[1]
        {
            break;
        }
    }
    assert_ne!(core.spatial_voices[&voice].impulse_response_id, previous[0]);
    assert_ne!(core.spatial_voices[&other].impulse_response_id, previous[1]);
    core.stop_voice(other);
    core.set_listener_pose(first).unwrap();
    core.render_realtime_block(&mut output, true).unwrap();
    assert!(core.pending_ir.is_some());
    let mut blocked = scene;
    blocked.diffraction_edges.clear();
    core.load_scene(blocked).unwrap();
    assert!(core.pending_ir.is_none(), "old scene pressure survived replacement");
    core.render_realtime_block(&mut output, true).unwrap();
    finish_pending_ir(&mut core, &mut output);
    assert!(samples(&core, voice).iter().all(|v| v.left == 0.0 && v.right == 0.0));
    core.set_listener_pose(second).unwrap();
    core.render_realtime_block(&mut output, true).unwrap();
    assert!(core.pending_ir.is_some());
    core.stop_voice(voice);
    assert!(core.pending_ir.is_none());
    assert!(!core.has_active_voices());
}

/// Completes only the captured query; a newer dirty revision must remain pending work.
fn finish_pending_ir(core: &mut EngineCore, output: &mut [f32]) {
    // At most sixteen ray ranges and one synthesis stage, independent of timing.
    // Intermediate buffers must never become the audible IR.
    let previous: Vec<_> = core
        .spatial_voices
        .iter()
        .map(|(&id, voice)| (id, voice.acoustic_ir_buffer.clone()))
        .collect();
    for _ in 0..17 {
        if core.pending_ir.is_none() {
            return;
        }
        core.render_realtime_block(output, true).unwrap();
        if core.pending_ir.is_some() {
            for (id, buffer) in &previous {
                assert!(Arc::ptr_eq(buffer, &core.spatial_voices[id].acoustic_ir_buffer));
            }
        }
    }
    assert!(
        core.pending_ir.is_none(),
        "adaptive query failed to complete its finite ray set"
    );
}
