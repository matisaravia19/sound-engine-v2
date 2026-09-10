//! Isolated source-floor-edge-listener pressure checks for the optional family.
use super::*;
use sound_engine::acoustics::IrSample;

fn fixture() -> SceneDescription {
    let mut scene = geometry::scene_from_walls(vec![geometry::quad_x(0.0, 0.0, 2.0, -20.0, 20.0)], 1.0);
    let mut floor = geometry::scene_from_walls(vec![geometry::quad_y(0.0, -20.0, 20.0, -20.0, 20.0)], 0.0);
    floor.meshes[0].id = 2;
    floor.materials[0].id = 2;
    floor.objects[0].id = 2;
    floor.objects[0].mesh_id = 2;
    floor.objects[0].material_id = 2;
    scene.meshes.extend(floor.meshes);
    scene.materials.extend(floor.materials);
    scene.objects.extend(floor.objects);
    scene.diffraction_edges.push(edge(
        1,
        vec3(0.0, 2.0, -20.0),
        vec3(0.0, 2.0, 20.0),
        Vec3::Y,
        std::f32::consts::TAU,
        0.2,
    ));
    scene
}

fn added_pressure(
    pipeline: &mut ExperimentalUtdPipeline,
    gpu: &VkBackend,
    scene: &mut SceneManager,
    query: AcousticQuery,
) -> Vec<IrSample> {
    pipeline.set_source_surface_paths(false);
    let raw = pipeline.build_ir_gpu(gpu, scene, query).unwrap();
    let control = pipeline.download_gpu_ir(gpu, &raw).unwrap();
    pipeline.set_source_surface_paths(true);
    let raw = pipeline.build_ir_gpu(gpu, scene, query).unwrap();
    let candidate = pipeline.download_gpu_ir(gpu, &raw).unwrap();
    candidate
        .samples
        .iter()
        .zip(&control.samples)
        .map(|(a, b)| IrSample {
            left: a.left - b.left,
            right: a.right - b.right,
        })
        .collect()
}

#[test]
#[ignore = "requires Vulkan ray tracing and glslangValidator"]
fn source_surface_edge_matches_plane_reflection_reference() {
    let _guard = super::tests::GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let gpu = VkBackend::new().unwrap();
    let config = EngineConfig {
        sound: SoundConfig {
            output_channels: OutputChannels::Mono,
            sample_rate: 48000,
            ir_num_samples: 4096,
        },
        acoustics: AcousticsConfig {
            listener_ray_count: 1_048_576,
            max_bounces: 2,
            maximum_path_length: 20.0,
            minimum_throughput: 0.0,
            sampling_seed: 7,
        },
        cache: IrCacheConfig::default(),
        binaural: BinauralConfig::default(),
        auralization: AuralizationConfig { block_size: 1024 },
    };
    let mut pipeline = ExperimentalUtdPipeline::with_band_orders(&gpu, config, 0.04, 20000.0, 16, 2).unwrap();
    let mut scene = SceneManager::new();
    let description = fixture();
    scene.load(description.clone()).unwrap();
    let query = AcousticQuery {
        query_id: 0,
        source_position: vec3(-2.0, 0.5, 0.0),
        listener_position: vec3(3.0, 1.2, 0.0),
        listener_forward: -Vec3::Z,
        listener_up: Vec3::Y,
        source_amplitude: 1.0,
        source_radiation: Default::default(),
    };
    let added = added_pressure(&mut pipeline, &gpu, &mut scene, query);
    let norm: f32 = added.iter().map(|v| v.left * v.left).sum();
    assert!(norm > 1e-8, "source-floor-edge family was missed");
    let fft = realfft::RealFftPlanner::<f32>::new().plan_fft_forward(4096);
    let mut transfer = fft.make_output_vec();
    fft.process(&mut added.iter().map(|v| v.left).collect::<Vec<_>>(), &mut transfer)
        .unwrap();
    // Independent SciPy coefficient and unfolded floor distance are evaluated
    // offline in mixed_surface_stationary_reference.py. Runtime uses real hits.
    for (bin, expected) in [
        (23, [0.0292105633, -0.0134676988]),
        (93, [0.00872376669, -0.0136875048]),
        (163, [0.000515799908, -0.0122592449]),
    ] {
        let reference = realfft::num_complex::Complex32::new(expected[0], expected[1]);
        let error = (transfer[bin] - reference).norm() / reference.norm();
        eprintln!(
            "source-floor-edge bin={bin}: {:?}, expected={reference:?}, relative={error}",
            transfer[bin]
        );
        assert!(
            error < 0.05,
            "reflected-edge pressure normalization differs from the independent reference"
        );
    }
    // The source-to-floor departure points downward, unlike the final edge leg.
    // This independent stationary angle catches using the wrong path direction.
    let profile = sound_engine::acoustics::SourceDirectivity::new(
        1,
        19,
        vec![20.0, 20000.0],
        (0..19).flat_map(|t| [[1.0 - 0.5 * t as f32 / 18.0, 0.0]; 2]).collect(),
    )
    .unwrap();
    let mut directional_query = query;
    directional_query.source_radiation.profile = pipeline.register_source_directivity(&gpu, &profile).unwrap();
    let directional = added_pressure(&mut pipeline, &gpu, &mut scene, directional_query);
    let expected_gain = 1.0 - 0.5 * 2.5_f32.atan2(2.0) / std::f32::consts::PI;
    let ratio_error = (directional
        .iter()
        .zip(&added)
        .map(|(a, b)| (a.left - expected_gain * b.left).powi(2))
        .sum::<f32>()
        / norm)
        .sqrt();
    eprintln!("source-floor-edge directional scaling error={ratio_error}, expected gain={expected_gain}");
    assert!(
        ratio_error < 0.01,
        "source-launched rays used the wrong departure direction"
    );
    let mut half_floor = description.clone();
    half_floor.materials[1].absorption_bands = vec![0.75];
    scene.load(half_floor).unwrap();
    let half = added_pressure(&mut pipeline, &gpu, &mut scene, query);
    let residual: f32 = half
        .iter()
        .zip(&added)
        .map(|(a, b)| (a.left - 0.5 * b.left).powi(2))
        .sum();
    assert!(
        (residual / norm).sqrt() < 1e-4,
        "floor energy coefficient was not converted to carried pressure"
    );
    let mut absorbing = description;
    absorbing.materials[1].absorption_bands = vec![1.0];
    scene.load(absorbing).unwrap();
    let silent = added_pressure(&mut pipeline, &gpu, &mut scene, query);
    assert!(silent.iter().all(|v| v.left.abs() < 1e-7 && v.right.abs() < 1e-7));
    // With the floor absorbing, only the source-edge-listener family survives.
    pipeline.set_source_surface_paths(false);
    let raw = pipeline.build_ir_gpu(&gpu, &mut scene, query).unwrap();
    let single = pipeline.download_gpu_ir(&gpu, &raw).unwrap();
    let raw = pipeline.build_ir_gpu(&gpu, &mut scene, directional_query).unwrap();
    let directional_single = pipeline.download_gpu_ir(&gpu, &raw).unwrap();
    let expected_gain = 1.0 - 0.5 * 1.5_f32.atan2(2.0) / std::f32::consts::PI;
    let single_error = (directional_single
        .samples
        .iter()
        .zip(&single.samples)
        .map(|(a, b)| (a.left - expected_gain * b.left).powi(2))
        .sum::<f32>()
        / single.energy)
        .sqrt();
    eprintln!("single-edge source direction error={single_error}, expected gain={expected_gain}");
    assert!(single.energy > 1e-8 && single_error < 0.01);
}

#[test]
#[ignore = "requires Vulkan ray tracing and glslangValidator"]
fn coplanar_surface_edge_does_not_create_a_reflection_path() {
    let _guard = super::tests::GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let gpu = VkBackend::new().unwrap();
    let config = EngineConfig {
        sound: SoundConfig {
            output_channels: OutputChannels::Mono,
            sample_rate: 48000,
            ir_num_samples: 4096,
        },
        acoustics: AcousticsConfig {
            listener_ray_count: 262_144,
            max_bounces: 2,
            maximum_path_length: 20.0,
            minimum_throughput: 0.0,
            sampling_seed: 7,
        },
        cache: IrCacheConfig::default(),
        binaural: BinauralConfig::default(),
        auralization: AuralizationConfig { block_size: 1024 },
    };
    // Both wedge faces end at the same edge. Any reflection followed by that
    // edge would have its outgoing leg tangent to the reflecting plane, which
    // cannot be the specular reflection of a ray from this off-plane source.
    let mut description = geometry::scene_from_walls(
        vec![
            geometry::quad_x(0.0, -10.0, 2.0, -20.0, 20.0),
            geometry::quad_y(2.0, 0.0, 10.0, -20.0, 20.0),
        ],
        0.0,
    );
    let n = std::f32::consts::FRAC_1_SQRT_2;
    description.diffraction_edges.push(edge(
        1,
        vec3(0.0, 2.0, -20.0),
        vec3(0.0, 2.0, 20.0),
        vec3(-n, n, 0.0),
        1.5 * std::f32::consts::PI,
        0.2,
    ));
    let mut scene = SceneManager::new();
    scene.load(description).unwrap();
    let mut pipeline = ExperimentalUtdPipeline::with_band_orders(&gpu, config, 0.04, 20000.0, 16, 2).unwrap();
    let query = AcousticQuery {
        query_id: 0,
        source_position: vec3(-0.25, -2.0, 0.0),
        listener_position: vec3(-2.0, 0.5, 0.0),
        listener_forward: -Vec3::Z,
        listener_up: Vec3::Y,
        source_amplitude: 1.0,
        source_radiation: Default::default(),
    };
    let residual = added_pressure(&mut pipeline, &gpu, &mut scene, query);
    let energy: f32 = residual.iter().map(|v| v.left * v.left).sum();
    eprintln!("coplanar reflected-edge added energy={energy}");
    assert!(
        energy < 1e-12,
        "proxy projection manufactured a coplanar reflection path"
    );
}
