//! Independent planar-reflection limits for ray-carried R-D-D and D-D-R paths.
use super::*;

fn fixture(floor_absorption: f32) -> SceneDescription {
    let mut scene = geometry::scene_from_walls(
        vec![
            geometry::quad_x(-0.36, -10.0, 1.0, -10.0, 10.0),
            geometry::quad_x(0.36, -10.0, 1.0, -10.0, 10.0),
            geometry::quad_y(1.0, -0.36, 0.36, -10.0, 10.0),
        ],
        1.0,
    );
    let mut floor = geometry::scene_from_walls(
        vec![geometry::quad_y(0.0, -20.0, 20.0, -20.0, 20.0)],
        floor_absorption,
    );
    floor.meshes[0].id = 2;
    floor.materials[0].id = 2;
    floor.objects[0].id = 2;
    floor.objects[0].mesh_id = 2;
    floor.objects[0].material_id = 2;
    scene.meshes.extend(floor.meshes);
    scene.materials.extend(floor.materials);
    scene.objects.extend(floor.objects);
    let n = std::f32::consts::FRAC_1_SQRT_2;
    scene.diffraction_edges = vec![
        edge(
            1,
            vec3(-0.36, 1.0, -10.0),
            vec3(-0.36, 1.0, 10.0),
            vec3(-n, n, 0.0),
            1.5 * std::f32::consts::PI,
            0.1,
        ),
        edge(
            2,
            vec3(0.36, 1.0, -10.0),
            vec3(0.36, 1.0, 10.0),
            vec3(n, n, 0.0),
            1.5 * std::f32::consts::PI,
            0.1,
        ),
    ];
    scene
}

fn added(
    pipeline: &mut ExperimentalUtdPipeline,
    gpu: &VkBackend,
    scene: &mut SceneManager,
    query: AcousticQuery,
) -> Vec<f32> {
    pipeline.set_surface_two_edge_paths(false);
    let raw = pipeline.build_ir_gpu(gpu, scene, query).unwrap();
    let control = pipeline.download_gpu_ir(gpu, &raw).unwrap();
    drop(raw);
    pipeline.set_surface_two_edge_paths(true);
    let raw = pipeline.build_ir_gpu(gpu, scene, query).unwrap();
    let candidate = pipeline.download_gpu_ir(gpu, &raw).unwrap();
    candidate
        .samples
        .iter()
        .zip(&control.samples)
        .map(|(a, b)| a.left - b.left)
        .collect()
}

#[test]
#[ignore = "requires Vulkan ray tracing and glslangValidator"]
fn surface_two_edges_match_planar_pressure_and_absorption() {
    let _guard = super::tests::GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let gpu = VkBackend::new().unwrap();
    let config = EngineConfig {
        sound: SoundConfig {
            output_channels: OutputChannels::Mono,
            sample_rate: 44100,
            ir_num_samples: 4096,
        },
        acoustics: AcousticsConfig {
            listener_ray_count: 1_048_576,
            max_bounces: 3,
            maximum_path_length: 15.0,
            minimum_throughput: 0.0,
            sampling_seed: 7,
        },
        auralization: AuralizationConfig {
            block_size: 1024,
        },
        cache: IrCacheConfig::default(),
        binaural: BinauralConfig::default(),
    };
    let query = AcousticQuery {
        query_id: 0,
        source_position: vec3(-3.0, 0.2, 0.0),
        listener_position: vec3(1.5, 0.4, 0.0),
        listener_forward: -Vec3::X,
        listener_up: Vec3::Y,
        source_amplitude: 1.0,
        source_radiation: Default::default(),
    };
    let mut pipeline =
        ExperimentalUtdPipeline::with_band_orders(&gpu, config, 0.04, 20000.0, 16, 2).unwrap();
    let mut scene = SceneManager::new();
    scene.load(fixture(0.0)).unwrap();
    let forward = added(&mut pipeline, &gpu, &mut scene, query);
    assert!(forward.iter().all(|v| v.is_finite()));
    let norm: f32 = forward.iter().map(|v| v * v).sum();
    assert!(norm > 1e-8, "mixed two-edge paths were missed");
    let fft = realfft::RealFftPlanner::<f32>::new().plan_fft_forward(4096);
    let mut spectrum = fft.make_output_vec();
    fft.process(&mut forward.clone(), &mut spectrum).unwrap();
    // Sum of the two physical floor orders, independently evaluated by SciPy
    // in floor_two_edge_reference.py::top_path for this isolated fixture.
    let mut worst_error = 0.0_f32;
    for (bin, expected) in [
        (23, [0.018310510, 0.100640530]),
        (93, [-0.065200695, 0.017063622]),
        (163, [-0.006623111, -0.045076562]),
    ] {
        let actual = spectrum[bin];
        let error = ((actual.re - expected[0]).powi(2) + (actual.im - expected[1]).powi(2)).sqrt()
            / (expected[0] * expected[0] + expected[1] * expected[1]).sqrt();
        eprintln!(
            "surface-two-edge bin={bin}, actual={actual:?}, expected={expected:?}, relative={error}"
        );
        worst_error = worst_error.max(error);
    }
    assert!(
        worst_error < 0.12,
        "mixed two-edge pressure differs from planar reference"
    );
    scene.load(fixture(0.75)).unwrap();
    let attenuated = added(&mut pipeline, &gpu, &mut scene, query);
    let scaling = (attenuated
        .iter()
        .zip(&forward)
        .map(|(a, b)| (a - 0.5 * b).powi(2))
        .sum::<f32>()
        / norm)
        .sqrt();
    eprintln!("surface-two-edge floor pressure scaling error={scaling}");
    assert!(
        scaling < 1e-4,
        "surface pressure was not carried into diffraction"
    );
    scene.load(fixture(1.0)).unwrap();
    let absorbed = added(&mut pipeline, &gpu, &mut scene, query);
    assert!(
        absorbed.iter().map(|v| v * v).sum::<f32>() < norm * 1e-8,
        "absorbing floor produced mixed pressure"
    );
    scene.load(fixture(0.0)).unwrap();
    let reversed = added(
        &mut pipeline,
        &gpu,
        &mut scene,
        AcousticQuery {
            source_position: query.listener_position,
            listener_position: query.source_position,
            ..query
        },
    );
    let reciprocity = (reversed
        .iter()
        .zip(&forward)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / norm)
        .sqrt();
    eprintln!("surface-two-edge reciprocity relative L2={reciprocity}");
    assert!(
        reciprocity < 0.15,
        "paired reflection orders lost reciprocity"
    );
    let mut alternate_config = config;
    alternate_config.acoustics.sampling_seed = 8;
    let mut alternate =
        ExperimentalUtdPipeline::with_band_orders(&gpu, alternate_config, 0.04, 20000.0, 16, 2)
            .unwrap();
    let alternate_wave = added(&mut alternate, &gpu, &mut scene, query);
    let seed_difference = (alternate_wave
        .iter()
        .zip(&forward)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / norm)
        .sqrt();
    eprintln!("surface-two-edge independent-seed relative difference={seed_difference}");
    assert!(
        seed_difference > 1e-4,
        "different seeds repeated the same mixed-path samples"
    );
    let profile = sound_engine::acoustics::SourceDirectivity::new(
        1,
        19,
        vec![20.0, 20000.0],
        (0..19)
            .flat_map(|t| [[1.0 - 0.5 * t as f32 / 18.0, 0.1 * t as f32 / 18.0]; 2])
            .collect(),
    )
    .unwrap();
    let profile_id = pipeline
        .register_source_directivity(&gpu, &profile)
        .unwrap();
    let directional = added(
        &mut pipeline,
        &gpu,
        &mut scene,
        AcousticQuery {
            source_radiation: sound_engine::acoustics::SourceRadiation {
                profile: profile_id,
                ..Default::default()
            },
            ..query
        },
    );
    fft.process(&mut directional.clone(), &mut spectrum)
        .unwrap();
    let gain = |angle: f32| {
        realfft::num_complex::Complex32::new(
            1.0 - 0.5 * angle / std::f32::consts::PI,
            0.1 * angle / std::f32::consts::PI,
        )
    };
    let floor_gain = gain(0.2_f32.atan2(0.44));
    let edge_gain = gain(0.8_f32.atan2(2.64));
    for (bin, rdd, ddr) in [
        (23, [-0.047629070, 0.077419649], [0.065939580, 0.023220881]),
        (
            93,
            [-0.041528259, -0.000581467],
            [-0.023672435, 0.017645089],
        ),
        (
            163,
            [-0.009103822, -0.025925930],
            [0.002480711, -0.019150632],
        ),
    ] {
        let expected = floor_gain * realfft::num_complex::Complex32::new(rdd[0], rdd[1])
            + edge_gain * realfft::num_complex::Complex32::new(ddr[0], ddr[1]);
        let error = (spectrum[bin] - expected).norm() / expected.norm();
        eprintln!("surface-two-edge source departure bin={bin}, relative={error}");
        assert!(
            error < 0.12,
            "mixed path used the wrong source departure or phase"
        );
    }

    // A positive proposal weight must not imply visibility. This proxy lies
    // entirely inside the absorbing box; it competes in the discrete proposal
    // but cannot add physical pressure. The external planar oracle is unchanged.
    let mut occluded = fixture(0.0);
    occluded.diffraction_edges.push(edge(
        3,
        vec3(0.0, 0.5, -10.0),
        vec3(0.0, 0.5, 10.0),
        Vec3::Y,
        std::f32::consts::TAU,
        0.1,
    ));
    scene.load(occluded).unwrap();
    let competing = added(&mut pipeline, &gpu, &mut scene, query);
    fft.process(&mut competing.clone(), &mut spectrum).unwrap();
    for (bin, expected) in [
        (23, [0.018310510, 0.100640530]),
        (93, [-0.065200695, 0.017063622]),
        (163, [-0.006623111, -0.045076562]),
    ] {
        let expected = realfft::num_complex::Complex32::new(expected[0], expected[1]);
        let error = (spectrum[bin] - expected).norm() / expected.norm();
        eprintln!("surface-two-edge occluded candidate bin={bin}, relative={error}");
        assert!(
            error < 0.12,
            "edge-selection PDF or physical visibility changed pressure"
        );
    }
}
