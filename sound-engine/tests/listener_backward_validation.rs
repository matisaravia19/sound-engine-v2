use glam::{Mat4, Vec3, vec3};
use sound_engine::acoustics::{AcousticPipeline, AcousticQuery, IrSnapshot};
use sound_engine::core::config::{
    AcousticsConfig, AuralizationConfig, EngineConfig, IrCacheConfig, OutputChannels, SoundConfig,
};
use sound_engine::core::error::{SoundError, SoundResult};
use sound_engine::gpu::backend::VkBackend;
use sound_engine::scene::{
    DiffractionEdge, Material, MeshAsset, SceneDescription, SceneManager, SceneObject,
};

const SAMPLE_RATE: u32 = 48_000;
const IR_SAMPLES: u32 = SAMPLE_RATE;

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn listener_transport_matches_reference() -> SoundResult<()> {
    let gpu = VkBackend::new()?;
    validate_direct_and_ray_count_invariance(&gpu)?;
    validate_occlusion(&gpu)?;
    validate_reflection(&gpu)?;
    validate_diffraction_convergence_and_radius(&gpu)?;
    println!("listener-backward validation passed");
    Ok(())
}

fn validate_direct_and_ray_count_invariance(gpu: &VkBackend) -> SoundResult<()> {
    let query = AcousticQuery {
        query_id: 1,
        source_position: Vec3::ZERO,
        listener_position: Vec3::X,
        listener_forward: -Vec3::Z,
        listener_up: Vec3::Y,
        source_amplitude: 1.0,
    };
    let low = trace(gpu, empty_scene(), config(64, 0), query)?;
    let high = trace(gpu, empty_scene(), config(16_384, 0), query)?;
    let low_sum = signed_channel_sum(&low);
    let high_sum = signed_channel_sum(&high);
    require(
        (low_sum - 1.0).abs() < 1.0e-5,
        "direct path amplitude must equal 1 / distance",
    )?;
    require(
        (low_sum - high_sum).abs() < 1.0e-6,
        "direct path must not scale with listener ray count",
    )?;
    require(
        nonzero_samples(&low) <= 16,
        "windowed-sinc direct deposition must touch at most sixteen samples",
    )?;
    let expected_position = SAMPLE_RATE as f32 / 343.0;
    require(
        (signed_arrival_centroid(&low) - expected_position).abs() < 0.01,
        "direct path centroid must match its fractional arrival time",
    )
}

fn validate_occlusion(gpu: &VkBackend) -> SoundResult<()> {
    let query = AcousticQuery {
        query_id: 2,
        source_position: vec3(-1.0, 0.0, 0.0),
        listener_position: vec3(1.0, 0.0, 0.0),
        listener_forward: -Vec3::Z,
        listener_up: Vec3::Y,
        source_amplitude: 1.0,
    };
    let ir = trace(gpu, occluding_wall_scene(), config(4_096, 0), query)?;
    require(
        channel_sum(&ir) == 0.0,
        "occluding wall must remove the direct path",
    )
}

fn validate_reflection(gpu: &VkBackend) -> SoundResult<()> {
    // Unequal wall distances expose source-connection footprint aliasing.
    // Swapping endpoints checks the same physical reflection independently.
    let a = vec3(1.1, -0.3, 0.2);
    let b = vec3(2.7, 0.8, -0.4);
    for (source, listener) in [(a, b), (b, a)] {
        for seed in [7, 8, 9] {
            let mut scene = reflection_scene();
            scene.objects[0].transform = Mat4::from_translation(-2.0 * Vec3::X);
            scene.materials[0].roughness = 0.0;
            // The wall becomes GPU material 1 after sorting, exercising the
            // packed material array stride rather than only its first record.
            scene.materials.push(Material {
                id: 0,
                absorption: 1.0,
                scattering: 0.0,
                roughness: 0.0,
            });
            let mut settings = config(65_536, 1);
            settings.sound.ir_num_samples = 2_048;
            settings.acoustics.maximum_path_length = 10.0;
            settings.acoustics.sampling_seed = seed;
            let query = AcousticQuery {
                query_id: 3,
                source_position: source,
                listener_position: listener,
                listener_forward: -Vec3::Z,
                listener_up: Vec3::Y,
                source_amplitude: 1.0,
            };
            let ir = trace(gpu, scene, settings, query)?;
            let reflected_distance = vec3(-source.x, source.y, source.z).distance(listener);
            let expected_pressure = 0.75_f32.sqrt() / reflected_distance;
            let measured_pressure = signed_channel_sum(&ir) - source.distance(listener).recip();
            require_relative_close(
                measured_pressure,
                expected_pressure,
                0.02,
                "smooth reflection pressure must agree with the planar reference within 2%",
            )?;
            let cutoff = ((source.distance(listener) + reflected_distance) * 0.5 / 343.0
                * SAMPLE_RATE as f32) as usize;
            let mut mass = 0.0;
            let mut moment = 0.0;
            for (index, sample) in ir.samples.iter().enumerate().skip(cutoff) {
                mass += sample.left;
                moment += index as f32 * sample.left;
            }
            let centroid = moment / mass;
            let expected_sample = reflected_distance / 343.0 * SAMPLE_RATE as f32;
            require(
                (centroid - expected_sample).abs() < 0.75,
                "smooth reflection centroid must be within 0.75 samples of the planar reference",
            )?;
            println!(
                "reflection seed={seed} source_x={} pressure={measured_pressure:.8} expected={expected_pressure:.8} delay_error_samples={:.4}",
                source.x,
                centroid - expected_sample
            );
        }
    }
    Ok(())
}

fn validate_diffraction_convergence_and_radius(gpu: &VkBackend) -> SoundResult<()> {
    let query = AcousticQuery {
        query_id: 4,
        source_position: vec3(1.0, 0.0, -1.0),
        listener_position: vec3(-1.0, 0.0, 0.5),
        listener_forward: -Vec3::Z,
        listener_up: Vec3::Y,
        source_amplitude: 1.0,
    };
    let baseline = trace(gpu, diffraction_scene(None), config(65_536, 1), query)?;
    require(
        channel_sum(&baseline) == 0.0,
        "diffraction test direct path must be occluded",
    )?;

    let low = trace(gpu, diffraction_scene(Some(0.45)), config(65_536, 1), query)?;
    let high = trace(
        gpu,
        diffraction_scene(Some(0.45)),
        config(262_144, 1),
        query,
    )?;
    require(
        nonzero_samples(&high) > 2 && channel_sum(&high) > 0.0,
        "authored edge must produce a distribution of diffraction delays",
    )?;
    require_relative_close(
        channel_sum(&low),
        channel_sum(&high),
        0.35,
        "diffraction level must converge without ray-count scaling",
    )?;

    let narrow = trace(gpu, diffraction_scene(Some(0.3)), config(262_144, 1), query)?;
    let wide = trace(gpu, diffraction_scene(Some(0.6)), config(262_144, 1), query)?;
    require_relative_close(
        channel_sum(&narrow),
        channel_sum(&wide),
        0.35,
        "normalized diffraction level must remain stable across proxy radii",
    )
}

fn trace(
    gpu: &VkBackend,
    description: SceneDescription,
    config: EngineConfig,
    query: AcousticQuery,
) -> SoundResult<IrSnapshot> {
    let mut scene = SceneManager::new();
    scene.load(description)?;
    let mut pipeline = AcousticPipeline::new(gpu, config)?;
    pipeline.build_ir(gpu, &mut scene, query)
}

fn config(listener_ray_count: u32, max_bounces: u32) -> EngineConfig {
    EngineConfig {
        sound: SoundConfig {
            output_channels: OutputChannels::Mono,
            sample_rate: SAMPLE_RATE,
            ir_num_samples: IR_SAMPLES,
        },
        acoustics: AcousticsConfig {
            listener_ray_count,
            max_bounces,
            maximum_path_length: 343.0,
            minimum_throughput: 1.0e-6,
            sampling_seed: 7,
        },
        cache: IrCacheConfig::default(),
        binaural: Default::default(),
        auralization: AuralizationConfig {
            block_size: 1024,
        },
    }
}

fn empty_scene() -> SceneDescription {
    SceneDescription::default()
}

fn occluding_wall_scene() -> SceneDescription {
    wall_scene(
        vec![
            vec3(0.0, -2.0, -2.0),
            vec3(0.0, 2.0, -2.0),
            vec3(0.0, 2.0, 2.0),
            vec3(0.0, -2.0, -2.0),
            vec3(0.0, 2.0, 2.0),
            vec3(0.0, -2.0, 2.0),
        ],
        1.0,
        0.0,
    )
}

fn reflection_scene() -> SceneDescription {
    let extent = 20.0;
    wall_scene(
        vec![
            vec3(2.0, -extent, -extent),
            vec3(2.0, extent, -extent),
            vec3(2.0, extent, extent),
            vec3(2.0, -extent, -extent),
            vec3(2.0, extent, extent),
            vec3(2.0, -extent, extent),
        ],
        0.25,
        0.02,
    )
}

fn diffraction_scene(radius: Option<f32>) -> SceneDescription {
    let mut scene = wall_scene(
        vec![
            vec3(0.0, -2.0, -3.0),
            vec3(0.0, 2.0, -3.0),
            vec3(0.0, 2.0, 0.0),
            vec3(0.0, -2.0, -3.0),
            vec3(0.0, 2.0, 0.0),
            vec3(0.0, -2.0, 0.0),
        ],
        1.0,
        0.0,
    );
    if let Some(radius) = radius {
        scene.diffraction_edges.push(DiffractionEdge {
            id: 1,
            start: vec3(0.0, -2.0, 0.0),
            end: vec3(0.0, 2.0, 0.0),
            // The mesh is a zero-thickness screen extending toward -Z.
            // Its exterior air angle is 360 degrees, opening toward +Z.
            bisector_dir: Vec3::Z,
            edge_angle_radians: std::f32::consts::TAU,
            diffraction_radius: radius,
        });
    }
    scene
}

fn wall_scene(vertices: Vec<Vec3>, absorption: f32, roughness: f32) -> SceneDescription {
    SceneDescription {
        meshes: vec![MeshAsset {
            id: 1,
            vertices,
            indices: Vec::new(),
            opaque: true,
        }],
        materials: vec![Material {
            id: 1,
            absorption,
            scattering: 0.0,
            roughness,
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

fn channel_sum(ir: &IrSnapshot) -> f32 {
    ir.samples.iter().map(|sample| sample.left.abs()).sum()
}

fn signed_channel_sum(ir: &IrSnapshot) -> f32 {
    ir.samples.iter().map(|sample| sample.left).sum()
}

fn signed_arrival_centroid(ir: &IrSnapshot) -> f32 {
    let pressure = signed_channel_sum(ir);
    ir.samples
        .iter()
        .enumerate()
        .map(|(index, sample)| index as f32 * sample.left)
        .sum::<f32>()
        / pressure
}

fn nonzero_samples(ir: &IrSnapshot) -> usize {
    ir.samples
        .iter()
        .filter(|sample| sample.left != 0.0)
        .count()
}

fn require(condition: bool, message: &str) -> SoundResult<()> {
    if condition {
        Ok(())
    } else {
        Err(SoundError::invalid_state(message))
    }
}

fn require_relative_close(a: f32, b: f32, tolerance: f32, message: &str) -> SoundResult<()> {
    let scale = a.abs().max(b.abs()).max(1.0e-9);
    require((a - b).abs() / scale <= tolerance, message)
}

#[cfg(test)]
mod reflection_tests {
    use super::*;

    #[test]
    fn diffraction_fixture_edge_matches_the_screen_mesh() {
        let scene = diffraction_scene(Some(0.45));
        let edge = &scene.diffraction_edges[0];
        let vertices = &scene.meshes[0].vertices;
        let wall_center = vertices.iter().copied().sum::<Vec3>() / vertices.len() as f32;
        let edge_center = (edge.start + edge.end) * 0.5;
        assert!(
            (edge.edge_angle_radians - std::f32::consts::TAU).abs() < 1.0e-6,
            "a single thin screen has a 360-degree exterior air wedge"
        );
        assert!(
            (wall_center - edge_center)
                .normalize()
                .dot(edge.bisector_dir)
                < -0.999,
            "the exterior bisector must point away from the physical screen"
        );
    }

    #[test]
    #[ignore = "requires Vulkan ray tracing; run hardware tests serially"]
    fn smooth_reflection_pressure_and_reciprocity() -> SoundResult<()> {
        validate_reflection(&VkBackend::new()?)
    }

    #[test]
    #[ignore = "requires Vulkan ray tracing; run hardware tests serially"]
    fn short_direct_smoke() -> SoundResult<()> {
        let gpu = VkBackend::new()?;
        let mut settings = config(64, 0);
        settings.sound.ir_num_samples = 2_048;
        settings.acoustics.maximum_path_length = 10.0;
        let query = AcousticQuery {
            query_id: 0,
            source_position: Vec3::ZERO,
            listener_position: Vec3::X,
            listener_forward: -Vec3::Z,
            listener_up: Vec3::Y,
            source_amplitude: 1.0,
        };
        let ir = trace(&gpu, empty_scene(), settings, query)?;
        require(
            (signed_channel_sum(&ir) - 1.0).abs() < 1.0e-5,
            "short direct IR must preserve 1/d pressure",
        )?;
        require(
            (signed_arrival_centroid(&ir) - SAMPLE_RATE as f32 / 343.0).abs() < 0.01,
            "short direct IR must preserve fractional arrival time",
        )
    }

    #[test]
    #[ignore = "requires Vulkan ray tracing; run hardware tests serially"]
    fn proxy_preserves_surface_paths_after_a_reflection() -> SoundResult<()> {
        let gpu = VkBackend::new()?;
        let mut vertices = Vec::new();
        for (x, top) in [(0.0, 20.0), (6.0, 20.0), (3.0, 0.0)] {
            vertices.extend([
                vec3(x, -20.0, -20.0),
                vec3(x, top, -20.0),
                vec3(x, top, 20.0),
                vec3(x, -20.0, -20.0),
                vec3(x, top, 20.0),
                vec3(x, -20.0, 20.0),
            ]);
        }
        // The narrow two-reflection lobe and proxy roulette need a large launch
        // budget to resolve the late pressure within the 5% tolerance below.
        let mut scene = wall_scene(vertices, 0.36, 0.15);
        let mut settings = config(4_194_304, 2);
        settings.sound.ir_num_samples = 2_048;
        settings.acoustics.maximum_path_length = 12.0;
        let query = AcousticQuery {
            query_id: 0,
            source_position: vec3(4.0, 1.0, 0.0),
            listener_position: vec3(2.0, 1.0, 1.0),
            listener_forward: -Vec3::X,
            listener_up: Vec3::Y,
            source_amplitude: 1.0,
        };
        let reference = trace(&gpu, scene.clone(), settings, query)?;
        scene.diffraction_edges.push(DiffractionEdge {
            id: 1,
            start: vec3(3.0, 0.0, -20.0),
            end: vec3(3.0, 0.0, 20.0),
            bisector_dir: Vec3::Y,
            // A flat (PI) wedge has an empty shadow domain. Its proxy still
            // intercepts rays, isolating surface-path preservation without an
            // artificial pressure multiplier.
            edge_angle_radians: std::f32::consts::PI,
            // Cover every clear two-wall path; a narrow proxy legitimately
            // leaves rough paths above the strip with their original budget.
            diffraction_radius: 40.0,
        });
        let exhausted = trace(&gpu, scene.clone(), settings, query)?;
        settings.acoustics.max_bounces = 3;
        let with_proxy = trace(&gpu, scene, settings, query)?;
        let late_start = (9.0 * SAMPLE_RATE as f32 / 343.0) as usize;
        require(
            reference.samples[late_start..]
                .iter()
                .any(|s| s.left.abs() > 1.0e-5),
            "fixture must contain a two-reflection arrival beyond the proxy",
        )?;
        require(
            exhausted.samples[late_start..]
                .iter()
                .all(|s| s.left.abs() < 1.0e-6),
            "a straight crossing must consume the second interaction before the next wall",
        )?;
        let late_pressure =
            |ir: &IrSnapshot| -> f32 { ir.samples[late_start..].iter().map(|s| s.left).sum() };
        eprintln!(
            "two reflections: reference={}, budget2={}, budget3={}",
            late_pressure(&reference),
            late_pressure(&exhausted),
            late_pressure(&with_proxy)
        );
        // Roulette preserves the mean pressure, not identical samples or RNG.
        require_relative_close(
            late_pressure(&reference),
            late_pressure(&with_proxy),
            0.05,
            "an extra interaction must recover the later reflection within sampling tolerance",
        )
    }

    #[test]
    #[ignore = "requires Vulkan ray tracing; run hardware tests serially"]
    fn transparent_proxy_preserves_unobstructed_direct_and_reflected_pressure() -> SoundResult<()> {
        let gpu = VkBackend::new()?;
        // The screen is below y=0. The direct and x=0 mirror paths lie at y=1.
        let mut scene = wall_scene(
            vec![
                vec3(0.0, -20.0, -20.0),
                vec3(0.0, 20.0, -20.0),
                vec3(0.0, 20.0, 20.0),
                vec3(0.0, -20.0, -20.0),
                vec3(0.0, 20.0, 20.0),
                vec3(0.0, -20.0, 20.0),
                vec3(1.0, -20.0, -20.0),
                vec3(1.0, 0.0, -20.0),
                vec3(1.0, 0.0, 20.0),
                vec3(1.0, -20.0, -20.0),
                vec3(1.0, 0.0, 20.0),
                vec3(1.0, -20.0, 20.0),
            ],
            0.36,
            0.0,
        );
        let query = AcousticQuery {
            query_id: 0,
            source_position: vec3(2.0, 1.0, 0.0),
            listener_position: vec3(2.0, 1.0, 1.0),
            listener_forward: -Vec3::X,
            listener_up: Vec3::Y,
            source_amplitude: 1.0,
        };
        let reflected_distance = 17.0_f32.sqrt();
        let reflected_sample = (reflected_distance * SAMPLE_RATE as f32 / 343.0).round() as usize;
        let direct_sample = (SAMPLE_RATE as f32 / 343.0).round() as usize;
        let pressure = |ir: &IrSnapshot, center: usize| -> [f32; 2] {
            let mut result = [0.0; 2];
            // Include binaural shifts and fractional-delay taps without mixing arrivals.
            for s in &ir.samples[center - 48..=center + 48] {
                result[0] += s.left;
                result[1] += s.right;
            }
            result
        };
        // Binary path selection adds variance to the narrow mirror lobe. Check
        // its mean over a fixed seed set, retaining the 3% pressure tolerance.
        // This is a convergence control, not the real-time benchmark budget.
        let mean_arrivals = |scene: &SceneDescription,
                             mut settings: EngineConfig|
         -> SoundResult<([f32; 2], [f32; 2])> {
            let mut direct = [0.0; 2];
            let mut reflected = [0.0; 2];
            for seed in 7..15 {
                settings.acoustics.sampling_seed = seed;
                let ir = trace(&gpu, scene.clone(), settings, query)?;
                let sample_direct = pressure(&ir, direct_sample);
                let sample_reflected = pressure(&ir, reflected_sample);
                eprintln!(
                    "proxies={} seed={seed} reflected={sample_reflected:?}",
                    scene.diffraction_edges.len()
                );
                for channel in 0..2 {
                    direct[channel] += sample_direct[channel] / 8.0;
                    reflected[channel] += sample_reflected[channel] / 8.0;
                }
            }
            Ok((direct, reflected))
        };
        for channels in [OutputChannels::Mono, OutputChannels::Stereo] {
            scene.diffraction_edges.clear();
            let mut settings = config(8_388_608, 1);
            // Both arrivals fit in 43 ms; no long IR or playback queue is needed.
            settings.sound.ir_num_samples = 2_048;
            settings.acoustics.maximum_path_length = 10.0;
            settings.sound.output_channels = channels;
            let reference = trace(&gpu, scene.clone(), settings, query)?;
            let direct = pressure(&reference, direct_sample);
            let reflected = pressure(&reference, reflected_sample);
            require(
                (direct[0].hypot(direct[1]) - 1.0).abs() < 1e-4,
                "direct pressure must equal 1/d",
            )?;
            require_relative_close(
                reflected[0].hypot(reflected[1]),
                0.8 / reflected_distance,
                0.03,
                "reflection pressure must equal sqrt(1-absorption)/image distance",
            )?;
            scene.diffraction_edges.push(DiffractionEdge {
                id: 1,
                start: vec3(1.0, 0.0, -20.0),
                end: vec3(1.0, 0.0, 20.0),
                bisector_dir: Vec3::Y,
                // A flat proxy isolates straight-path weighting and interaction
                // cost. A real edge can add pressure in the mirror's time window.
                edge_angle_radians: std::f32::consts::PI,
                diffraction_radius: 1.25,
            });
            let exhausted = trace(&gpu, scene.clone(), settings, query)?;
            let lost_reflected = pressure(&exhausted, reflected_sample);
            require(
                lost_reflected[0].hypot(lost_reflected[1]) < 1.0e-6,
                "a proxy must consume the sole interaction before the mirror wall",
            )?;

            settings.acoustics.max_bounces = 2;
            let (next_direct, next_reflected) = mean_arrivals(&scene, settings)?;
            eprintln!(
                "{channels:?}: direct {direct:?} -> {next_direct:?}, reflected {reflected:?} -> {next_reflected:?}"
            );
            for channel in 0..2 {
                require(
                    (direct[channel] - next_direct[channel]).abs() < 1e-5,
                    "edge discovery must preserve unobstructed direct pressure",
                )?;
                require_relative_close(
                    reflected[channel],
                    next_reflected[channel],
                    0.03,
                    "a sampled straight crossing must preserve reflected pressure in expectation",
                )?;
            }

            // This second, flat proxy has no diffraction response. It isolates
            // repeated straight selection, its inverse weight and cost budget.
            scene.diffraction_edges.push(DiffractionEdge {
                id: 2,
                start: vec3(1.5, 0.0, -20.0),
                end: vec3(1.5, 0.0, 20.0),
                bisector_dir: Vec3::Y,
                edge_angle_radians: std::f32::consts::PI,
                diffraction_radius: 1.25,
            });
            let exhausted = trace(&gpu, scene.clone(), settings, query)?;
            let lost_reflected = pressure(&exhausted, reflected_sample);
            require(
                lost_reflected[0].hypot(lost_reflected[1]) < 1.0e-6,
                "two straight proxies must exhaust two interactions before the wall",
            )?;
            settings.acoustics.max_bounces = 3;
            let (_, recovered) = mean_arrivals(&scene, settings)?;
            eprintln!("{channels:?}: two proxies, budget3 reflected {recovered:?}");
            for channel in 0..2 {
                require_relative_close(
                    reflected[channel],
                    recovered[channel],
                    0.03,
                    "both straight choices must compensate their sampling probability",
                )?;
            }
        }
        Ok(())
    }
}
