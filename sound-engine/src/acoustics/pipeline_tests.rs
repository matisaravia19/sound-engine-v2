//! CPU reference checks for acoustic transport and query validation.

use super::*;

fn power_cosine_pdf(cosine: f32, exponent: f32) -> f32 {
    if !cosine.is_finite() || !exponent.is_finite() || cosine <= 0.0 || exponent < 0.0 {
        return 0.0;
    }
    (exponent + 1.0) * cosine.powf(exponent) / (2.0 * std::f32::consts::PI)
}

fn pressure_branch_probabilities(energies: &[f32]) -> Vec<f32> {
    let weights = energies
        .iter()
        .map(|energy| energy.max(0.0).sqrt())
        .collect::<Vec<_>>();
    let sum = weights.iter().sum::<f32>();
    if sum <= 0.0 || !sum.is_finite() {
        return vec![0.0; energies.len()];
    }
    weights.into_iter().map(|weight| weight / sum).collect()
}

fn deposit_fractional(samples: &mut [f32], sample_position: f32, amplitude: f32) {
    if !sample_position.is_finite()
        || !amplitude.is_finite()
        || sample_position <= -1.0
        || sample_position >= samples.len() as f32
    {
        return;
    }
    let base = sample_position.floor() as isize;
    let fraction = sample_position - base as f32;
    if base >= 0
        && let Some(sample) = samples.get_mut(base as usize)
    {
        *sample += amplitude * (1.0 - fraction);
    }
    let next = base + 1;
    if next >= 0
        && let Some(sample) = samples.get_mut(next as usize)
    {
        *sample += amplitude * fraction;
    }
}

fn direct_lanczos_weights(fraction: f32) -> [f32; 16] {
    const RADIUS: i32 = 8;
    let mut weights = [0.0; 16];
    for (index, offset) in (-RADIUS + 1..=RADIUS).enumerate() {
        let distance = offset as f32 - fraction;
        let sinc = if distance.abs() < 1.0e-6 {
            1.0
        } else {
            (std::f32::consts::PI * distance).sin() / (std::f32::consts::PI * distance)
        };
        let window_argument = distance / RADIUS as f32;
        let window = if distance.abs() < 1.0e-6 {
            1.0
        } else {
            (std::f32::consts::PI * window_argument).sin()
                / (std::f32::consts::PI * window_argument)
        };
        weights[index] = sinc * window;
    }
    let sum = weights.iter().sum::<f32>();
    for weight in &mut weights {
        *weight /= sum;
    }
    weights
}

fn specular_exponent_from_roughness(roughness: f32) -> f32 {
    let alpha = roughness.clamp(0.0, 1.0).max(0.0001);
    (2.0 / (alpha * alpha) - 2.0).clamp(1.0, 4096.0)
}

fn folded_power_cosine_pdf(direction: Vec3, axis: Vec3, normal: Vec3, exponent: f32) -> f32 {
    power_cosine_pdf(direction.dot(axis).max(0.0), exponent)
        + power_cosine_pdf(direction.reflect(normal).dot(axis).max(0.0), exponent)
}

fn diffraction_radial_kernel(distance_to_edge: f32, radius: f32) -> f32 {
    if distance_to_edge < 0.0 {
        return 0.0;
    }
    2.0 * (1.0 - distance_to_edge / radius).clamp(0.0, 1.0) / radius
}

fn analytic_binaural_estimate(
    lateral: f32,
    head_radius_meters: f32,
    max_ild_db: f32,
    sample_rate: u32,
) -> ([f32; 2], [f32; 2]) {
    let lateral = lateral.clamp(-1.0, 1.0);
    let absolute_lateral = lateral.abs();
    let itd_seconds = lateral.signum() * head_radius_meters / SPEED_OF_SOUND_METERS_PER_SECOND
        * (absolute_lateral.asin() + absolute_lateral);
    let half_itd_samples = 0.5 * itd_seconds * sample_rate as f32;
    let far_to_near = 10.0_f32.powf(-max_ild_db * absolute_lateral / 20.0);
    let near_factor = (1.0 + far_to_near * far_to_near).sqrt().recip();
    let far_factor = far_to_near * near_factor;
    let gains = if lateral < 0.0 {
        [near_factor, far_factor]
    } else if lateral > 0.0 {
        [far_factor, near_factor]
    } else {
        [near_factor, near_factor]
    };
    ([half_itd_samples, -half_itd_samples], gains)
}

fn test_config() -> EngineConfig {
    EngineConfig {
        sound: crate::core::config::SoundConfig {
            output_channels: OutputChannels::Binaural,
            sample_rate: 48_000,
            ir_num_samples: 16_384,
        },
        acoustics: crate::core::config::AcousticsConfig {
            listener_ray_count: 1,
            max_bounces: 0,
            maximum_path_length: 343.0,
            minimum_throughput: 0.0,
            sampling_seed: 0,
        },
        cache: crate::core::config::IrCacheConfig::default(),
        binaural: crate::core::config::BinauralConfig::default(),
        auralization: crate::core::config::AuralizationConfig {
            block_size: 1_024,
        },
    }
}

fn test_query() -> AcousticQuery {
    AcousticQuery {
        query_id: 1,
        source_position: Vec3::Z,
        listener_position: Vec3::ZERO,
        listener_forward: -Vec3::Z,
        listener_up: Vec3::Y,
        source_amplitude: 1.0,
    }
}

#[test]
#[ignore = "requires Vulkan ray tracing; run hardware tests serially"]
fn audit_repeated_queries_reuse_tlas_and_snapshot_retains_device() {
    let gpu = VkBackend::new().unwrap();
    let mut pipeline = AcousticPipeline::new(&gpu, test_config()).unwrap();
    let mut scene = SceneManager::new();
    let device = Arc::downgrade(&gpu.device_context());
    let mut previous_tlas = None;
    for _ in 0..4 {
        let ir = pipeline.build_ir(&gpu, &mut scene, test_query()).unwrap();
        assert!(
            ir.samples
                .iter()
                .all(|sample| sample.left.is_finite() && sample.right.is_finite())
        );
        assert!(ir.energy > 0.0);
        assert_eq!(gpu.rt().resource_counts(), (1, 1));
        let id = pipeline.query_tlas.as_ref().unwrap().tlas.id;
        assert!(previous_tlas.is_none_or(|previous| previous == id));
        previous_tlas = Some(id);
    }
    let snapshot = pipeline
        .build_ir_gpu(&gpu, &mut scene, test_query())
        .unwrap();
    drop(pipeline);
    assert_eq!(gpu.rt().pipeline_count(), 0);
    for _ in 0..2 {
        let temporary = AcousticPipeline::new(&gpu, test_config()).unwrap();
        assert_eq!(gpu.rt().pipeline_count(), 1);
        drop(temporary);
        assert_eq!(gpu.rt().pipeline_count(), 0);
    }
    drop(scene);
    assert_eq!(gpu.rt().resource_counts(), (0, 0));
    drop(gpu);
    assert!(device.upgrade().is_some());
    drop(snapshot);
    assert!(device.upgrade().is_none());
}

#[test]
#[ignore = "requires Vulkan ray tracing; run hardware tests serially"]
fn audit_query_tlas_distinguishes_scene_owners_and_geometry_updates() {
    use crate::scene::*;
    let gpu = VkBackend::new().unwrap();
    let mut config = test_config();
    config.sound.ir_num_samples = 1024;
    config.acoustics.max_bounces = 1;
    config.acoustics.listener_ray_count = 1024;
    let mut pipeline = AcousticPipeline::new(&gpu, config).unwrap();
    let wall = SceneDescription {
        meshes: vec![MeshAsset {
            id: 1,
            vertices: vec![
                Vec3::new(-10.0, -10.0, 0.5),
                Vec3::new(10.0, -10.0, 0.5),
                Vec3::new(0.0, 10.0, 0.5),
            ],
            indices: vec![],
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
        diffraction_edges: vec![],
    };
    let mut obstructed = SceneManager::new();
    obstructed.load(wall.clone()).unwrap();
    let mut open_description = wall;
    open_description.objects[0].transform = Mat4::from_translation(Vec3::X * 100.0);
    let mut open = SceneManager::new();
    open.load(open_description).unwrap();
    assert_eq!(obstructed.version(), open.version());
    for _ in 0..4 {
        assert_eq!(
            pipeline
                .build_ir(&gpu, &mut obstructed, test_query())
                .unwrap()
                .energy,
            0.0
        );
        assert!(
            pipeline
                .build_ir(&gpu, &mut open, test_query())
                .unwrap()
                .energy
                > 0.0
        );
        assert_eq!(gpu.rt().resource_counts(), (3, 3));
    }
    pipeline
        .build_ir(&gpu, &mut obstructed, test_query())
        .unwrap();
    let material_query_tlas = pipeline.query_tlas.as_ref().unwrap().tlas.id;
    obstructed
        .apply_updates(SceneUpdates {
            updates: vec![SceneUpdate::ReplaceMaterial(Material {
                id: 1,
                absorption: 0.0,
                scattering: 0.0,
                roughness: 0.0,
            })],
        })
        .unwrap();
    let reflected = pipeline
        .build_ir(&gpu, &mut obstructed, test_query())
        .unwrap();
    // A reflecting surface still blocks a source on the opposite side.
    assert_eq!(reflected.energy, 0.0);
    assert_eq!(
        pipeline.query_tlas.as_ref().unwrap().tlas.id,
        material_query_tlas
    );
    obstructed
        .apply_updates(SceneUpdates {
            updates: vec![SceneUpdate::SetActive {
                id: 1,
                active: false,
            }],
        })
        .unwrap();
    assert!(
        pipeline
            .build_ir(&gpu, &mut obstructed, test_query())
            .unwrap()
            .energy
            > 0.0
    );
    drop(open);
    drop(obstructed);
    // Edge-only scenes have no surface TLAS, so their BLAS identity must
    // independently invalidate the cached procedural query structure.
    let mut edges = SceneManager::new();
    let mut edge = DiffractionEdge {
        id: 1,
        start: Vec3::new(0.0, -1.0, 0.5),
        end: Vec3::new(0.0, 1.0, 0.5),
        bisector_dir: Vec3::X,
        edge_angle_radians: std::f32::consts::TAU,
        diffraction_radius: 0.1,
    };
    edges
        .load(SceneDescription {
            meshes: vec![],
            materials: vec![],
            objects: vec![],
            diffraction_edges: vec![edge.clone()],
        })
        .unwrap();
    let mut previous = None;
    for step in 0..4 {
        edge.diffraction_radius = 0.1 + step as f32 * 0.1;
        edges
            .apply_updates(SceneUpdates {
                updates: vec![SceneUpdate::ReplaceDiffractionEdge(edge.clone())],
            })
            .unwrap();
        let ir = pipeline.build_ir(&gpu, &mut edges, test_query()).unwrap();
        assert!(
            ir.samples
                .iter()
                .all(|sample| sample.left.is_finite() && sample.right.is_finite())
        );
        let id = pipeline.query_tlas.as_ref().unwrap().tlas.id;
        assert!(previous.is_none_or(|previous| previous != id));
        previous = Some(id);
        assert_eq!(gpu.rt().resource_counts(), (2, 1));
    }
    drop(edges);
    drop(pipeline);
    assert_eq!(gpu.rt().resource_counts(), (0, 0));
}

#[test]
fn power_cosine_pdf_integrates_to_one() {
    let exponent = 32.0;
    let steps = 100_000;
    let integral = (0..steps)
        .map(|index| {
            let cosine = (index as f32 + 0.5) / steps as f32;
            power_cosine_pdf(cosine, exponent) * 2.0 * std::f32::consts::PI / steps as f32
        })
        .sum::<f32>();
    assert!((integral - 1.0).abs() < 0.001);
}

#[test]
fn folded_surface_lobe_preserves_probability_at_grazing_angles() {
    let normal = Vec3::Y;
    let axis = Vec3::new(1.0, 0.05, 0.0).normalize();
    let exponent = 32.0;
    let polar_steps = 512;
    let azimuth_steps = 1_024;
    let mut integral = 0.0;
    for polar_index in 0..polar_steps {
        let cosine = (polar_index as f32 + 0.5) / polar_steps as f32;
        let sine = (1.0 - cosine * cosine).sqrt();
        for azimuth_index in 0..azimuth_steps {
            let azimuth =
                std::f32::consts::TAU * (azimuth_index as f32 + 0.5) / azimuth_steps as f32;
            let direction = Vec3::new(sine * azimuth.cos(), cosine, sine * azimuth.sin());
            integral += folded_power_cosine_pdf(direction, axis, normal, exponent);
        }
    }
    integral *= std::f32::consts::TAU / (polar_steps * azimuth_steps) as f32;
    assert!(
        (integral - 1.0).abs() < 0.002,
        "folded lobe integral was {integral}"
    );
}

#[test]
fn diffraction_strip_kernel_is_finite_normalized_and_bounded() {
    for radius in [0.05, 0.2, 1.0, 4.0] {
        let steps = 100_000;
        let width = radius / steps as f32;
        let integral = (0..steps)
            .map(|index| {
                let distance = (index as f32 + 0.5) * width;
                diffraction_radial_kernel(distance, radius) * width
            })
            .sum::<f32>();
        assert!((integral - 1.0).abs() < 1.0e-4);
        assert_eq!(diffraction_radial_kernel(0.0, radius), 2.0 / radius);
        assert_eq!(diffraction_radial_kernel(-0.5 * radius, radius), 0.0);
        assert_eq!(diffraction_radial_kernel(radius, radius), 0.0);
        assert_eq!(diffraction_radial_kernel(2.0 * radius, radius), 0.0);
    }
}

#[test]
fn projected_strip_density_is_radius_independent_for_uniform_launches() {
    // Independent area-to-solid-angle integration. Relocating proxy samples
    // must recover the edge density for both approaching and receding strips.
    for normal_distance in [0.1_f64, 1.0, 3.0] {
        for lateral in [-1.0_f64, 0.0, 1.0] {
            for radius in [0.05_f64, 0.2, 0.9] {
                let edge_distance = normal_distance.hypot(lateral);
                let expected = normal_distance / edge_distance.powi(3);
                let mut measured = 0.0;
                for step in 0..4096 {
                    let r = (step as f64 + 0.5) * radius / 4096.0;
                    let proxy_distance = normal_distance.hypot(lateral + r);
                    let density = normal_distance / proxy_distance.powi(3);
                    let weight = (proxy_distance / edge_distance).powi(3);
                    measured += density * weight * 2.0 * (1.0 - r / radius) / 4096.0;
                }
                assert!((measured / expected - 1.0).abs() < 1e-10);
            }
        }
    }
}

#[test]
fn branch_probabilities_follow_pressure_weights() {
    let probabilities = pressure_branch_probabilities(&[0.81, 0.09]);
    assert!((probabilities[0] - 0.75).abs() < 1.0e-6);
    assert!((probabilities[1] - 0.25).abs() < 1.0e-6);
}

#[test]
fn fractional_deposition_preserves_signed_amplitude() {
    let mut samples = [0.0; 4];
    deposit_fractional(&mut samples, 1.25, -2.0);
    assert_eq!(samples, [0.0, -1.5, -0.5, 0.0]);
}

#[test]
fn fractional_deposition_rejects_invalid_state() {
    let mut samples = [0.0; 2];
    deposit_fractional(&mut samples, f32::NAN, 1.0);
    deposit_fractional(&mut samples, 0.0, f32::INFINITY);
    assert_eq!(samples, [0.0, 0.0]);
}

#[test]
fn direct_lanczos_delay_preserves_pressure_and_bounds_energy_ripple() {
    let mut minimum_energy = f32::INFINITY;
    let mut maximum_energy = f32::NEG_INFINITY;
    for step in 0..=100 {
        let weights = direct_lanczos_weights(step as f32 / 100.0);
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1.0e-5);
        let energy = weights.iter().map(|weight| weight * weight).sum::<f32>();
        minimum_energy = minimum_energy.min(energy);
        maximum_energy = maximum_energy.max(energy);
    }
    let ripple_db = 10.0 * (maximum_energy / minimum_energy).log10();
    assert!(
        ripple_db < 0.4,
        "direct fractional-delay energy ripple was {ripple_db} dB"
    );
}

#[test]
fn specular_exponent_uses_rms_slope_relation() {
    assert_eq!(specular_exponent_from_roughness(0.0), 4096.0);
    assert!((specular_exponent_from_roughness(0.15) - 86.888_89).abs() < 1.0e-4);
    assert_eq!(specular_exponent_from_roughness(1.0), 1.0);
    assert!(specular_exponent_from_roughness(0.1) > specular_exponent_from_roughness(0.2));
}

#[test]
fn half_period_two_tap_ir_cancels_a_sine() {
    let period = 32;
    let half_period = period / 2;
    let input = (0..256)
        .map(|index| (std::f32::consts::TAU * index as f32 / period as f32).sin())
        .collect::<Vec<_>>();
    let mut output = vec![0.0; input.len() + half_period];
    for (index, sample) in input.iter().copied().enumerate() {
        output[index] += sample;
        output[index + half_period] += sample;
    }
    let steady_peak = output[half_period..input.len()]
        .iter()
        .fold(0.0_f32, |peak, value| peak.max(value.abs()));
    assert!(steady_peak < 1.0e-5);
}

#[test]
fn push_constants_match_shader_layout() {
    assert_eq!(size_of::<ListenerTracePushConstants>(), 84);
    assert_eq!(
        std::mem::offset_of!(ListenerTracePushConstants, minimum_throughput),
        72
    );
    assert_eq!(
        std::mem::offset_of!(ListenerTracePushConstants, head_radius_meters),
        76
    );
    assert_eq!(
        std::mem::offset_of!(ListenerTracePushConstants, max_ild_db),
        80
    );
}

#[test]
fn centered_binaural_arrivals_have_equal_levels_and_no_itd() {
    let right = Vec3::X;
    for direction in [-Vec3::Z, Vec3::Z, Vec3::Y, Vec3::ZERO] {
        let lateral = direction.normalize_or_zero().dot(right);
        let (offsets, gains) = analytic_binaural_estimate(lateral, 0.0875, 12.0, 48_000);
        assert_eq!(offsets, [0.0, 0.0]);
        assert!((gains[0] - gains[1]).abs() < 1.0e-7);
    }
}

#[test]
fn left_and_right_binaural_arrivals_are_mirrored() {
    let (left_offsets, left_gains) = analytic_binaural_estimate(-0.75, 0.0875, 12.0, 48_000);
    let (right_offsets, right_gains) = analytic_binaural_estimate(0.75, 0.0875, 12.0, 48_000);
    assert_eq!(left_offsets, [right_offsets[1], right_offsets[0]]);
    assert_eq!(left_gains, [right_gains[1], right_gains[0]]);
}

#[test]
fn maximum_binaural_itd_matches_spherical_head_formula() {
    let radius = 0.0875;
    let sample_rate = 48_000;
    let (offsets, _) = analytic_binaural_estimate(1.0, radius, 12.0, sample_rate);
    let expected = radius / SPEED_OF_SOUND_METERS_PER_SECOND
        * (std::f32::consts::FRAC_PI_2 + 1.0)
        * sample_rate as f32;
    assert!(((offsets[0] - offsets[1]) - expected).abs() < 1.0e-5);
}

#[test]
fn binaural_ild_gains_preserve_unit_energy() {
    for lateral in [-1.0, -0.5, 0.0, 0.5, 1.0] {
        let (_, gains) = analytic_binaural_estimate(lateral, 0.0875, 12.0, 48_000);
        assert!((gains[0] * gains[0] + gains[1] * gains[1] - 1.0).abs() < 1.0e-6);
    }
}

#[test]
fn binaural_fractional_deposition_preserves_each_ear_pressure() {
    let (offsets, gains) = analytic_binaural_estimate(0.6, 0.0875, 12.0, 48_000);
    let center = 10.25;
    let amplitude = -2.0;
    let mut left = [0.0; 32];
    let mut right = [0.0; 32];
    deposit_fractional(&mut left, center + offsets[0], amplitude * gains[0]);
    deposit_fractional(&mut right, center + offsets[1], amplitude * gains[1]);
    assert!((left.iter().sum::<f32>() - amplitude * gains[0]).abs() < 1.0e-6);
    assert!((right.iter().sum::<f32>() - amplitude * gains[1]).abs() < 1.0e-6);
}

#[test]
fn fractional_deposition_clips_only_the_out_of_bounds_portion() {
    let mut samples = [0.0; 2];
    deposit_fractional(&mut samples, -0.25, -2.0);
    assert_eq!(samples, [-1.5, 0.0]);

    samples.fill(0.0);
    deposit_fractional(&mut samples, 1.25, 2.0);
    assert_eq!(samples, [0.0, 1.5]);
}

#[test]
fn rejects_invalid_binaural_configuration() {
    let mut config = test_config();
    config.binaural.head_radius_meters = 0.0;
    assert!(validate_config(config).is_err());

    config = test_config();
    config.binaural.head_radius_meters = f32::NAN;
    assert!(validate_config(config).is_err());

    config = test_config();
    config.binaural.max_ild_db = -1.0;
    assert!(validate_config(config).is_err());

    config = test_config();
    config.binaural.max_ild_db = f32::INFINITY;
    assert!(validate_config(config).is_err());
}

#[test]
fn rejects_invalid_acoustic_query_orientation() {
    let mut query = test_query();
    query.listener_forward = Vec3::ZERO;
    assert!(validate_query(query).is_err());

    query = test_query();
    query.listener_up = query.listener_forward;
    assert!(validate_query(query).is_err());

    query = test_query();
    query.listener_up = Vec3::splat(f32::NAN);
    assert!(validate_query(query).is_err());
}

#[test]
fn visibility_hit_group_follows_surface_continuation() {
    assert_eq!(SURFACE_VISIBILITY_HIT_GROUP_OFFSET, 1);
}
