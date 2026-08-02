use glam::{Mat4, Vec3, vec3};
use sound_engine::acoustics::{AcousticPipeline, AcousticQuery};
use sound_engine::core::config::{
    AcousticsConfig, AuralizationConfig, EngineConfig, IrCacheConfig, OutputChannels, SoundConfig,
};
use sound_engine::core::debug::export_ir;
use sound_engine::core::error::{SoundError, SoundResult};
use sound_engine::gpu::backend::VkBackend;
use sound_engine::scene::{DiffractionEdge, Material, MeshAsset, SceneDescription, SceneManager, SceneObject};
use std::f64::consts::PI;

const BASELINE_IR_EXPORT_PATH: &str = "target/gpu_ir_diffraction_baseline_ir.csv";
const DIFFRACTION_IR_EXPORT_PATH: &str = "target/gpu_ir_diffraction_edge_ir.csv";
const SAMPLE_RATE: u32 = 48_000;
const IR_SAMPLES: u32 = SAMPLE_RATE;
const REFERENCE_SEGMENTS: usize = 131_072;
const TEST_FREQUENCIES_HZ: &[f64] = &[125.0, 250.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0];

fn main() -> SoundResult<()> {
    let gpu = VkBackend::new()?;
    let source_position = vec3(1.0, 0.0, -1.0);
    let listener_position = vec3(-1.0, 0.0, 0.5);
    let edge = diffraction_edge(0.45, 0.65);

    let baseline = trace_scene(
        &gpu,
        diffraction_scene(None, 1.0),
        1,
        source_position,
        listener_position,
    )?;
    let diffracted = trace_scene(
        &gpu,
        diffraction_scene(Some(edge.clone()), 1.0),
        2,
        source_position,
        listener_position,
    )?;
    let alternate_radius = trace_scene(
        &gpu,
        diffraction_scene(Some(diffraction_edge(0.9, 0.65)), 1.0),
        3,
        source_position,
        listener_position,
    )?;

    export_ir(&baseline, BASELINE_IR_EXPORT_PATH)?;
    export_ir(&diffracted, DIFFRACTION_IR_EXPORT_PATH)?;

    let baseline_nonzero = baseline
        .samples
        .iter()
        .filter(|sample| sample.left != 0.0 || sample.right != 0.0)
        .count();
    let diffracted_nonzero = diffracted
        .samples
        .iter()
        .filter(|sample| sample.left != 0.0 || sample.right != 0.0)
        .count();

    println!(
        "diffraction edge smoke: baseline_energy={}, diffraction_energy={}",
        baseline.energy, diffracted.energy
    );
    println!("  baseline nonzero samples: {baseline_nonzero}");
    println!("  diffraction nonzero samples: {diffracted_nonzero}");
    println!("  exported baseline ir: {BASELINE_IR_EXPORT_PATH}");
    println!("  exported diffraction ir: {DIFFRACTION_IR_EXPORT_PATH}");

    if diffracted.energy <= 0.0 {
        return Err(SoundError::invalid_state(
            "Expected diffraction edge to produce non-zero IR energy",
        ));
    }
    if diffracted_nonzero <= baseline_nonzero {
        return Err(SoundError::invalid_state(
            "Expected diffraction edge to add extra shadow-region arrivals",
        ));
    }

    let reference = btm_reference_ir(source_position, listener_position, &edge, 2_000.0, REFERENCE_SEGMENTS);
    let measured = diffracted
        .samples
        .iter()
        .map(|sample| sample.left as f64)
        .collect::<Vec<_>>();
    let measured_pressure = measured.iter().sum::<f64>();
    let reference_pressure = reference.iter().sum::<f64>();
    let pressure_error = (measured_pressure - reference_pressure).abs() / reference_pressure.abs().max(1.0e-12);
    if pressure_error > 1.0e-4 {
        return Err(SoundError::invalid_state(format!(
            "BTM pressure integral relative error {pressure_error:.6} exceeds 0.0001"
        )));
    }

    let maximum_spectral_error = TEST_FREQUENCIES_HZ
        .iter()
        .map(|frequency| {
            let measured_transfer = transfer(&measured, *frequency);
            let reference_transfer = transfer(&reference, *frequency);
            complex_distance(measured_transfer, reference_transfer) / magnitude(reference_transfer).max(1.0e-12)
        })
        .fold(0.0_f64, f64::max);
    if maximum_spectral_error > 0.04 {
        return Err(SoundError::invalid_state(format!(
            "BTM spectral relative error {maximum_spectral_error:.6} exceeds 0.04"
        )));
    }

    let alternate_radius_samples = alternate_radius
        .samples
        .iter()
        .map(|sample| sample.left as f64)
        .collect::<Vec<_>>();
    let radius_error = relative_l2(&measured, &alternate_radius_samples);
    if radius_error > 1.0e-5 {
        return Err(SoundError::invalid_state(format!(
            "BTM response changed with legacy proxy radius: relative L2 {radius_error:.9}"
        )));
    }

    // The sandbox authors four doorway edges. Exercise that dispatch shape and
    // verify that independent edge contributions remain linear under atomic
    // accumulation; coincident edges deliberately maximize write contention.
    let coincident_edges = (1..=4).map(|id| DiffractionEdge { id, ..edge.clone() }).collect();
    let four_edge = trace_scene(
        &gpu,
        diffraction_scene_with_edges(coincident_edges, 1.0),
        6,
        source_position,
        listener_position,
    )?;
    let four_edge_samples = four_edge
        .samples
        .iter()
        .map(|sample| sample.left as f64)
        .collect::<Vec<_>>();
    let expected_four_edge = measured.iter().map(|sample| sample * 4.0).collect::<Vec<_>>();
    let four_edge_linearity_error = relative_l2(&four_edge_samples, &expected_four_edge);
    if four_edge_linearity_error > 1.0e-5 {
        return Err(SoundError::invalid_state(format!(
            "Four-edge BTM accumulation is not linear: relative L2 {four_edge_linearity_error:.9}"
        )));
    }

    let visible_source = vec3(1.0, 0.0, -1.0);
    let visible_listener = vec3(1.0, 0.0, -2.0);
    let visible_baseline = trace_scene(&gpu, diffraction_scene(None, 0.18), 4, visible_source, visible_listener)?;
    let visible_zero_strength = trace_scene(
        &gpu,
        diffraction_scene(Some(diffraction_edge(3.0, 0.0)), 0.18),
        5,
        visible_source,
        visible_listener,
    )?;
    let visible_baseline_samples = visible_baseline
        .samples
        .iter()
        .flat_map(|sample| [sample.left as f64, sample.right as f64])
        .collect::<Vec<_>>();
    let visible_zero_strength_samples = visible_zero_strength
        .samples
        .iter()
        .flat_map(|sample| [sample.left as f64, sample.right as f64])
        .collect::<Vec<_>>();
    let visible_invariance_error = relative_l2(&visible_baseline_samples, &visible_zero_strength_samples);
    if visible_invariance_error > 1.0e-6 {
        return Err(SoundError::invalid_state(format!(
            "A zero-strength edge changed visible transport: relative L2 {visible_invariance_error:.9}"
        )));
    }

    println!("  BTM pressure relative error: {pressure_error:.9}");
    println!("  BTM maximum spectral relative error: {maximum_spectral_error:.9}");
    println!("  proxy-radius relative L2: {radius_error:.9}");
    println!("  four-edge linearity relative L2: {four_edge_linearity_error:.9}");
    println!("  visible zero-strength relative L2: {visible_invariance_error:.9}");

    Ok(())
}

fn trace_scene(
    gpu: &VkBackend,
    scene_description: SceneDescription,
    query_id: u32,
    source_position: Vec3,
    listener_position: Vec3,
) -> SoundResult<sound_engine::acoustics::IrSnapshot> {
    let mut scene = SceneManager::new();
    scene.load(scene_description)?;
    let mut pipeline = AcousticPipeline::new(gpu, engine_config())?;

    pipeline.build_ir(
        gpu,
        &mut scene,
        AcousticQuery {
            query_id,
            source_position,
            listener_position,
            listener_forward: -Vec3::Z,
            listener_up: Vec3::Y,
            source_amplitude: 2_000.0,
        },
    )
}

fn engine_config() -> EngineConfig {
    EngineConfig {
        sound: SoundConfig {
            output_channels: OutputChannels::Mono,
            sample_rate: SAMPLE_RATE,
            ir_num_samples: IR_SAMPLES,
        },
        acoustics: AcousticsConfig {
            listener_ray_count: 131_072,
            max_bounces: 1,
            maximum_path_length: 343.0,
            minimum_throughput: 1.0e-6,
            sampling_seed: 0,
        },
        cache: IrCacheConfig::default(),
        binaural: Default::default(),
        auralization: AuralizationConfig { block_size: 1024 },
    }
}

fn diffraction_scene(edge: Option<DiffractionEdge>, absorption: f32) -> SceneDescription {
    diffraction_scene_with_edges(edge.into_iter().collect(), absorption)
}

fn diffraction_scene_with_edges(diffraction_edges: Vec<DiffractionEdge>, absorption: f32) -> SceneDescription {
    SceneDescription {
        meshes: vec![MeshAsset {
            id: 1,
            vertices: baffle_vertices(),
            indices: Vec::new(),
            opaque: true,
        }],
        materials: vec![Material {
            id: 1,
            absorption_bands: vec![absorption],
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
        diffraction_edges,
    }
}

fn diffraction_edge(radius: f32, strength: f32) -> DiffractionEdge {
    DiffractionEdge {
        id: 1,
        start: vec3(0.0, -2.0, 0.0),
        end: vec3(0.0, 2.0, 0.0),
        bisector_dir: vec3(1.0, 0.0, 0.0),
        edge_angle_radians: 1.5 * std::f32::consts::PI,
        diffraction_radius: radius,
        base_strength: strength,
    }
}

fn btm_reference_ir(
    source: Vec3,
    listener: Vec3,
    edge: &DiffractionEdge,
    source_amplitude: f64,
    segments: usize,
) -> Vec<f64> {
    let mut samples = vec![0.0; IR_SAMPLES as usize];
    let edge_vector = edge.end - edge.start;
    let edge_length = edge_vector.length() as f64;
    let edge_axis = edge_vector.normalize();
    let bisector = (edge.bisector_dir - edge_axis * edge.bisector_dir.dot(edge_axis)).normalize();
    let plane_normal = bisector.cross(edge_axis).normalize();
    let delta = ((edge.edge_angle_radians - std::f32::consts::PI) * 0.5).clamp(0.0, std::f32::consts::FRAC_PI_2);
    let positive_shadow = (delta.cos() * plane_normal - delta.sin() * bisector).normalize();
    let source_relative = source - edge.start;
    let listener_relative = listener - edge.start;
    let source_radial = source_relative - source_relative.dot(edge_axis) * edge_axis;
    let listener_radial = listener_relative - listener_relative.dot(edge_axis) * edge_axis;
    let source_radius = source_radial.length() as f64;
    let listener_radius = listener_radial.length() as f64;
    let source_angle = wedge_angle(positive_shadow, edge_axis, source_radial);
    let listener_angle = wedge_angle(positive_shadow, edge_axis, listener_radial);
    let wedge_index = PI / edge.edge_angle_radians as f64;
    let segment_length = edge_length / segments as f64;
    let strength = (edge.base_strength as f64).clamp(0.0, 1.0).sqrt();

    for segment in 0..segments {
        let coordinate = (segment as f64 + 0.5) * segment_length;
        let edge_point = edge.start + edge_axis * coordinate as f32;
        let source_axial = (edge_point - source).dot(edge_axis) as f64;
        let listener_axial = (edge_point - listener).dot(edge_axis) as f64;
        let source_distance = source_radius.hypot(source_axial);
        let listener_distance = listener_radius.hypot(listener_axial);
        let eta_argument =
            (source_distance * listener_distance + source_axial * listener_axial) / (source_radius * listener_radius);
        let eta = eta_argument.max(1.0).acosh();
        let mut beta_sum = 0.0;
        for source_sign in [-1.0, 1.0] {
            for listener_sign in [-1.0, 1.0] {
                let phi = PI + source_sign * source_angle + listener_sign * listener_angle;
                beta_sum += (wedge_index * phi).sin() / ((wedge_index * eta).cosh() - (wedge_index * phi).cos());
            }
        }
        let total_distance = source_distance + listener_distance;
        let pressure = source_amplitude * strength * -wedge_index * beta_sum * segment_length
            / (source_distance * listener_distance);
        deposit_fractional(&mut samples, total_distance / 343.0 * SAMPLE_RATE as f64, pressure);
    }
    samples
}

fn wedge_angle(origin: Vec3, axis: Vec3, direction: Vec3) -> f64 {
    let origin = origin.normalize();
    let direction = direction.normalize();
    let sine = axis.dot(origin.cross(direction)) as f64;
    let cosine = origin.dot(direction) as f64;
    sine.atan2(cosine).rem_euclid(2.0 * PI)
}

fn deposit_fractional(samples: &mut [f64], position: f64, pressure: f64) {
    let base = position.floor() as usize;
    let fraction = position - base as f64;
    if let Some(sample) = samples.get_mut(base) {
        *sample += pressure * (1.0 - fraction);
    }
    if let Some(sample) = samples.get_mut(base + 1) {
        *sample += pressure * fraction;
    }
}

#[derive(Clone, Copy)]
struct Complex {
    real: f64,
    imaginary: f64,
}

fn transfer(samples: &[f64], frequency: f64) -> Complex {
    let angular_step = -2.0 * PI * frequency / SAMPLE_RATE as f64;
    samples.iter().enumerate().fold(
        Complex {
            real: 0.0,
            imaginary: 0.0,
        },
        |sum, (index, sample)| {
            let phase = angular_step * index as f64;
            Complex {
                real: sum.real + sample * phase.cos(),
                imaginary: sum.imaginary + sample * phase.sin(),
            }
        },
    )
}

fn complex_distance(left: Complex, right: Complex) -> f64 {
    (left.real - right.real).hypot(left.imaginary - right.imaginary)
}

fn magnitude(value: Complex) -> f64 {
    value.real.hypot(value.imaginary)
}

fn relative_l2(left: &[f64], right: &[f64]) -> f64 {
    let error = left
        .iter()
        .zip(right)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>();
    let reference = right.iter().map(|value| value * value).sum::<f64>();
    (error / reference.max(1.0e-24)).sqrt()
}

fn baffle_vertices() -> Vec<Vec3> {
    let x = 0.0;
    let y0 = -2.0;
    let y1 = 2.0;
    let z0 = -3.0;
    let z1 = 0.0;
    vec![
        vec3(x, y0, z0),
        vec3(x, y1, z0),
        vec3(x, y1, z1),
        vec3(x, y0, z0),
        vec3(x, y1, z1),
        vec3(x, y0, z1),
    ]
}
