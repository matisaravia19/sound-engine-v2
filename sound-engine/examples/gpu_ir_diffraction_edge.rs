use glam::{Mat4, Vec3, vec3};
use sound_engine::acoustics::{AcousticPipeline, AcousticQuery};
use sound_engine::core::config::{
    AcousticsConfig, AuralizationConfig, EngineConfig, IrCacheConfig, OutputChannels, SoundConfig,
};
use sound_engine::core::debug::export_ir;
use sound_engine::core::error::{SoundError, SoundResult};
use sound_engine::gpu::backend::VkBackend;
use sound_engine::scene::{
    DiffractionEdge, Material, MeshAsset, SceneDescription, SceneManager, SceneObject,
};

const BASELINE_IR_EXPORT_PATH: &str = "target/gpu_ir_diffraction_baseline_ir.csv";
const DIFFRACTION_IR_EXPORT_PATH: &str = "target/gpu_ir_diffraction_edge_ir.csv";
const SAMPLE_RATE: u32 = 48_000;
const IR_SAMPLES: u32 = SAMPLE_RATE;

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

    let measured = diffracted
        .samples
        .iter()
        .map(|sample| sample.left as f64)
        .collect::<Vec<_>>();
    let measured_pressure = measured.iter().sum::<f64>();
    if !measured_pressure.is_finite() || measured.iter().any(|sample| !sample.is_finite()) {
        return Err(SoundError::invalid_state(format!(
            "Ray-traced diffraction produced a non-finite pressure"
        )));
    }

    let alternate_radius_samples = alternate_radius
        .samples
        .iter()
        .map(|sample| sample.left as f64)
        .collect::<Vec<_>>();
    let alternate_pressure = alternate_radius_samples.iter().sum::<f64>();
    let radius_error =
        (measured_pressure - alternate_pressure).abs() / measured_pressure.abs().max(1.0e-12);
    if radius_error > 0.35 {
        return Err(SoundError::invalid_state(format!(
            "Normalized ray diffraction changed across proxy radii by {radius_error:.6}"
        )));
    }

    let unobstructed = trace_scene(&gpu, empty_scene(), 6, source_position, listener_position)?;
    if diffracted.energy > unobstructed.energy * 1.01 {
        return Err(SoundError::invalid_state(format!(
            "Shadow diffraction energy {} exceeds unobstructed energy {}",
            diffracted.energy, unobstructed.energy
        )));
    }

    let visible_source = vec3(1.0, 0.0, -1.0);
    let visible_listener = vec3(1.0, 0.0, -2.0);
    let visible_baseline = trace_scene(
        &gpu,
        diffraction_scene(None, 0.18),
        4,
        visible_source,
        visible_listener,
    )?;
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
    let visible_invariance_error =
        relative_l2(&visible_baseline_samples, &visible_zero_strength_samples);
    if visible_invariance_error > 1.0e-6 {
        return Err(SoundError::invalid_state(format!(
            "A transparent zero-strength proxy changed ordinary transport: relative L2 {visible_invariance_error:.9}"
        )));
    }

    let sandbox_source_position = vec3(-4.5, 0.45, 0.0);
    let (doorway_maximum_step_db, doorway_span_db, shadow_excess_db, boundary_step_db) =
        doorway_position_sweep(&gpu, sandbox_source_position)?;
    let (multibounce_energy_ratio, multibounce_peak_ratio) =
        doorway_multibounce_growth(&gpu, sandbox_source_position)?;
    if doorway_maximum_step_db > 2.0 {
        return Err(SoundError::invalid_state(format!(
            "Visible doorway sweep changed by {doorway_maximum_step_db:.6} dB between adjacent 5 cm positions"
        )));
    }
    if doorway_span_db > 3.0 {
        return Err(SoundError::invalid_state(format!(
            "Visible doorway sweep spans {doorway_span_db:.6} dB across the direct-view region"
        )));
    }
    if shadow_excess_db > 0.0 {
        return Err(SoundError::invalid_state(format!(
            "Doorway shadow exceeds the direct-view peak by {shadow_excess_db:.6} dB"
        )));
    }
    if boundary_step_db > 12.5 {
        return Err(SoundError::invalid_state(format!(
            "Doorway visibility boundary changes by {boundary_step_db:.6} dB over 2.5 mm"
        )));
    }
    if multibounce_energy_ratio > 16.0 || multibounce_peak_ratio > 4.0 {
        return Err(SoundError::invalid_state(format!(
            "Doorway multi-bounce diffraction grew excessively: energy ratio {multibounce_energy_ratio:.6}, peak ratio {multibounce_peak_ratio:.6}"
        )));
    }

    println!("  signed diffraction pressure: {measured_pressure:.9}");
    println!("  proxy-radius relative pressure difference: {radius_error:.9}");
    println!(
        "  shadow/unobstructed energy ratio: {:.9}",
        diffracted.energy / unobstructed.energy
    );
    println!("  zero-strength proxy relative L2: {visible_invariance_error:.9}");
    println!("  doorway maximum adjacent level step: {doorway_maximum_step_db:.6} dB");
    println!("  doorway direct-view level span: {doorway_span_db:.6} dB");
    println!("  doorway shadow excess over direct view: {shadow_excess_db:.6} dB");
    println!("  doorway 3-bounce/1-bounce energy ratio: {multibounce_energy_ratio:.9}");
    println!("  doorway 3-bounce/1-bounce peak ratio: {multibounce_peak_ratio:.9}");

    Ok(())
}

fn doorway_multibounce_growth(gpu: &VkBackend, source_position: Vec3) -> SoundResult<(f64, f64)> {
    let listener_position = vec3(1.8, 0.45, 0.0);
    let first_order = trace_scene_with_bounces(
        gpu,
        doorway_scene(),
        9_000,
        source_position,
        listener_position,
        1,
    )?;
    let multibounce = trace_scene_with_bounces(
        gpu,
        doorway_scene(),
        9_001,
        source_position,
        listener_position,
        3,
    )?;
    let first_peak = first_order
        .samples
        .iter()
        .map(|sample| sample.left.abs() as f64)
        .fold(0.0_f64, f64::max);
    let multibounce_peak = multibounce
        .samples
        .iter()
        .map(|sample| sample.left.abs() as f64)
        .fold(0.0_f64, f64::max);
    Ok((
        multibounce.energy as f64 / (first_order.energy as f64).max(1.0e-24),
        multibounce_peak / first_peak.max(1.0e-24),
    ))
}

fn doorway_position_sweep(
    gpu: &VkBackend,
    source_position: Vec3,
) -> SoundResult<(f64, f64, f64, f64)> {
    let mut scene = SceneManager::new();
    scene.load(doorway_scene())?;
    let mut pipeline = AcousticPipeline::new(gpu, engine_config())?;
    let mut levels = Vec::new();

    const SWEEP_START: f32 = -3.0;
    const SWEEP_STEP: f32 = 0.05;
    for step in 0..=120 {
        let listener_position = vec3(1.8, 0.45, SWEEP_START + step as f32 * SWEEP_STEP);
        let ir = pipeline.build_ir(
            gpu,
            &mut scene,
            AcousticQuery {
                query_id: 100 + step,
                source_position,
                listener_position,
                listener_forward: -Vec3::X,
                listener_up: Vec3::Y,
                source_amplitude: 1.0,
            },
        )?;
        if !ir.energy.is_finite()
            || ir
                .samples
                .iter()
                .any(|sample| !sample.left.is_finite() || !sample.right.is_finite())
        {
            return Err(SoundError::invalid_state(
                "Doorway sweep produced non-finite IR data",
            ));
        }
        levels.push(10.0 * (ir.energy as f64).max(1.0e-24).log10());
    }

    let visible_levels = &levels[33..=87];
    let (maximum_step_index, maximum_step_db) = visible_levels
        .windows(2)
        .enumerate()
        .map(|(index, pair)| (index, (pair[1] - pair[0]).abs()))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .unwrap_or((0, 0.0));
    let minimum_level = visible_levels.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum_level = visible_levels
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let maximum_shadow_level = levels[..=32]
        .iter()
        .chain(&levels[88..])
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let maximum_level_index = levels
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .unwrap_or(0);
    let (full_step_index, full_step_db) = levels
        .windows(2)
        .enumerate()
        .map(|(index, pair)| (index, (pair[1] - pair[0]).abs()))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .unwrap_or((0, 0.0));
    println!(
        "  doorway largest step z={:.2}->{:.2}: {:.6}->{:.6} dB",
        -1.35 + maximum_step_index as f32 * SWEEP_STEP,
        -1.35 + (maximum_step_index + 1) as f32 * SWEEP_STEP,
        visible_levels[maximum_step_index],
        visible_levels[maximum_step_index + 1]
    );
    println!(
        "  doorway full-sweep peak z={:.2}: {:.6} dB",
        SWEEP_START + maximum_level_index as f32 * SWEEP_STEP,
        levels[maximum_level_index]
    );
    println!(
        "  doorway full-sweep largest step z={:.2}->{:.2}: {:.6}->{:.6} dB ({:.6} dB)",
        SWEEP_START + full_step_index as f32 * SWEEP_STEP,
        SWEEP_START + (full_step_index + 1) as f32 * SWEEP_STEP,
        levels[full_step_index],
        levels[full_step_index + 1],
        full_step_db
    );

    let mut boundary_levels = Vec::new();
    for step in 0..=40 {
        let listener_position = vec3(1.8, 0.45, -1.40 + step as f32 * 0.0025);
        let ir = pipeline.build_ir(
            gpu,
            &mut scene,
            AcousticQuery {
                query_id: 1_000 + step,
                source_position,
                listener_position,
                listener_forward: -Vec3::X,
                listener_up: Vec3::Y,
                source_amplitude: 1.0,
            },
        )?;
        boundary_levels.push(10.0 * (ir.energy as f64).max(1.0e-24).log10());
    }
    let (boundary_step_index, boundary_step_db) = boundary_levels
        .windows(2)
        .enumerate()
        .map(|(index, pair)| (index, (pair[1] - pair[0]).abs()))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .unwrap_or((0, 0.0));
    println!(
        "  doorway 2.5mm boundary step z={:.4}->{:.4}: {:.6}->{:.6} dB ({:.6} dB)",
        -1.40 + boundary_step_index as f32 * 0.0025,
        -1.40 + (boundary_step_index + 1) as f32 * 0.0025,
        boundary_levels[boundary_step_index],
        boundary_levels[boundary_step_index + 1],
        boundary_step_db
    );
    Ok((
        maximum_step_db,
        maximum_level - minimum_level,
        maximum_shadow_level - maximum_level,
        boundary_step_db,
    ))
}

fn trace_scene(
    gpu: &VkBackend,
    scene_description: SceneDescription,
    query_id: u32,
    source_position: Vec3,
    listener_position: Vec3,
) -> SoundResult<sound_engine::acoustics::IrSnapshot> {
    trace_scene_with_bounces(
        gpu,
        scene_description,
        query_id,
        source_position,
        listener_position,
        1,
    )
}

fn trace_scene_with_bounces(
    gpu: &VkBackend,
    scene_description: SceneDescription,
    query_id: u32,
    source_position: Vec3,
    listener_position: Vec3,
    max_bounces: u32,
) -> SoundResult<sound_engine::acoustics::IrSnapshot> {
    let mut scene = SceneManager::new();
    scene.load(scene_description)?;
    let mut pipeline = AcousticPipeline::new(gpu, engine_config_with_bounces(max_bounces))?;

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
    engine_config_with_bounces(1)
}

fn engine_config_with_bounces(max_bounces: u32) -> EngineConfig {
    EngineConfig {
        sound: SoundConfig {
            output_channels: OutputChannels::Mono,
            sample_rate: SAMPLE_RATE,
            ir_num_samples: IR_SAMPLES,
        },
        acoustics: AcousticsConfig {
            listener_ray_count: 131_072,
            max_bounces,
            maximum_path_length: 343.0,
            minimum_throughput: 1.0e-6,
            sampling_seed: 0,
        },
        cache: IrCacheConfig::default(),
        binaural: Default::default(),
        auralization: AuralizationConfig {
            block_size: 1024,
        },
    }
}

fn diffraction_scene(edge: Option<DiffractionEdge>, absorption: f32) -> SceneDescription {
    diffraction_scene_with_edges(edge.into_iter().collect(), absorption)
}

fn diffraction_scene_with_edges(
    diffraction_edges: Vec<DiffractionEdge>,
    absorption: f32,
) -> SceneDescription {
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

fn empty_scene() -> SceneDescription {
    SceneDescription {
        meshes: Vec::new(),
        materials: Vec::new(),
        objects: Vec::new(),
        diffraction_edges: Vec::new(),
    }
}

fn doorway_scene() -> SceneDescription {
    let mut vertices = Vec::new();
    vertices.extend(box_vertices(vec3(0.0, 1.5, -3.0), vec3(0.25, 3.0, 4.0)));
    vertices.extend(box_vertices(vec3(0.0, 1.5, 3.0), vec3(0.25, 3.0, 4.0)));
    let half_thickness = 0.125;
    let half_width = 1.0;
    let diagonal = std::f32::consts::FRAC_1_SQRT_2;
    let edges = vec![
        DiffractionEdge {
            id: 1,
            start: vec3(-half_thickness, 0.0, -half_width),
            end: vec3(-half_thickness, 3.0, -half_width),
            bisector_dir: vec3(-diagonal, 0.0, diagonal),
            edge_angle_radians: 1.5 * std::f32::consts::PI,
            diffraction_radius: 1.25,
            base_strength: 0.85,
        },
        DiffractionEdge {
            id: 2,
            start: vec3(half_thickness, 0.0, -half_width),
            end: vec3(half_thickness, 3.0, -half_width),
            bisector_dir: vec3(diagonal, 0.0, diagonal),
            edge_angle_radians: 1.5 * std::f32::consts::PI,
            diffraction_radius: 1.25,
            base_strength: 0.85,
        },
        DiffractionEdge {
            id: 3,
            start: vec3(-half_thickness, 0.0, half_width),
            end: vec3(-half_thickness, 3.0, half_width),
            bisector_dir: vec3(-diagonal, 0.0, -diagonal),
            edge_angle_radians: 1.5 * std::f32::consts::PI,
            diffraction_radius: 1.25,
            base_strength: 0.85,
        },
        DiffractionEdge {
            id: 4,
            start: vec3(half_thickness, 0.0, half_width),
            end: vec3(half_thickness, 3.0, half_width),
            bisector_dir: vec3(diagonal, 0.0, -diagonal),
            edge_angle_radians: 1.5 * std::f32::consts::PI,
            diffraction_radius: 1.25,
            base_strength: 0.85,
        },
    ];
    SceneDescription {
        meshes: vec![MeshAsset {
            id: 1,
            vertices,
            indices: Vec::new(),
            opaque: true,
        }],
        materials: vec![Material {
            id: 1,
            absorption_bands: vec![0.18],
            scattering: 0.12,
            roughness: 0.15,
            transmission: None,
        }],
        objects: vec![SceneObject {
            id: 1,
            mesh_id: 1,
            material_id: 1,
            transform: Mat4::IDENTITY,
            active: true,
        }],
        diffraction_edges: edges,
    }
}

fn box_vertices(center: Vec3, size: Vec3) -> Vec<Vec3> {
    let minimum = center - size * 0.5;
    let maximum = center + size * 0.5;
    let corners = [
        vec3(minimum.x, minimum.y, minimum.z),
        vec3(maximum.x, minimum.y, minimum.z),
        vec3(maximum.x, maximum.y, minimum.z),
        vec3(minimum.x, maximum.y, minimum.z),
        vec3(minimum.x, minimum.y, maximum.z),
        vec3(maximum.x, minimum.y, maximum.z),
        vec3(maximum.x, maximum.y, maximum.z),
        vec3(minimum.x, maximum.y, maximum.z),
    ];
    let faces = [
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [3, 7, 6, 2],
        [0, 4, 7, 3],
        [1, 2, 6, 5],
    ];
    let mut vertices = Vec::with_capacity(36);
    for [a, b, c, d] in faces {
        vertices.extend_from_slice(&[
            corners[a], corners[b], corners[c], corners[a], corners[c], corners[d],
        ]);
    }
    vertices
}

fn diffraction_edge(radius: f32, strength: f32) -> DiffractionEdge {
    DiffractionEdge {
        id: 1,
        start: vec3(0.0, -2.0, 0.0),
        end: vec3(0.0, 2.0, 0.0),
        // The mesh is a zero-thickness screen extending toward -Z. A 270-degree
        // wedge with bisector +X would incorrectly put the listener inside solid.
        bisector_dir: Vec3::Z,
        edge_angle_radians: std::f32::consts::TAU,
        diffraction_radius: radius,
        base_strength: strength,
    }
}

// Retained as an offline mathematical oracle; production diffraction is traced
// entirely in the ray-generation pipeline.
#[allow(dead_code)]
mod btm_reference {
    use super::*;
    use std::f64::consts::PI;

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
        let bisector =
            (edge.bisector_dir - edge_axis * edge.bisector_dir.dot(edge_axis)).normalize();
        let plane_normal = bisector.cross(edge_axis).normalize();
        let delta = ((edge.edge_angle_radians - std::f32::consts::PI) * 0.5)
            .clamp(0.0, std::f32::consts::FRAC_PI_2);
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
            let eta_argument = (source_distance * listener_distance
                + source_axial * listener_axial)
                / (source_radius * listener_radius);
            let eta = eta_argument.max(1.0).acosh();
            let mut beta_sum = 0.0;
            for source_sign in [-1.0, 1.0] {
                for listener_sign in [-1.0, 1.0] {
                    let phi = PI + source_sign * source_angle + listener_sign * listener_angle;
                    beta_sum += (wedge_index * phi).sin()
                        / ((wedge_index * eta).cosh() - (wedge_index * phi).cos());
                }
            }
            let total_distance = source_distance + listener_distance;
            let pressure =
                source_amplitude * strength * -wedge_index / (4.0 * PI) * beta_sum * segment_length
                    / (source_distance * listener_distance);
            deposit_fractional(
                &mut samples,
                total_distance / 343.0 * SAMPLE_RATE as f64,
                pressure,
            );
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
