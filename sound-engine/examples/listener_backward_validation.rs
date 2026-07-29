use glam::{Mat4, Vec3, vec3};
use sound_engine::acoustics::{AcousticPipeline, AcousticQuery, IrSnapshot};
use sound_engine::core::config::{
    AcousticsConfig, AuralizationConfig, EngineConfig, IrCacheConfig, OutputChannels, SoundConfig,
};
use sound_engine::core::error::{SoundError, SoundResult};
use sound_engine::gpu::backend::VkBackend;
use sound_engine::scene::{DiffractionEdge, Material, MeshAsset, SceneDescription, SceneManager, SceneObject};

const SAMPLE_RATE: u32 = 48_000;
const IR_SAMPLES: u32 = SAMPLE_RATE;

fn main() -> SoundResult<()> {
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
        listener_right: Vec3::X,
        source_amplitude: 1.0,
    };
    let low = trace(gpu, empty_scene(), config(64, 0), query)?;
    let high = trace(gpu, empty_scene(), config(16_384, 0), query)?;
    let low_sum = channel_sum(&low);
    let high_sum = channel_sum(&high);
    require(
        (low_sum - 1.0).abs() < 1.0e-5,
        "direct path amplitude must equal 1 / distance",
    )?;
    require(
        (low_sum - high_sum).abs() < 1.0e-6,
        "direct path must not scale with listener ray count",
    )?;
    require(
        nonzero_samples(&low) <= 2,
        "fractional direct deposition must touch at most two samples",
    )
}

fn validate_occlusion(gpu: &VkBackend) -> SoundResult<()> {
    let query = AcousticQuery {
        query_id: 2,
        source_position: vec3(-1.0, 0.0, 0.0),
        listener_position: vec3(1.0, 0.0, 0.0),
        listener_right: Vec3::X,
        source_amplitude: 1.0,
    };
    let ir = trace(gpu, occluding_wall_scene(), config(4_096, 0), query)?;
    require(channel_sum(&ir) == 0.0, "occluding wall must remove the direct path")
}

fn validate_reflection(gpu: &VkBackend) -> SoundResult<()> {
    let source = Vec3::ZERO;
    let listener = vec3(0.0, 0.7, 0.0);
    let query = AcousticQuery {
        query_id: 3,
        source_position: source,
        listener_position: listener,
        listener_right: Vec3::X,
        source_amplitude: 1.0,
    };
    let ir = trace(gpu, reflection_scene(), config(262_144, 1), query)?;
    let image_source = vec3(4.0, 0.0, 0.0);
    let expected_sample = (image_source.distance(listener) / 343.0 * SAMPLE_RATE as f32).round() as usize;
    let start = expected_sample.saturating_sub(1);
    let end = (expected_sample + 2).min(ir.samples.len());
    let reflected_peak = ir.samples[start..end]
        .iter()
        .map(|sample| sample.left.abs())
        .fold(0.0_f32, f32::max);
    require(
        reflected_peak > 1.0e-7,
        "single-wall reflection must appear near the image-source delay",
    )
}

fn validate_diffraction_convergence_and_radius(gpu: &VkBackend) -> SoundResult<()> {
    let query = AcousticQuery {
        query_id: 4,
        source_position: vec3(1.0, 0.0, -1.0),
        listener_position: vec3(-1.0, 0.0, 0.5),
        listener_right: Vec3::X,
        source_amplitude: 1.0,
    };
    let baseline = trace(gpu, diffraction_scene(None), config(65_536, 1), query)?;
    require(
        channel_sum(&baseline) == 0.0,
        "diffraction test direct path must be occluded",
    )?;

    let low = trace(gpu, diffraction_scene(Some(0.45)), config(65_536, 1), query)?;
    let high = trace(gpu, diffraction_scene(Some(0.45)), config(262_144, 1), query)?;
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
        auralization: AuralizationConfig { block_size: 1024 },
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
            bisector_dir: Vec3::X,
            edge_angle_radians: 1.5 * std::f32::consts::PI,
            diffraction_radius: radius,
            base_strength: 0.65,
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
            absorption_bands: vec![absorption],
            scattering: 0.0,
            roughness,
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
    }
}

fn channel_sum(ir: &IrSnapshot) -> f32 {
    ir.samples.iter().map(|sample| sample.left.abs()).sum()
}

fn nonzero_samples(ir: &IrSnapshot) -> usize {
    ir.samples.iter().filter(|sample| sample.left != 0.0).count()
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
