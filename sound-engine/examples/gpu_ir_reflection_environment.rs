use glam::{Mat4, Vec3, vec3};
use sound_engine::acoustics::{AcousticPipeline, AcousticQuery};
use sound_engine::core::config::{
    AcousticsConfig, AuralizationConfig, EngineConfig, IrCacheConfig, OutputChannels, SoundConfig,
};
use sound_engine::core::debug::export_ir;
use sound_engine::core::error::{SoundError, SoundResult};
use sound_engine::gpu::backend::VkBackend;
use sound_engine::scene::{Material, MeshAsset, SceneDescription, SceneManager, SceneObject};

const IR_EXPORT_PATH: &str = "target/gpu_ir_reflection_environment_ir.csv";

fn main() -> SoundResult<()> {
    let gpu = VkBackend::new()?;
    let mut scene = SceneManager::new();
    scene.load(reflective_environment_scene())?;

    let sample_rate = 48_000;
    let source_position = vec3(0.0, 0.0, 0.0);
    let listener_position = vec3(0.0, 0.7, 0.0);
    let listener_radius = 0.35;
    let mut pipeline = AcousticPipeline::new(
        &gpu,
        engine_config(sample_rate, sample_rate, listener_radius, 1_000_000, 10),
    )?;

    let ir = pipeline.build_ir(
        &gpu,
        &mut scene,
        AcousticQuery {
            query_id: 2,
            source_position,
            listener_position,
            listener_right: vec3(1.0, 0.0, 0.0),
            source_energy: 500.0,
        },
    )?;

    println!(
        "gpu reflection environment: version={}, query={}, samples={}, energy={}",
        ir.scene_version,
        ir.query_id,
        ir.samples.len(),
        ir.energy
    );
    export_ir(&ir, IR_EXPORT_PATH)?;
    println!("  exported ir: {IR_EXPORT_PATH}");

    let direct_center_sample =
        ((listener_position - source_position).length() / 343.0 * sample_rate as f32).round() as usize;
    let late_start = direct_center_sample + 100;
    let mut nonzero = 0;
    let mut late_nonzero = 0;
    for (sample_idx, sample) in ir.samples.iter().enumerate() {
        if sample.left != 0.0 || sample.right != 0.0 {
            nonzero += 1;
            if sample_idx >= late_start {
                late_nonzero += 1;
            }
            println!("  sample {}: left={}, right={}", sample_idx, sample.left, sample.right);
        }
    }

    if nonzero == 0 {
        return Err(SoundError::invalid_state("Expected direct listener hits"));
    }
    if late_nonzero == 0 {
        return Err(SoundError::invalid_state("Expected late reflected listener hits"));
    }

    Ok(())
}

fn engine_config(
    sample_rate: u32,
    ir_num_samples: u32,
    listener_radius: f32,
    rays_per_query: u32,
    max_bounces: u32,
) -> EngineConfig {
    EngineConfig {
        sound: SoundConfig {
            output_channels: OutputChannels::Stereo,
            sample_rate,
            ir_num_samples,
        },
        acoustics: AcousticsConfig {
            listener_radius,
            rays_per_query,
            max_bounces,
        },
        cache: IrCacheConfig::default(),
        auralization: AuralizationConfig { block_size: 1024 },
    }
}

fn reflective_environment_scene() -> SceneDescription {
    SceneDescription {
        meshes: vec![MeshAsset {
            id: 1,
            vertices: reflector_wall_vertices(2.0, 20.0),
            indices: Vec::new(),
            opaque: true,
        }],
        materials: vec![Material {
            id: 1,
            absorption_bands: vec![0.25],
            scattering: 0.0,
            transmission: None,
        }],
        objects: vec![SceneObject {
            id: 1,
            mesh_id: 1,
            material_id: 1,
            transform: Mat4::IDENTITY,
            active: true,
        }],
    }
}

fn reflector_wall_vertices(x: f32, half_extent: f32) -> Vec<Vec3> {
    let n = -half_extent;
    let p = half_extent;
    vec![
        vec3(x, n, n),
        vec3(x, p, n),
        vec3(x, p, p),
        vec3(x, n, n),
        vec3(x, p, p),
        vec3(x, n, p),
    ]
}
