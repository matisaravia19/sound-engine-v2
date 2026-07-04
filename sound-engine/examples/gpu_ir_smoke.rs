use glam::vec3;
use sound_engine::acoustics::{AcousticPipeline, AcousticQuery};
use sound_engine::core::config::{
    AcousticsConfig, AuralizationConfig, EngineConfig, IrCacheConfig, OutputChannels, SoundConfig,
};
use sound_engine::core::debug::export_ir;
use sound_engine::core::error::{SoundError, SoundResult};
use sound_engine::gpu::backend::VkBackend;
use sound_engine::scene::{Material, MeshAsset, SceneDescription, SceneManager, SceneObject};

const IR_EXPORT_PATH: &str = "target/gpu_ir_smoke_ir.csv";

fn main() -> SoundResult<()> {
    let gpu = VkBackend::new()?;
    let mut scene = SceneManager::new();
    scene.load(smoke_scene())?;

    let mut pipeline = AcousticPipeline::new(&gpu, engine_config(44_100, 44_100, 0.2, 1024, 0))?;

    let ir = pipeline.build_ir(
        &gpu,
        &mut scene,
        AcousticQuery {
            query_id: 1,
            source_position: vec3(0.0, 0.0, 0.0),
            listener_position: vec3(1.0, 0.0, 0.0),
            listener_right: vec3(1.0, 0.0, 0.0),
            source_energy: 1.0,
        },
    )?;

    println!(
        "gpu ir smoke: version={}, query={}, samples={}, energy={}",
        ir.scene_version,
        ir.query_id,
        ir.samples.len(),
        ir.energy
    );
    export_ir(&ir, IR_EXPORT_PATH)?;
    println!("  exported ir: {IR_EXPORT_PATH}");

    for (sample_idx, sample) in ir.samples.iter().enumerate() {
        if sample.left != 0.0 || sample.right != 0.0 {
            println!("  sample {}: left={}, right={}", sample_idx, sample.left, sample.right);
        }
    }

    if ir.energy <= 0.0 {
        return Err(SoundError::invalid_state("Expected non-zero IR energy"));
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

fn smoke_scene() -> SceneDescription {
    SceneDescription {
        meshes: vec![MeshAsset {
            id: 1,
            vertices: vec![vec3(-0.5, -0.5, -1.0), vec3(0.5, -0.5, -1.0), vec3(0.0, 0.5, -1.0)],
            indices: Vec::new(),
            opaque: true,
        }],
        materials: vec![Material {
            id: 1,
            absorption_bands: vec![0.2],
            scattering: 0.1,
            transmission: None,
        }],
        objects: vec![SceneObject {
            id: 1,
            mesh_id: 1,
            material_id: 1,
            transform: glam::Mat4::IDENTITY,
            active: true,
        }],
        diffraction_edges: Vec::new(),
    }
}
