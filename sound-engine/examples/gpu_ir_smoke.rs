use sound_engine::acoustics::{AcousticConfig, AcousticPipeline, AcousticQuery};
use sound_engine::error::{SoundError, SoundResult};
use sound_engine::gpu::backend::VkBackend;
use sound_engine::scene::{
    Material, MaterialId, MeshAsset, MeshId, ObjectId, SceneDescription, SceneManager, SceneObject, identity_transform,
};

fn main() -> SoundResult<()> {
    let gpu = VkBackend::new()?;
    let mut scene = SceneManager::new();
    scene.load(smoke_scene())?;

    let mut pipeline = AcousticPipeline::new(
        &gpu,
        AcousticConfig {
            sample_rate: 44_100,
            ir_len_samples: 44_100,
            rays_per_query: 1,
            max_bounces: 0,
        },
    )?;

    let ir = pipeline.build_ir(
        &gpu,
        &mut scene,
        AcousticQuery {
            query_id: 1,
            source_position: [0.0, 0.0, 0.0],
            listener_position: [1.0, 0.0, 0.0],
            gain: 1.0,
        },
    )?;

    println!(
        "gpu ir smoke: version={}, query={}, samples={}, energy={}",
        ir.scene_version.0,
        ir.query_id,
        ir.samples.len(),
        ir.energy
    );

    for (sample_idx, sample) in ir.samples.iter().enumerate() {
        if *sample > 0.0 {
            println!("  sample {}: {}", sample_idx, sample);
        }
    }

    if ir.energy <= 0.0 {
        return Err(SoundError::invalid_state("Expected non-zero IR energy"));
    }

    Ok(())
}

fn smoke_scene() -> SceneDescription {
    SceneDescription {
        meshes: vec![MeshAsset {
            id: MeshId(1),
            vertices: vec![[-0.5, -0.5, -1.0], [0.5, -0.5, -1.0], [0.0, 0.5, -1.0]],
            indices: Vec::new(),
            opaque: true,
        }],
        materials: vec![Material {
            id: MaterialId(1),
            absorption_bands: vec![0.2],
            scattering: 0.1,
            transmission: None,
        }],
        objects: vec![SceneObject {
            id: ObjectId(1),
            mesh_id: MeshId(1),
            material_id: MaterialId(1),
            transform: identity_transform(),
            active: true,
        }],
    }
}
