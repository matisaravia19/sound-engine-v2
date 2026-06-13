use glam::vec3;
use sound_engine::error::{SoundError, SoundResult};
use sound_engine::gpu::backend::VkBackend;
use sound_engine::scene::{
    Material, MeshAsset, SceneDescription, SceneManager, SceneObject, SceneUpdate, SceneUpdates,
};

fn main() -> SoundResult<()> {
    let gpu = VkBackend::new()?;
    let mut scene = SceneManager::new();
    scene.load(two_triangle_scene())?;

    let synced = scene.sync_gpu_if_needed(&gpu)?;
    println!(
        "initial scene sync: version={}, instances={}",
        synced.version(),
        synced.instance_count()
    );
    if synced.instance_count() != 2 {
        return Err(SoundError::invalid_state(
            "Expected two active instances after initial sync",
        ));
    }

    scene.apply_updates(SceneUpdates {
        updates: vec![SceneUpdate::SetTransform {
            id: 2,
            transform: glam::Mat4::from_translation(glam::Vec3::new(0.25, 0.0, 0.0)),
        }],
    })?;
    let synced = scene.sync_gpu_if_needed(&gpu)?;
    println!(
        "moved object sync: version={}, instances={}",
        synced.version(),
        synced.instance_count()
    );

    scene.apply_updates(SceneUpdates {
        updates: vec![SceneUpdate::RemoveObject(2)],
    })?;
    let synced = scene.sync_gpu_if_needed(&gpu)?;
    println!(
        "removed object sync: version={}, instances={}",
        synced.version(),
        synced.instance_count()
    );
    if synced.instance_count() != 1 {
        return Err(SoundError::invalid_state("Expected one active instance after removal"));
    }

    Ok(())
}

fn two_triangle_scene() -> SceneDescription {
    SceneDescription {
        meshes: vec![MeshAsset {
            id: 1,
            vertices: vec![vec3(-0.5, -0.5, 0.0), vec3(0.5, -0.5, 0.0), vec3(0.0, 0.5, 0.0)],
            indices: Vec::new(),
            opaque: true,
        }],
        materials: vec![Material {
            id: 1,
            absorption_bands: vec![0.2],
            scattering: 0.1,
            transmission: None,
        }],
        objects: vec![
            SceneObject {
                id: 1,
                mesh_id: 1,
                material_id: 1,
                transform: glam::Mat4::IDENTITY,
                active: true,
            },
            SceneObject {
                id: 2,
                mesh_id: 1,
                material_id: 1,
                transform: glam::Mat4::from_translation(glam::Vec3::new(1.0, 0.0, 0.0)),
                active: true,
            },
        ],
    }
}
