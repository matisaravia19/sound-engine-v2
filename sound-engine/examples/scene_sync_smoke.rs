use sound_engine::error::{SoundError, SoundResult};
use sound_engine::gpu::backend::VkBackend;
use sound_engine::scene::{
    Material, MaterialId, MeshAsset, MeshId, ObjectId, ObjectUpdate, SceneDescription, SceneManager, SceneObject,
    SceneUpdates, TopologyChange, identity_transform, translation_transform,
};

fn main() -> SoundResult<()> {
    let gpu = VkBackend::new()?;
    let mut scene = SceneManager::new();
    scene.load(two_triangle_scene())?;

    let synced = scene.sync_gpu_if_needed(&gpu)?;
    println!(
        "initial scene sync: version={}, instances={}",
        synced.version().0,
        synced.instance_count()
    );
    if synced.instance_count() != 2 {
        return Err(SoundError::invalid_state(
            "Expected two active instances after initial sync",
        ));
    }

    scene.apply_updates(SceneUpdates {
        object_updates: vec![ObjectUpdate::SetTransform {
            id: ObjectId(2),
            transform: translation_transform(0.25, 0.0, 0.0),
        }],
        ..SceneUpdates::default()
    })?;
    let synced = scene.sync_gpu_if_needed(&gpu)?;
    println!(
        "moved object sync: version={}, instances={}",
        synced.version().0,
        synced.instance_count()
    );

    scene.apply_updates(SceneUpdates {
        topology_changes: vec![TopologyChange::RemoveObject(ObjectId(2))],
        ..SceneUpdates::default()
    })?;
    let synced = scene.sync_gpu_if_needed(&gpu)?;
    println!(
        "removed object sync: version={}, instances={}",
        synced.version().0,
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
            id: MeshId(1),
            vertices: vec![[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]],
            indices: Vec::new(),
            opaque: true,
        }],
        materials: vec![Material {
            id: MaterialId(1),
            absorption_bands: vec![0.2],
            scattering: 0.1,
            transmission: None,
        }],
        objects: vec![
            SceneObject {
                id: ObjectId(1),
                mesh_id: MeshId(1),
                material_id: MaterialId(1),
                transform: identity_transform(),
                active: true,
            },
            SceneObject {
                id: ObjectId(2),
                mesh_id: MeshId(1),
                material_id: MaterialId(1),
                transform: translation_transform(1.0, 0.0, 0.0),
                active: true,
            },
        ],
    }
}
