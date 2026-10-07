use super::*;
use crate::scene::*;
use glam::Mat4;

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn audit_scene_sync_reuses_independent_geometry_and_updates_gpu_records() {
    let gpu = VkBackend::new().unwrap();
    let mut scene = SceneManager::new();
    let mesh = MeshAsset {
        id: 1,
        vertices: vec![Vec3::ZERO, Vec3::X, Vec3::Y],
        indices: vec![],
        opaque: true,
    };
    let material = Material {
        id: 1,
        absorption: 0.2,
        scattering: 0.0,
        roughness: 0.0,
    };
    let object = SceneObject {
        id: 10,
        mesh_id: 1,
        material_id: 1,
        transform: Mat4::IDENTITY,
        active: true,
    };
    let mut edge = DiffractionEdge {
        id: 1,
        start: Vec3::ZERO,
        end: Vec3::Y,
        bisector_dir: Vec3::X,
        edge_angle_radians: std::f32::consts::TAU,
        diffraction_radius: 0.1,
    };
    scene
        .load(SceneDescription {
            meshes: vec![mesh.clone()],
            materials: vec![material.clone()],
            objects: vec![object.clone()],
            diffraction_edges: vec![edge.clone()],
        })
        .unwrap();
    let initial = scene.sync_gpu_if_needed(&gpu).unwrap();
    let tlas = initial.tlas_id();
    let blas = initial.instances()[0].blas_id;
    let diffraction = initial.diffraction_blas_id();
    let edge_buffer = initial.diffraction_edge_buffer().buffer;

    let mut absorbing = material.clone();
    absorbing.absorption = 0.75;
    let resources = update(
        &gpu,
        &mut scene,
        vec![SceneUpdate::ReplaceMaterial(absorbing)],
    );
    assert_eq!(resources.tlas_id(), tlas);
    assert_eq!(resources.instances()[0].blas_id, blas);
    assert_eq!(resources.diffraction_blas_id(), diffraction);
    assert_eq!(resources.diffraction_edge_buffer().buffer, edge_buffer);
    assert_eq!(
        read_records::<GpuMaterial>(&gpu, resources.material_buffer())[0].specular_pressure,
        0.5
    );

    // Inserting/removing a lower material ID shifts sorted GPU indices even
    // though no instance moves; a material reassignment also changes that table.
    let lower = Material {
        id: 0,
        ..material.clone()
    };
    let resources = update(&gpu, &mut scene, vec![SceneUpdate::AddMaterial(lower)]);
    assert_eq!(resources.tlas_id(), tlas);
    assert_eq!(
        read_records::<GpuObject>(&gpu, resources.object_buffer())[0].material_index,
        1
    );
    let resources = update(
        &gpu,
        &mut scene,
        vec![
            SceneUpdate::SetMaterial {
                id: 10,
                material_id: 0,
            },
            SceneUpdate::RemoveMaterial(1),
        ],
    );
    assert_eq!(resources.tlas_id(), tlas);
    assert_eq!(
        read_records::<GpuObject>(&gpu, resources.object_buffer())[0].material_index,
        0
    );

    let transform = Mat4::from_translation(Vec3::X * 4.0);
    let resources = update(
        &gpu,
        &mut scene,
        vec![SceneUpdate::SetTransform {
            id: 10,
            transform,
        }],
    );
    assert_ne!(resources.tlas_id(), tlas);
    assert_eq!(resources.instances()[0].blas_id, blas);
    assert_eq!(resources.diffraction_blas_id(), diffraction);
    assert_eq!(
        read_records::<GpuObject>(&gpu, resources.object_buffer())[0].object_to_world,
        transform.to_cols_array_2d()
    );
    let translated_tlas = resources.tlas_id();
    let mut replacement = mesh;
    replacement.vertices[1].x = 2.0;
    let resources = update(
        &gpu,
        &mut scene,
        vec![SceneUpdate::ReplaceMesh(replacement)],
    );
    assert_ne!(resources.instances()[0].blas_id, blas);
    assert_ne!(resources.tlas_id(), translated_tlas);
    assert_eq!(resources.diffraction_blas_id(), diffraction);
    assert_eq!(
        read_records::<GpuObject>(&gpu, resources.object_buffer())[0].vertex_address,
        gpu.memory().buffer_device_address(
            &resources.geometry.uploaded_meshes[&1]
                ._buffers
                .vertex_buffer
        )
    );
    let tlas = resources.tlas_id();
    let object_buffer = resources.object_buffer().buffer;
    let material_buffer = resources.material_buffer().buffer;

    edge.diffraction_radius = 0.3;
    let resources = update(
        &gpu,
        &mut scene,
        vec![SceneUpdate::ReplaceDiffractionEdge(edge.clone())],
    );
    assert_eq!(resources.tlas_id(), tlas);
    assert_eq!(resources.object_buffer().buffer, object_buffer);
    assert_eq!(resources.material_buffer().buffer, material_buffer);
    assert_ne!(resources.diffraction_blas_id(), diffraction);
    assert_eq!(
        read_records::<GpuDiffractionEdge>(&gpu, resources.diffraction_edge_buffer())[0]
            .diffraction_radius,
        0.3
    );
    let resources = update(
        &gpu,
        &mut scene,
        vec![
            SceneUpdate::AddDiffractionEdge(DiffractionEdge {
                id: 2,
                ..edge
            }),
            SceneUpdate::RemoveDiffractionEdge(1),
        ],
    );
    assert_eq!(resources.tlas_id(), tlas);
    assert_eq!(
        resources.diffraction_edge_buffer().size,
        size_of::<GpuDiffractionEdge>() as u64
    );
    let resources = update(
        &gpu,
        &mut scene,
        vec![SceneUpdate::RemoveDiffractionEdge(2)],
    );
    assert_eq!(resources.tlas_id(), tlas);
    assert_eq!(resources.diffraction_blas_id(), None);

    let resources = update(
        &gpu,
        &mut scene,
        vec![SceneUpdate::SetActive {
            id: 10,
            active: false,
        }],
    );
    assert_eq!(resources.tlas_id(), None);
    assert_eq!(resources.instance_count(), 0);
    let resources = update(
        &gpu,
        &mut scene,
        vec![SceneUpdate::ReplaceObject(SceneObject {
            material_id: 0,
            ..object.clone()
        })],
    );
    assert!(resources.tlas_id().is_some());
    assert_eq!(resources.instance_count(), 1);
    let resources = update(
        &gpu,
        &mut scene,
        vec![SceneUpdate::AddObject(SceneObject {
            id: 5,
            material_id: 0,
            ..object
        })],
    );
    assert_eq!(
        resources
            .instances()
            .iter()
            .map(|instance| instance.custom_index)
            .collect::<Vec<_>>(),
        [0, 1]
    );
    let resources = update(
        &gpu,
        &mut scene,
        vec![
            SceneUpdate::RemoveObject(5),
            SceneUpdate::RemoveObject(10),
            SceneUpdate::RemoveMesh(1),
        ],
    );
    assert_eq!(resources.tlas_id(), None);
    assert_eq!(gpu.rt().resource_counts(), (0, 0));
    drop(scene);
    assert_eq!(
        gpu.memory().live_allocation_stats().unwrap(),
        (1, 16 * 1024 * 1024)
    );
}

fn update<'a>(
    gpu: &VkBackend,
    scene: &'a mut SceneManager,
    updates: Vec<SceneUpdate>,
) -> &'a GpuSceneResources {
    scene
        .apply_updates(SceneUpdates {
            updates,
        })
        .unwrap();
    scene.sync_gpu_if_needed(gpu).unwrap()
}

fn read_records<T: bytemuck::Pod>(gpu: &VkBackend, buffer: &BufferHandle) -> Vec<T> {
    let count = buffer.size as usize / size_of::<T>();
    let mut records = vec![T::zeroed(); count];
    gpu.memory().download_typed(buffer, &mut records).unwrap();
    records
}
