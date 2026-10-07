use super::RtContext;

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn audit_tlas_retains_blas_after_registry_owner_is_released() {
    use super::*;
    use crate::gpu::backend::VkBackend;
    use glam::{Mat4, Vec3};
    let gpu = VkBackend::new().unwrap();
    let mesh = gpu
        .rt()
        .upload_mesh(RtMeshSpec {
            vertices: &[Vec3::ZERO, Vec3::X, Vec3::Y],
            indices: None,
            opaque: true,
        })
        .unwrap();
    // This validated, immutable upload stays alive throughout the build.
    let id = unsafe {
        gpu.rt().build_blas(BlasBuildSpec {
            mesh: &mesh,
            flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
        })
    }
    .unwrap();
    let owner = gpu.rt_context().own_blas(id);
    let weak_blas = Arc::downgrade(gpu.rt().blas.lock().unwrap().get(&id).unwrap());
    let instance = RtInstanceSpec {
        blas_id: id,
        transform: Mat4::IDENTITY,
        custom_index: 0,
        mask: 1,
        sbt_record_offset: 0,
    };
    for flags in [
        vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
            | vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_BUILD,
        vk::BuildAccelerationStructureFlagsKHR::MOTION_NV,
    ] {
        assert!(
            gpu.rt()
                .build_tlas(TlasBuildSpec {
                    instances: &[instance],
                    flags,
                })
                .is_err()
        );
    }
    let tlas = gpu
        .rt()
        .build_tlas(TlasBuildSpec {
            instances: &[instance, instance],
            flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
        })
        .unwrap();
    assert_eq!(
        gpu.rt().tlas.lock().unwrap()[&tlas]._referenced_blas.len(),
        1
    );
    drop(owner);
    drop(mesh);
    assert_eq!(gpu.rt().resource_counts(), (0, 1));
    assert!(weak_blas.upgrade().is_some());
    // This test owns the TLAS and has never submitted a trace that uses it.
    unsafe { gpu.rt().destroy_tlas(tlas) }.unwrap();
    assert!(weak_blas.upgrade().is_none());
}

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn audit_scene_rebuilds_release_old_acceleration_structures() {
    use crate::gpu::backend::VkBackend;
    use crate::scene::*;
    use glam::{Mat4, Vec3};
    let gpu = VkBackend::new().unwrap();
    let mut scene = SceneManager::new();
    let mesh = MeshAsset {
        id: 1,
        vertices: vec![Vec3::ZERO, Vec3::X, Vec3::Y],
        indices: vec![],
        opaque: true,
    };
    scene
        .load(SceneDescription {
            meshes: vec![mesh.clone()],
            materials: vec![Material {
                id: 1,
                absorption: 0.2,
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
            diffraction_edges: vec![DiffractionEdge {
                id: 1,
                start: Vec3::ZERO,
                end: Vec3::Y,
                bisector_dir: Vec3::X,
                edge_angle_radians: std::f32::consts::TAU,
                diffraction_radius: 0.1,
            }],
        })
        .unwrap();
    let first_tlas = scene.sync_gpu_if_needed(&gpu).unwrap().tlas_id();
    for step in 1..9 {
        let mut replacement = mesh.clone();
        replacement.vertices[1].x += step as f32 * 0.1;
        scene
            .apply_updates(SceneUpdates {
                updates: vec![SceneUpdate::ReplaceMesh(replacement)],
            })
            .unwrap();
        let rebuilt = scene.sync_gpu_if_needed(&gpu).unwrap();
        assert_ne!(rebuilt.tlas_id(), first_tlas);
        assert_eq!(gpu.rt().resource_counts(), (2, 1));
    }
    drop(scene);
    assert_eq!(gpu.rt().resource_counts(), (0, 0));
}

impl RtContext {
    /// Returns the live pipeline count for lifetime regression tests.
    pub(crate) fn pipeline_count(&self) -> usize {
        self.pipelines.lock().unwrap().len()
    }

    /// Returns live AS registry counts for hardware lifetime regression tests.
    pub(crate) fn resource_counts(&self) -> (usize, usize) {
        (
            self.blas.lock().unwrap().len(),
            self.tlas.lock().unwrap().len(),
        )
    }
}
