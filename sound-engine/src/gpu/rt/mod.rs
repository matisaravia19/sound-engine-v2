use crate::gpu::backend::VkDeviceContext;
use crate::gpu::compute::ComputeContext;
use crate::gpu::descriptor::DescriptorPools;
use crate::gpu::memory::{BufferHandle, GpuAllocator};
use crate::gpu::shader::ShaderLibrary;
use ash::vk;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

mod acceleration;
mod dispatch;
mod pipeline;
mod scene;

pub use acceleration::{AabbBlasBuildSpec, BlasBuildSpec, BlasId, TlasBuildSpec, TlasId};
pub use dispatch::{RtDescriptorWrite, RtTraceSpec};
pub use pipeline::{
    RtDescriptorBindingSpec, RtPipelineId, RtPipelineSpec, RtPushConstantSpec, RtShaderGroupSpec,
    RtShaderStageSpec,
};
pub use scene::{RtAabbBuffers, RtAabbSpec, RtInstanceSpec, RtMeshBuffers, RtMeshSpec};
pub(crate) use scene::{validate_instance_transform, validate_triangle_mesh};

/// Owns Vulkan ray tracing resources for the backend.
///
/// Acceleration structures and RT pipelines are stored by small IDs so callers
/// can build scenes and record traces without managing raw Vulkan lifetimes.
pub struct RtContext {
    next_blas_id: AtomicU32,
    next_tlas_id: AtomicU32,
    next_pipeline_id: AtomicU32,
    device_context: Arc<VkDeviceContext>,
    memory: Arc<GpuAllocator>,
    compute: Arc<ComputeContext>,
    shaders: Arc<ShaderLibrary>,
    blas: Mutex<HashMap<BlasId, Arc<AccelerationStructureHandle>>>,
    tlas: Mutex<HashMap<TlasId, Arc<AccelerationStructureHandle>>>,
    pipelines: Mutex<HashMap<RtPipelineId, RayTracingPipeline>>,
}

struct AccelerationStructureHandle {
    handle: vk::AccelerationStructureKHR,
    device_address: vk::DeviceAddress,
    _storage: BufferHandle,
    // TLAS owners retain each distinct referenced BLAS; BLAS records have no children.
    _referenced_blas: Vec<Arc<AccelerationStructureHandle>>,
    device_context: Arc<VkDeviceContext>,
}

struct RayTracingPipeline {
    device_context: Arc<VkDeviceContext>,
    descriptor_bindings: Vec<vk::DescriptorSetLayoutBinding<'static>>,
    handle: vk::Pipeline,
    layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    push_constant_ranges: Vec<vk::PushConstantRange>,
    descriptor_pools: DescriptorPools,
    _sbt: BufferHandle,
    raygen_region: vk::StridedDeviceAddressRegionKHR,
    miss_region: vk::StridedDeviceAddressRegionKHR,
    hit_region: vk::StridedDeviceAddressRegionKHR,
}

impl RtContext {
    /// Creates an RT context sharing backend-owned device, memory, shader, and queue state.
    pub(super) fn new(
        device_context: Arc<VkDeviceContext>,
        memory: Arc<GpuAllocator>,
        shaders: Arc<ShaderLibrary>,
        compute: Arc<ComputeContext>,
    ) -> Self {
        Self {
            next_blas_id: AtomicU32::new(0),
            next_tlas_id: AtomicU32::new(0),
            next_pipeline_id: AtomicU32::new(0),
            device_context,
            memory,
            compute,
            shaders,
            blas: Mutex::new(HashMap::new()),
            tlas: Mutex::new(HashMap::new()),
            pipelines: Mutex::new(HashMap::new()),
        }
    }
}

impl Drop for RtContext {
    fn drop(&mut self) {
        let _ = self.compute.wait_for_all();

        self.pipelines
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clear();
        self.tlas
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clear();
        self.blas
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clear();
    }
}

impl Drop for RayTracingPipeline {
    fn drop(&mut self) {
        unsafe {
            self.device_context
                .device
                .destroy_pipeline(self.handle, None);
            self.device_context
                .device
                .destroy_pipeline_layout(self.layout, None);
            self.device_context
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

/// Releases an acoustic pipeline and its SBT after its synchronous owner finishes.
pub(crate) struct OwnedRtPipeline {
    /// Pipeline ID referenced by traces from this owner.
    pub(crate) id: RtPipelineId,
    context: Arc<RtContext>,
}

impl Drop for OwnedRtPipeline {
    fn drop(&mut self) {
        self.context
            .pipelines
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remove(&self.id);
    }
}

/// Removes a scene BLAS from the registry when its owner is replaced.
/// Existing TLASes retain its storage until their own references are released.
pub(crate) struct OwnedBlas {
    /// Registry identifier used by TLAS instances.
    pub(crate) id: BlasId,
    context: Arc<RtContext>,
}

/// Releases a scene or query TLAS on success, error, and scene replacement.
/// All referencing submissions must finish before this lease is dropped.
pub(crate) struct OwnedTlas {
    /// Registry identifier used by trace descriptors.
    pub(crate) id: TlasId,
    context: Arc<RtContext>,
}

impl RtContext {
    /// Transfers a private, synchronously used pipeline to a scoped owner.
    pub(crate) fn own_pipeline(self: &Arc<Self>, id: RtPipelineId) -> OwnedRtPipeline {
        OwnedRtPipeline {
            id,
            context: self.clone(),
        }
    }

    /// Returns the live pipeline count for lifetime regression tests.
    #[cfg(test)]
    pub(crate) fn pipeline_count(&self) -> usize {
        self.pipelines.lock().unwrap().len()
    }

    /// Returns live AS registry counts for hardware lifetime regression tests.
    #[cfg(test)]
    pub(crate) fn resource_counts(&self) -> (usize, usize) {
        (
            self.blas.lock().unwrap().len(),
            self.tlas.lock().unwrap().len(),
        )
    }

    /// Transfers responsibility for releasing a newly built BLAS to a lease.
    pub(crate) fn own_blas(self: &Arc<Self>, id: BlasId) -> OwnedBlas {
        OwnedBlas {
            id,
            context: self.clone(),
        }
    }

    /// Transfers responsibility for releasing a newly built TLAS to a lease.
    pub(crate) fn own_tlas(self: &Arc<Self>, id: TlasId) -> OwnedTlas {
        OwnedTlas {
            id,
            context: self.clone(),
        }
    }
}

impl Drop for OwnedBlas {
    fn drop(&mut self) {
        self.context
            .blas
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remove(&self.id);
    }
}

impl Drop for OwnedTlas {
    fn drop(&mut self) {
        self.context
            .tlas
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remove(&self.id);
    }
}

impl Drop for AccelerationStructureHandle {
    fn drop(&mut self) {
        unsafe {
            self.device_context
                .acceleration_structure
                .destroy_acceleration_structure(self.handle, None);
        }
    }
}

fn align_up(value: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two());
    (value + alignment - 1) & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::align_up;

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
        scene.sync_gpu_if_needed(&gpu).unwrap();
        let first_blas = scene.sync_gpu_if_needed(&gpu).unwrap().instances()[0].blas_id;
        for step in 1..9 {
            let mut replacement = mesh.clone();
            replacement.vertices[1].x += step as f32 * 0.1;
            scene
                .apply_updates(SceneUpdates {
                    updates: vec![SceneUpdate::ReplaceMesh(replacement)],
                })
                .unwrap();
            let rebuilt = scene.sync_gpu_if_needed(&gpu).unwrap();
            assert_ne!(rebuilt.instances()[0].blas_id, first_blas);
            assert_eq!(gpu.rt().resource_counts(), (2, 1));
        }
        drop(scene);
        assert_eq!(gpu.rt().resource_counts(), (0, 0));
    }

    #[test]
    fn align_up_keeps_aligned_values() {
        assert_eq!(align_up(64, 32), 64);
    }

    #[test]
    fn align_up_rounds_to_next_boundary() {
        assert_eq!(align_up(65, 32), 96);
    }
}
