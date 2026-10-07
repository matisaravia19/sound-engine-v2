//! Scene uploads and acceleration structures, rebuilt by independent resource revisions.

mod records;

use super::store::{SceneRevisions, SceneStore};
use super::types::{DiffractionEdge, MaterialId, MeshId, SceneObject, SceneVersion};
use crate::core::error::{SoundError, SoundResult};
use crate::gpu::backend::VkBackend;
use crate::gpu::memory::BufferHandle;
use crate::gpu::rt::{
    AabbBlasBuildSpec, BlasBuildSpec, BlasId, OwnedBlas, OwnedTlas, RtInstanceSpec, RtMeshBuffers,
    RtMeshSpec, TlasBuildSpec, TlasId,
};
use ash::vk;
use records::{GpuDiffractionEdge, GpuMaterial, GpuObject};
use std::collections::HashMap;

// Physical geometry is the only class allowed to occlude explicit source
// connections. Diffraction proxies use 0x02 when assembled into a query TLAS.
const SURFACE_GEOMETRY_MASK: u8 = 0x01;

/// GPU-side mirror of a `SceneStore`.
///
/// The resource owns uploaded mesh buffers, acceleration-structure handles
/// stored in `gpu::rt`, compact material/object buffers, and the latest TLAS id.
/// Leases retain the RT context and buffers retain the Vulkan device.
pub struct GpuSceneResources {
    version: SceneVersion,
    revisions: SceneRevisions,
    geometry: SurfaceGpuResources,
    shading: SceneGpuBuffers,
    diffraction: DiffractionGpuResources,
}

struct SurfaceGpuResources {
    // Destroy TLAS references before releasing scene BLAS storage.
    tlas: Option<OwnedTlas>,
    uploaded_meshes: HashMap<MeshId, UploadedMesh>,
    instances: Vec<RtInstanceSpec>,
}

struct UploadedMesh {
    revision: SceneVersion,
    // Surface hit shaders read these buffers through the uploaded device addresses,
    // so their allocations must remain alive for the entire scene lifetime.
    buffers: RtMeshBuffers,
    blas: OwnedBlas,
}

impl SurfaceGpuResources {
    fn build(
        gpu: &VkBackend,
        store: &SceneStore,
        mut previous: Option<Self>,
        active_objects: &[&SceneObject],
    ) -> SoundResult<Self> {
        let mut uploaded_meshes = previous.as_mut().map_or_else(HashMap::new, |resources| {
            std::mem::take(&mut resources.uploaded_meshes)
        });
        // Retire the old TLAS before dropping any BLAS it could reference.
        drop(previous);
        uploaded_meshes.retain(|id, uploaded| store.mesh_revision(*id) == Some(uploaded.revision));
        for mesh in store.meshes() {
            if uploaded_meshes.contains_key(&mesh.id) {
                continue;
            }

            let buffers = gpu.rt().upload_mesh(RtMeshSpec {
                vertices: &mesh.vertices,
                indices: (!mesh.indices.is_empty()).then_some(mesh.indices.as_slice()),
                opaque: mesh.opaque,
            })?;

            // Fresh, validated upload remains owned and unchanged through the build.
            let blas_id = unsafe {
                gpu.rt().build_blas(BlasBuildSpec {
                    mesh: &buffers,
                    flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
                })?
            };

            uploaded_meshes.insert(
                mesh.id,
                UploadedMesh {
                    revision: store
                        .mesh_revision(mesh.id)
                        .expect("Mesh came from this scene store"),
                    buffers,
                    blas: gpu.rt_context().own_blas(blas_id),
                },
            );
        }

        let instances = active_objects
            .iter()
            .enumerate()
            .map(|(object_index, object)| {
                let uploaded = uploaded_meshes.get(&object.mesh_id).ok_or_else(|| {
                    SoundError::not_found(format!("Mesh {} is not uploaded", object.mesh_id))
                })?;
                Ok(RtInstanceSpec {
                    blas_id: uploaded.blas.id,
                    transform: object.transform,
                    custom_index: object_index as u32,
                    mask: SURFACE_GEOMETRY_MASK,
                    sbt_record_offset: 0,
                })
            })
            .collect::<SoundResult<Vec<_>>>()?;

        let tlas = if instances.is_empty() {
            None
        } else {
            Some(
                gpu.rt_context()
                    .own_tlas(gpu.rt().build_tlas(TlasBuildSpec {
                        instances: &instances,
                        flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
                    })?),
            )
        };

        Ok(Self {
            uploaded_meshes,
            tlas,
            instances,
        })
    }
}

impl GpuSceneResources {
    /// Rebuilds changed resource groups, retaining independent geometry and buffers.
    pub(crate) fn build(
        gpu: &VkBackend,
        store: &SceneStore,
        previous: Option<Self>,
    ) -> SoundResult<Self> {
        let revisions = store.revisions();
        let previous_revisions = previous.as_ref().map(|resources| resources.revisions);
        let (previous_geometry, previous_shading, previous_diffraction) = match previous {
            Some(resources) => (
                Some(resources.geometry),
                Some(resources.shading),
                Some(resources.diffraction),
            ),
            None => (None, None, None),
        };
        let mut active_objects = store
            .objects()
            .filter(|object| object.active)
            .collect::<Vec<_>>();
        active_objects.sort_by_key(|object| object.id);
        let geometry = match previous_geometry {
            Some(geometry)
                if previous_revisions.is_some_and(|old| old.geometry == revisions.geometry) =>
            {
                geometry
            }
            previous => SurfaceGpuResources::build(gpu, store, previous, &active_objects)?,
        };
        let shading = match previous_shading {
            Some(shading)
                if previous_revisions.is_some_and(|old| old.shading == revisions.shading) =>
            {
                shading
            }
            _ => upload_scene_buffers(gpu, store, &geometry.uploaded_meshes, &active_objects)?,
        };
        let diffraction = match previous_diffraction {
            Some(diffraction)
                if previous_revisions
                    .is_some_and(|old| old.diffraction == revisions.diffraction) =>
            {
                diffraction
            }
            _ => upload_diffraction_edges(gpu, &diffraction_edges(store))?,
        };
        Ok(Self {
            version: store.version(),
            revisions,
            geometry,
            shading,
            diffraction,
        })
    }

    /// Returns the scene version represented by these GPU resources.
    pub fn version(&self) -> SceneVersion {
        self.version
    }

    /// Returns the current TLAS id, or `None` when no object is active.
    pub fn tlas_id(&self) -> Option<TlasId> {
        self.geometry.tlas.as_ref().map(|tlas| tlas.id)
    }

    /// Returns the number of active object instances in the TLAS.
    pub fn instance_count(&self) -> u32 {
        self.geometry.instances.len() as u32
    }

    /// Returns active scene BLAS instances used to build the scene TLAS.
    pub fn instances(&self) -> &[RtInstanceSpec] {
        &self.geometry.instances
    }

    /// Returns compact material data used by GPU scene shaders.
    pub fn material_buffer(&self) -> &BufferHandle {
        &self.shading.material_buffer
    }

    /// Returns compact object data used by GPU scene shaders.
    pub fn object_buffer(&self) -> &BufferHandle {
        &self.shading.object_buffer
    }

    /// Returns compact diffraction edge data used by GPU diffraction shaders.
    pub fn diffraction_edge_buffer(&self) -> &BufferHandle {
        &self.diffraction.edge_buffer
    }

    /// Returns the procedural diffraction-edge BLAS when the scene has edges.
    pub fn diffraction_blas_id(&self) -> Option<BlasId> {
        self.diffraction.blas.as_ref().map(|blas| blas.id)
    }
}

struct SceneGpuBuffers {
    material_buffer: BufferHandle,
    object_buffer: BufferHandle,
}

struct DiffractionGpuResources {
    edge_buffer: BufferHandle,
    blas: Option<OwnedBlas>,
}

fn upload_scene_buffers(
    gpu: &VkBackend,
    store: &SceneStore,
    uploaded_meshes: &HashMap<MeshId, UploadedMesh>,
    active_objects: &[&SceneObject],
) -> SoundResult<SceneGpuBuffers> {
    let (materials, material_indices) = gpu_materials(store);
    let mut objects = Vec::with_capacity(active_objects.len());

    for object in active_objects {
        let uploaded = uploaded_meshes.get(&object.mesh_id).ok_or_else(|| {
            SoundError::not_found(format!("Mesh {} is not uploaded", object.mesh_id))
        })?;
        let material_index = *material_indices.get(&object.material_id).ok_or_else(|| {
            SoundError::not_found(format!("Material {} is not available", object.material_id))
        })?;

        objects.push(GpuObject::new(
            material_index,
            uploaded.buffers.vertex_buffer.device_address(),
            uploaded
                .buffers
                .index_buffer
                .as_ref()
                .map(BufferHandle::device_address),
            object.transform,
        ));
    }

    let material_buffer = upload_gpu_slice(gpu, &materials)?;
    let object_buffer = upload_gpu_slice(gpu, &objects)?;
    Ok(SceneGpuBuffers {
        material_buffer,
        object_buffer,
    })
}

fn gpu_materials(store: &SceneStore) -> (Vec<GpuMaterial>, HashMap<MaterialId, u32>) {
    let mut source_materials = store.materials().collect::<Vec<_>>();
    source_materials.sort_by_key(|material| material.id);

    let mut material_indices = HashMap::new();
    let mut materials = Vec::with_capacity(source_materials.len());
    for material in source_materials {
        let material_index = materials.len() as u32;
        material_indices.insert(material.id, material_index);
        materials.push(GpuMaterial::from(material));
    }

    (materials, material_indices)
}

fn diffraction_edges(store: &SceneStore) -> Vec<&DiffractionEdge> {
    let mut edges = store.diffraction_edges().collect::<Vec<_>>();
    edges.sort_by_key(|edge| edge.id);
    edges
}

fn upload_diffraction_edges(
    gpu: &VkBackend,
    diffraction_edges: &[&DiffractionEdge],
) -> SoundResult<DiffractionGpuResources> {
    let gpu_edges = diffraction_edges
        .iter()
        .map(|edge| GpuDiffractionEdge::from(*edge))
        .collect::<Vec<_>>();
    let edge_buffer = upload_gpu_slice(gpu, &gpu_edges)?;

    if diffraction_edges.is_empty() {
        return Ok(DiffractionGpuResources {
            edge_buffer,
            blas: None,
        });
    }

    let aabbs = gpu_edges
        .iter()
        .map(GpuDiffractionEdge::bounds)
        .collect::<Vec<_>>();
    let bounds = gpu.rt().upload_aabbs(&aabbs)?;
    // The uploaded bounds are build inputs only. The synchronous build finishes
    // before they drop; intersection shaders read edge_buffer during tracing.
    let blas_id = unsafe {
        gpu.rt().build_aabb_blas(AabbBlasBuildSpec {
            aabbs: &bounds,
            flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
        })?
    };

    Ok(DiffractionGpuResources {
        edge_buffer,
        blas: Some(gpu.rt_context().own_blas(blas_id)),
    })
}

fn upload_gpu_slice<T: bytemuck::Pod>(gpu: &VkBackend, values: &[T]) -> SoundResult<BufferHandle> {
    // Empty scene tables still need a valid descriptor buffer. No active instance
    // can index the zero record; keeping it here avoids fabricated scene data.
    let empty = [T::zeroed()];
    let values = if values.is_empty() {
        &empty
    } else {
        values
    };
    let buffer = gpu
        .memory()
        .create_storage_buffer(std::mem::size_of_val(values) as u64)?;
    gpu.memory().upload_typed(&buffer, values)?;
    Ok(buffer)
}

#[cfg(test)]
#[path = "../../tests/unit/scene/gpu_sync.rs"]
mod sync_tests;
