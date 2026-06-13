use super::store::SceneStore;
use super::types::{MeshId, SceneVersion};
use crate::error::{SoundError, SoundResult};
use crate::gpu::backend::VkBackend;
use crate::gpu::memory::BufferHandle;
use crate::gpu::rt::{BlasBuildSpec, BlasId, RtInstanceSpec, RtMeshBuffers, RtMeshSpec, TlasBuildSpec, TlasId};
use ash::vk;
use std::collections::HashMap;

/// GPU-side mirror of a `SceneStore`.
///
/// The resource owns uploaded mesh buffers, acceleration-structure handles
/// stored in `gpu::rt`, compact material/object buffers, and the latest TLAS id.
/// It must not outlive the `VkBackend` that created its buffers.
pub struct GpuSceneResources {
    version: SceneVersion,
    uploaded_meshes: HashMap<MeshId, UploadedMesh>,
    material_buffer: Option<BufferHandle>,
    object_buffer: Option<BufferHandle>,
    tlas_id: Option<TlasId>,
    instance_count: u32,
}

struct UploadedMesh {
    // Keep mesh buffers alive because BLAS build inputs and future rebuilds
    // depend on their device allocations remaining valid.
    _buffers: RtMeshBuffers,
    blas_id: BlasId,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GpuMaterial {
    absorption: [f32; 4],
    scattering: f32,
    transmission: f32,
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GpuObject {
    object_id: u32,
    material_id: u32,
    mesh_id: u32,
    active: u32,
}

unsafe impl bytemuck::Zeroable for GpuMaterial {}
unsafe impl bytemuck::Pod for GpuMaterial {}
unsafe impl bytemuck::Zeroable for GpuObject {}
unsafe impl bytemuck::Pod for GpuObject {}

impl GpuSceneResources {
    /// Builds a GPU scene mirror, reusing uploaded mesh BLAS data when possible.
    pub(crate) fn build(gpu: &VkBackend, store: &SceneStore, previous: Option<Self>) -> SoundResult<Self> {
        let mut uploaded_meshes = previous.map_or_else(HashMap::new, |resources| resources.uploaded_meshes);
        for mesh in store.meshes() {
            if !uploaded_meshes.contains_key(&mesh.id) {
                let buffers = gpu.rt().upload_mesh(RtMeshSpec {
                    vertices: &mesh.vertices,
                    indices: (!mesh.indices.is_empty()).then_some(mesh.indices.as_slice()),
                    opaque: mesh.opaque,
                })?;
                let blas_id = gpu.rt().build_blas(BlasBuildSpec {
                    mesh: &buffers,
                    flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
                })?;
                uploaded_meshes.insert(
                    mesh.id,
                    UploadedMesh {
                        _buffers: buffers,
                        blas_id,
                    },
                );
            }
        }
        uploaded_meshes.retain(|mesh_id, _| store.meshes.contains_key(mesh_id));

        let instances = store
            .objects()
            .filter(|object| object.active)
            .map(|object| {
                let uploaded = uploaded_meshes
                    .get(&object.mesh_id)
                    .ok_or_else(|| SoundError::not_found(format!("Mesh {} is not uploaded", object.mesh_id.0)))?;
                Ok(RtInstanceSpec {
                    blas_id: uploaded.blas_id,
                    transform: object.transform,
                    custom_index: object.id.0,
                    mask: 0xff,
                    sbt_record_offset: 0,
                })
            })
            .collect::<SoundResult<Vec<_>>>()?;

        // MVP sync policy: rebuild TLAS fully for any scene version change.
        let tlas_id = if instances.is_empty() {
            None
        } else {
            Some(gpu.rt().build_tlas(TlasBuildSpec {
                instances: &instances,
                flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
            })?)
        };

        let material_buffer = upload_materials(gpu, store)?;
        let object_buffer = upload_objects(gpu, store)?;

        Ok(Self {
            version: store.version(),
            uploaded_meshes,
            material_buffer,
            object_buffer,
            tlas_id,
            instance_count: instances.len() as u32,
        })
    }

    /// Returns the scene version represented by these GPU resources.
    pub fn version(&self) -> SceneVersion {
        self.version
    }

    /// Returns the current TLAS id, or `None` when no object is active.
    pub fn tlas_id(&self) -> Option<TlasId> {
        self.tlas_id
    }

    /// Returns the number of active object instances in the TLAS.
    pub fn instance_count(&self) -> u32 {
        self.instance_count
    }

    /// Returns the compact GPU material buffer when the scene has materials.
    pub fn material_buffer(&self) -> Option<&BufferHandle> {
        self.material_buffer.as_ref()
    }

    /// Returns the compact GPU object metadata buffer when the scene has objects.
    pub fn object_buffer(&self) -> Option<&BufferHandle> {
        self.object_buffer.as_ref()
    }
}

fn upload_materials(gpu: &VkBackend, store: &SceneStore) -> SoundResult<Option<BufferHandle>> {
    let materials = store
        .materials()
        .map(|material| {
            let mut absorption = [0.0; 4];
            for (dst, src) in absorption.iter_mut().zip(material.absorption_bands.iter().copied()) {
                *dst = src;
            }
            GpuMaterial {
                absorption,
                scattering: material.scattering,
                transmission: material
                    .transmission
                    .as_ref()
                    .and_then(|bands| bands.first().copied())
                    .unwrap_or(0.0),
                _pad: [0.0; 2],
            }
        })
        .collect::<Vec<_>>();

    if materials.is_empty() {
        return Ok(None);
    }

    let buffer = gpu
        .memory()
        .create_storage_buffer(std::mem::size_of_val(materials.as_slice()) as u64)?;
    gpu.memory().upload_typed(&buffer, &materials)?;
    Ok(Some(buffer))
}

fn upload_objects(gpu: &VkBackend, store: &SceneStore) -> SoundResult<Option<BufferHandle>> {
    let objects = store
        .objects()
        .map(|object| GpuObject {
            object_id: object.id.0,
            material_id: object.material_id.0,
            mesh_id: object.mesh_id.0,
            active: u32::from(object.active),
        })
        .collect::<Vec<_>>();

    if objects.is_empty() {
        return Ok(None);
    }

    let buffer = gpu
        .memory()
        .create_storage_buffer(std::mem::size_of_val(objects.as_slice()) as u64)?;
    gpu.memory().upload_typed(&buffer, &objects)?;
    Ok(Some(buffer))
}
