use super::store::SceneStore;
use super::types::{MaterialId, MeshId, SceneObject, SceneVersion};
use crate::core::error::{SoundError, SoundResult};
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
    material_buffer: BufferHandle,
    object_buffer: BufferHandle,
    tlas_id: Option<TlasId>,
    instances: Vec<RtInstanceSpec>,
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
    absorption_scattering: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GpuObject {
    material_index: u32,
    indexed: u32,
    _pad: [u32; 2],
    vertex_address: u64,
    index_address: u64,
    object_to_world: [[f32; 4]; 4],
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
            if uploaded_meshes.contains_key(&mesh.id) {
                continue;
            }

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
        uploaded_meshes.retain(|mesh_id, _| store.meshes.contains_key(mesh_id));

        let mut active_objects = store.objects().filter(|object| object.active).collect::<Vec<_>>();
        active_objects.sort_by_key(|object| object.id);

        let instances = active_objects
            .iter()
            .enumerate()
            .map(|(object_index, object)| {
                let uploaded = uploaded_meshes
                    .get(&object.mesh_id)
                    .ok_or_else(|| SoundError::not_found(format!("Mesh {} is not uploaded", object.mesh_id)))?;
                Ok(RtInstanceSpec {
                    blas_id: uploaded.blas_id,
                    transform: object.transform,
                    custom_index: object_index as u32,
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

        let scene_buffers = upload_scene_buffers(gpu, store, &uploaded_meshes, &active_objects)?;

        Ok(Self {
            version: store.version(),
            uploaded_meshes,
            material_buffer: scene_buffers.material_buffer,
            object_buffer: scene_buffers.object_buffer,
            tlas_id,
            instance_count: instances.len() as u32,
            instances,
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

    /// Returns active scene BLAS instances used to build the scene TLAS.
    pub fn instances(&self) -> &[RtInstanceSpec] {
        &self.instances
    }

    /// Returns compact material data used by GPU scene shaders.
    pub fn material_buffer(&self) -> &BufferHandle {
        &self.material_buffer
    }

    /// Returns compact object data used by GPU scene shaders.
    pub fn object_buffer(&self) -> &BufferHandle {
        &self.object_buffer
    }
}

struct SceneGpuBuffers {
    material_buffer: BufferHandle,
    object_buffer: BufferHandle,
}

fn upload_scene_buffers(
    gpu: &VkBackend,
    store: &SceneStore,
    uploaded_meshes: &HashMap<MeshId, UploadedMesh>,
    active_objects: &[&SceneObject],
) -> SoundResult<SceneGpuBuffers> {
    let (materials, material_indices) = gpu_materials(store);
    let mut objects = Vec::with_capacity(active_objects.len().max(1));

    for object in active_objects {
        let mesh = store
            .meshes
            .get(&object.mesh_id)
            .ok_or_else(|| SoundError::not_found(format!("Mesh {} is not available", object.mesh_id)))?;
        let uploaded = uploaded_meshes
            .get(&object.mesh_id)
            .ok_or_else(|| SoundError::not_found(format!("Mesh {} is not uploaded", object.mesh_id)))?;
        let material_index = *material_indices
            .get(&object.material_id)
            .ok_or_else(|| SoundError::not_found(format!("Material {} is not available", object.material_id)))?;

        objects.push(GpuObject {
            material_index,
            indexed: u32::from(!mesh.indices.is_empty()),
            _pad: [0; 2],
            vertex_address: gpu.memory().buffer_device_address(&uploaded._buffers.vertex_buffer),
            index_address: uploaded
                ._buffers
                .index_buffer
                .as_ref()
                .map_or(0, |buffer| gpu.memory().buffer_device_address(buffer)),
            object_to_world: matrix_to_columns(object.transform),
        });
    }

    if objects.is_empty() {
        objects.push(GpuObject {
            material_index: 0,
            indexed: 0,
            _pad: [0; 2],
            vertex_address: 0,
            index_address: 0,
            object_to_world: matrix_to_columns(glam::Mat4::IDENTITY),
        });
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
    let mut materials = Vec::with_capacity(source_materials.len().max(1));
    for material in source_materials {
        let material_index = materials.len() as u32;
        material_indices.insert(material.id, material_index);
        materials.push(GpuMaterial {
            absorption_scattering: [
                material
                    .absorption_bands
                    .first()
                    .copied()
                    .unwrap_or(0.0)
                    .clamp(0.0, 1.0),
                material.scattering,
                material
                    .transmission
                    .as_ref()
                    .and_then(|bands| bands.first().copied())
                    .unwrap_or(0.0),
                0.0,
            ],
        });
    }
    if materials.is_empty() {
        materials.push(GpuMaterial {
            absorption_scattering: [1.0, 0.0, 0.0, 0.0],
        });
    }

    (materials, material_indices)
}

fn matrix_to_columns(matrix: glam::Mat4) -> [[f32; 4]; 4] {
    matrix.to_cols_array_2d()
}

fn upload_gpu_slice<T: bytemuck::Pod>(gpu: &VkBackend, values: &[T]) -> SoundResult<BufferHandle> {
    let buffer = gpu
        .memory()
        .create_storage_buffer(std::mem::size_of_val(values) as u64)?;
    gpu.memory().upload_typed(&buffer, values)?;
    Ok(buffer)
}
