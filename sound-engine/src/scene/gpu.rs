use super::store::{SceneRevisions, SceneStore};
use super::types::{DiffractionEdge, MaterialId, MeshId, SceneObject, SceneVersion};
use crate::core::error::{SoundError, SoundResult};
use crate::gpu::backend::VkBackend;
use crate::gpu::memory::BufferHandle;
use crate::gpu::rt::{
    AabbBlasBuildSpec, BlasBuildSpec, BlasId, OwnedBlas, OwnedTlas, RtAabbBuffers, RtAabbSpec,
    RtInstanceSpec, RtMeshBuffers, RtMeshSpec, TlasBuildSpec, TlasId,
};
use ash::vk;
use glam::Vec3;
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
    // Keep mesh buffers alive because BLAS build inputs and future rebuilds
    // depend on their device allocations remaining valid.
    _buffers: RtMeshBuffers,
    blas: OwnedBlas,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GpuMaterial {
    // Convert authored energy fractions into broadband pressure gains before upload.
    specular_pressure: f32,
    rough_pressure: f32,
    roughness: f32,
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

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GpuDiffractionEdge {
    start: [f32; 3],
    diffraction_radius: f32,
    end: [f32; 3],
    // std430 keeps the following vec3 aligned to a 16-byte boundary.
    _pad_end: f32,
    edge_dir: [f32; 3],
    edge_length: f32,
    bisector_dir: [f32; 3],
    edge_angle_radians: f32,
    positive_plane_normal: [f32; 3],
    _pad0: f32,
    positive_shadow_dir: [f32; 3],
    _pad1: f32,
    negative_shadow_dir: [f32; 3],
    solid_wedge_cos: f32,
}

unsafe impl bytemuck::Zeroable for GpuMaterial {}
unsafe impl bytemuck::Pod for GpuMaterial {}
unsafe impl bytemuck::Zeroable for GpuObject {}
unsafe impl bytemuck::Pod for GpuObject {}
unsafe impl bytemuck::Zeroable for GpuDiffractionEdge {}
unsafe impl bytemuck::Pod for GpuDiffractionEdge {}

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
                    _buffers: buffers,
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
    _bounds: Option<RtAabbBuffers>,
    blas: Option<OwnedBlas>,
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
        let mesh = store.mesh(object.mesh_id).ok_or_else(|| {
            SoundError::not_found(format!("Mesh {} is not available", object.mesh_id))
        })?;
        let uploaded = uploaded_meshes.get(&object.mesh_id).ok_or_else(|| {
            SoundError::not_found(format!("Mesh {} is not uploaded", object.mesh_id))
        })?;
        let material_index = *material_indices.get(&object.material_id).ok_or_else(|| {
            SoundError::not_found(format!("Material {} is not available", object.material_id))
        })?;

        objects.push(GpuObject {
            material_index,
            indexed: u32::from(!mesh.indices.is_empty()),
            _pad: [0; 2],
            vertex_address: gpu
                .memory()
                .buffer_device_address(&uploaded._buffers.vertex_buffer),
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

// Bound the coherent scalar reflection mixture by the authored pressure loss.
fn gpu_material(material: &super::types::Material) -> GpuMaterial {
    let reflection = 1.0 - material.absorption.clamp(0.0, 1.0);
    let scattering = material.scattering.clamp(0.0, 1.0);
    // The transport sums these lobes as pressure, not independent energies.
    // Normalize their relative amplitudes so branch sampling cannot multiply
    // reflected throughput by sqrt(1-s) + sqrt(s) at every interaction.
    let specular = (1.0 - scattering).sqrt();
    let rough = scattering.sqrt();
    let reflection_pressure = reflection.sqrt() / (specular + rough);
    GpuMaterial {
        specular_pressure: reflection_pressure * specular,
        rough_pressure: reflection_pressure * rough,
        roughness: material.roughness.clamp(0.0, 1.0),
    }
}

fn gpu_materials(store: &SceneStore) -> (Vec<GpuMaterial>, HashMap<MaterialId, u32>) {
    let mut source_materials = store.materials().collect::<Vec<_>>();
    source_materials.sort_by_key(|material| material.id);

    let mut material_indices = HashMap::new();
    let mut materials = Vec::with_capacity(source_materials.len().max(1));
    for material in source_materials {
        let material_index = materials.len() as u32;
        material_indices.insert(material.id, material_index);
        materials.push(gpu_material(material));
    }
    if materials.is_empty() {
        // Descriptor binding 3 must remain valid even for an empty scene. The
        // absorbing fallback cannot be indexed unless malformed geometry exists.
        materials.push(GpuMaterial {
            specular_pressure: 0.0,
            rough_pressure: 0.0,
            roughness: 0.0,
        });
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
    let gpu_edges = if diffraction_edges.is_empty() {
        vec![GpuDiffractionEdge {
            start: [0.0; 3],
            diffraction_radius: 0.0,
            end: [0.0, 1.0, 0.0],
            _pad_end: 0.0,
            edge_dir: [0.0, 1.0, 0.0],
            edge_length: 1.0,
            bisector_dir: [1.0, 0.0, 0.0],
            edge_angle_radians: std::f32::consts::PI,
            positive_plane_normal: [0.0, 0.0, 1.0],
            _pad0: 0.0,
            positive_shadow_dir: [0.0, 0.0, 1.0],
            _pad1: 0.0,
            negative_shadow_dir: [0.0, 0.0, -1.0],
            solid_wedge_cos: 1.0,
        }]
    } else {
        diffraction_edges
            .iter()
            .map(|edge| gpu_diffraction_edge(edge))
            .collect::<Vec<_>>()
    };
    let edge_buffer = upload_gpu_slice(gpu, &gpu_edges)?;

    if diffraction_edges.is_empty() {
        return Ok(DiffractionGpuResources {
            edge_buffer,
            _bounds: None,
            blas: None,
        });
    }

    let aabbs = gpu_edges
        .iter()
        .map(diffraction_edge_aabb)
        .collect::<SoundResult<Vec<_>>>()?;
    let bounds = gpu.rt().upload_aabbs(&aabbs)?;
    // upload_aabbs validates finite ordered bounds and supplies AS input usage.
    let blas_id = unsafe {
        gpu.rt().build_aabb_blas(AabbBlasBuildSpec {
            aabbs: &bounds,
            flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
        })?
    };

    Ok(DiffractionGpuResources {
        edge_buffer,
        _bounds: Some(bounds),
        blas: Some(gpu.rt_context().own_blas(blas_id)),
    })
}

fn diffraction_edge_aabb(edge: &GpuDiffractionEdge) -> SoundResult<RtAabbSpec> {
    let radial = Vec3::from_array(edge.bisector_dir);
    let plane_normal = Vec3::from_array(edge.positive_plane_normal);
    let radius = edge.diffraction_radius;
    // A procedural AABB needs non-zero thickness around its mathematical plane
    // so Vulkan traversal can invoke the exact strip intersection shader.
    let plane_epsilon = radius.max(1.0) * 0.001;
    let offsets = [
        radial * radius + plane_normal * plane_epsilon,
        radial * radius - plane_normal * plane_epsilon,
        -radial * radius + plane_normal * plane_epsilon,
        -radial * radius - plane_normal * plane_epsilon,
    ];

    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for point in [Vec3::from_array(edge.start), Vec3::from_array(edge.end)] {
        for offset in offsets {
            let corner = point + offset;
            min = min.min(corner);
            max = max.max(corner);
        }
    }

    Ok(RtAabbSpec {
        min,
        max,
        opaque: true,
    })
}

fn gpu_diffraction_edge(edge: &DiffractionEdge) -> GpuDiffractionEdge {
    let edge_vector = edge.end - edge.start;
    let edge_length = edge_vector.length();
    let edge_dir = edge_vector / edge_length;
    let bisector_dir = (edge.bisector_dir - edge_dir * edge.bisector_dir.dot(edge_dir)).normalize();
    let positive_plane_normal = bisector_dir.cross(edge_dir).normalize();
    let delta = ((edge.edge_angle_radians - std::f32::consts::PI) * 0.5)
        .clamp(0.0, std::f32::consts::FRAC_PI_2);
    let (sin_delta, cos_delta) = delta.sin_cos();
    let positive_shadow_dir =
        (cos_delta * positive_plane_normal - sin_delta * bisector_dir).normalize();
    let negative_shadow_dir =
        (-cos_delta * positive_plane_normal - sin_delta * bisector_dir).normalize();
    let solid_wedge_cos = ((std::f32::consts::TAU - edge.edge_angle_radians) * 0.5).cos();

    GpuDiffractionEdge {
        start: edge.start.to_array(),
        diffraction_radius: edge.diffraction_radius,
        end: edge.end.to_array(),
        _pad_end: 0.0,
        edge_dir: edge_dir.to_array(),
        edge_length,
        bisector_dir: bisector_dir.to_array(),
        edge_angle_radians: edge.edge_angle_radians,
        positive_plane_normal: positive_plane_normal.to_array(),
        _pad0: 0.0,
        positive_shadow_dir: positive_shadow_dir.to_array(),
        _pad1: 0.0,
        negative_shadow_dir: negative_shadow_dir.to_array(),
        solid_wedge_cos,
    }
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

#[cfg(test)]
#[path = "../../tests/unit/scene/gpu_sync.rs"]
mod sync_tests;

#[cfg(test)]
#[path = "../../tests/unit/scene/gpu.rs"]
mod tests;
