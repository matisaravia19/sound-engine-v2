use super::*;
use crate::error::{SoundError, SoundResult};

/// CPU-side triangle mesh data to upload for BLAS construction.
pub struct RtMeshSpec<'a> {
    /// Vertex positions as tightly packed `vec3<f32>` values.
    pub vertices: &'a [[f32; 3]],
    /// Optional triangle indices. If omitted, vertices are consumed as triples.
    pub indices: Option<&'a [u32]>,
    /// Whether triangles can be treated as opaque by RT shaders.
    pub opaque: bool,
}

/// GPU buffers containing uploaded mesh data for a BLAS build.
pub struct RtMeshBuffers {
    /// Device-addressable vertex buffer.
    pub vertex_buffer: BufferHandle,
    /// Optional device-addressable index buffer.
    pub index_buffer: Option<BufferHandle>,
    /// Number of vertices uploaded.
    pub vertex_count: u32,
    /// Number of indices uploaded, or zero for non-indexed meshes.
    pub index_count: u32,
    /// Opaque flag copied from `RtMeshSpec`.
    pub opaque: bool,
}

/// One TLAS instance referencing a previously built BLAS.
#[derive(Debug, Clone, Copy)]
pub struct RtInstanceSpec {
    /// BLAS to instance into the TLAS.
    pub blas_id: BlasId,
    /// Row-major 3x4 object-to-world transform expected by Vulkan.
    pub transform: [f32; 12],
    /// 24-bit custom index visible to shaders.
    pub custom_index: u32,
    /// Visibility mask used during ray traversal.
    pub mask: u8,
    /// Hit-group SBT record offset for this instance.
    pub sbt_record_offset: u32,
}

impl RtContext {
    /// Uploads triangle mesh data into device-addressable GPU buffers.
    ///
    /// The returned buffers are suitable as BLAS build inputs.
    pub fn upload_mesh(&self, spec: RtMeshSpec<'_>) -> SoundResult<RtMeshBuffers> {
        if spec.vertices.is_empty() {
            return Err(SoundError::invalid_argument("RT mesh must contain at least one vertex"));
        }

        let vertex_buffer = self.memory.create_device_address_buffer(
            std::mem::size_of_val(spec.vertices) as vk::DeviceSize,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        self.memory.upload_typed(&vertex_buffer, spec.vertices)?;

        let index_buffer = if let Some(indices) = spec.indices {
            if indices.is_empty() {
                return Err(SoundError::invalid_argument("RT mesh index slice must not be empty"));
            }

            let buffer = self.memory.create_device_address_buffer(
                std::mem::size_of_val(indices) as vk::DeviceSize,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::STORAGE_BUFFER,
            )?;
            self.memory.upload_typed(&buffer, indices)?;
            Some(buffer)
        } else {
            None
        };

        Ok(RtMeshBuffers {
            vertex_buffer,
            index_buffer,
            vertex_count: spec.vertices.len() as u32,
            index_count: spec.indices.map_or(0, |indices| indices.len() as u32),
            opaque: spec.opaque,
        })
    }
}
