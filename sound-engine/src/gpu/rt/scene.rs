use super::*;
use crate::core::error::{SoundError, SoundResult};
use glam::{Mat4, Vec3};
use std::mem::size_of;

/// CPU-side triangle mesh data to upload for BLAS construction.
pub struct RtMeshSpec<'a> {
    /// Vertex positions in mesh-local space.
    pub vertices: &'a [Vec3],
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

/// CPU-side procedural AABB data to upload for BLAS construction.
pub struct RtAabbSpec {
    /// Minimum corner in BLAS-local space.
    pub min: Vec3,
    /// Maximum corner in BLAS-local space.
    pub max: Vec3,
    /// Whether the primitive can skip any-hit shaders during traversal.
    pub opaque: bool,
}

/// GPU buffer containing uploaded procedural AABB data for a BLAS build.
pub struct RtAabbBuffers {
    /// Device-addressable AABB buffer consumed during BLAS construction.
    pub aabb_buffer: BufferHandle,
    /// Number of AABB primitives uploaded.
    pub primitive_count: u32,
    /// Opaque flag copied from `RtAabbSpec`.
    pub opaque: bool,
}

/// One TLAS instance referencing a previously built BLAS.
#[derive(Debug, Clone, Copy)]
pub struct RtInstanceSpec {
    /// BLAS to instance into the TLAS.
    pub blas_id: BlasId,
    /// Object-to-world transform for this TLAS instance.
    pub transform: Mat4,
    /// 24-bit custom index visible to shaders.
    pub custom_index: u32,
    /// Visibility mask used during ray traversal.
    pub mask: u8,
    /// Hit-group SBT record offset for this instance.
    pub sbt_record_offset: u32,
}

/// Packed Vulkan AABB build input; six contiguous floats with no implicit padding.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct RtAabbPrimitive {
    min_x: f32,
    min_y: f32,
    min_z: f32,
    max_x: f32,
    max_y: f32,
    max_z: f32,
}

unsafe impl bytemuck::Zeroable for RtAabbPrimitive {}
unsafe impl bytemuck::Pod for RtAabbPrimitive {}

impl RtContext {
    /// Uploads triangle mesh data into device-addressable GPU buffers.
    ///
    /// The returned buffers are suitable as BLAS build inputs.
    pub fn upload_mesh(&self, spec: RtMeshSpec<'_>) -> SoundResult<RtMeshBuffers> {
        validate_triangle_mesh(spec.vertices, spec.indices)?;
        if spec.vertices.is_empty() {
            return Err(SoundError::invalid_argument(
                "RT mesh must contain at least one vertex",
            ));
        }

        let vertex_buffer = self.memory.create_device_address_buffer(
            std::mem::size_of_val(spec.vertices) as vk::DeviceSize,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        self.memory.upload_typed(&vertex_buffer, spec.vertices)?;

        let index_buffer = if let Some(indices) = spec.indices {
            if indices.is_empty() {
                return Err(SoundError::invalid_argument(
                    "RT mesh index slice must not be empty",
                ));
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

    /// Uploads procedural AABB data into a device-addressable GPU buffer.
    ///
    /// The returned buffer is suitable as AABB BLAS build input.
    pub fn upload_aabbs(&self, specs: &[RtAabbSpec]) -> SoundResult<RtAabbBuffers> {
        if specs.len() > u32::MAX as usize {
            return Err(SoundError::invalid_argument(
                "AABB count exceeds Vulkan's 32-bit range",
            ));
        }
        if specs.is_empty() {
            return Err(SoundError::invalid_argument(
                "RT AABB build input must not be empty",
            ));
        }

        let opaque = specs[0].opaque;
        let mut aabbs = Vec::with_capacity(specs.len());
        for spec in specs {
            if spec.opaque != opaque {
                return Err(SoundError::invalid_argument(
                    "RT AABB upload requires a single opaque flag for all primitives",
                ));
            }
            if !spec.min.is_finite() || !spec.max.is_finite() || spec.min.cmpge(spec.max).any() {
                return Err(SoundError::invalid_argument(
                    "RT AABB min/max corners must be finite and ordered",
                ));
            }

            aabbs.push(RtAabbPrimitive {
                min_x: spec.min.x,
                min_y: spec.min.y,
                min_z: spec.min.z,
                max_x: spec.max.x,
                max_y: spec.max.y,
                max_z: spec.max.z,
            });
        }

        let aabb_buffer = self.memory.create_device_address_buffer(
            std::mem::size_of_val(aabbs.as_slice()) as vk::DeviceSize,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        )?;
        self.memory.upload_typed(&aabb_buffer, &aabbs)?;

        Ok(RtAabbBuffers {
            aabb_buffer,
            primitive_count: aabbs.len() as u32,
            opaque,
        })
    }
}

/// Byte stride of the packed Vulkan AABB build input.
pub(crate) fn rt_aabb_stride() -> vk::DeviceSize {
    size_of::<RtAabbPrimitive>() as vk::DeviceSize
}

/// Validates triangle input before GPU traversal can dereference device addresses.
pub(crate) fn validate_triangle_mesh(
    vertices: &[Vec3],
    indices: Option<&[u32]>,
) -> SoundResult<()> {
    if vertices.is_empty()
        || vertices.len() > u32::MAX as usize / 3
        || vertices.iter().any(|vertex| !vertex.is_finite())
    {
        return Err(SoundError::invalid_argument(
            "Mesh vertices must be finite and fit 32-bit shader indexing",
        ));
    }
    let count = indices.map_or(vertices.len(), |values| values.len());
    if count == 0 || !count.is_multiple_of(3) || count > u32::MAX as usize {
        return Err(SoundError::invalid_argument(
            "Mesh must contain complete triangle triples within the 32-bit index range",
        ));
    }
    if indices.is_some_and(|values| values.iter().any(|index| *index as usize >= vertices.len())) {
        return Err(SoundError::invalid_argument(
            "Mesh index references a vertex outside the vertex buffer",
        ));
    }
    Ok(())
}

/// Rejects transforms that cannot be represented by a finite, invertible Vulkan 3x4 matrix.
pub(crate) fn validate_instance_transform(transform: Mat4) -> SoundResult<()> {
    let determinant = glam::Mat3::from_mat4(transform).determinant();
    if !transform.is_finite()
        || !determinant.is_finite()
        || determinant == 0.0
        || transform.row(3) != glam::Vec4::W
    {
        return Err(SoundError::invalid_argument(
            "Instance transform must be finite, affine and invertible",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangle_validation_rejects_out_of_bounds_indices_and_partial_triangles() {
        let vertices = [Vec3::ZERO, Vec3::X, Vec3::Y];
        assert!(validate_triangle_mesh(&vertices, Some(&[0, 1, 2])).is_ok());
        assert!(validate_triangle_mesh(&vertices, Some(&[0, 1, 3])).is_err());
        assert!(validate_triangle_mesh(&vertices, Some(&[0, 1, u32::MAX])).is_err());
        assert!(validate_triangle_mesh(&vertices[..2], None).is_err());
        assert!(validate_triangle_mesh(&[Vec3::NAN, Vec3::X, Vec3::Y], None).is_err());
    }

    #[test]
    fn transforms_reject_non_finite_singular_and_projective_matrices() {
        assert!(validate_instance_transform(Mat4::IDENTITY).is_ok());
        assert!(validate_instance_transform(Mat4::from_scale(Vec3::new(1.0, 0.0, 1.0))).is_err());
        assert!(validate_instance_transform(Mat4::NAN).is_err());
        assert!(validate_instance_transform(Mat4::perspective_rh(1.0, 1.0, 0.1, 100.0)).is_err());
    }
}
