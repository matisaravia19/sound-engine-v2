//! CPU geometry invariants shared by scene editing and GPU uploads.

use super::error::{SoundError, SoundResult};
use glam::{Mat4, Vec3};

/// Checks finite vertices, complete triples, and the engine's 32-bit mesh indexing.
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

/// Requires finite, invertible affine transforms for scene instances.
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
#[path = "../../tests/unit/core/geometry.rs"]
mod tests;
