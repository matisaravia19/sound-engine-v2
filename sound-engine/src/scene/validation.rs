//! Validation of authored records before scene storage or incremental replacement.

use super::types::{DiffractionEdge, Material, MeshAsset};
use crate::core::error::{SoundError, SoundResult};

/// Checks complete triangle data and valid indices before storing a mesh.
pub(super) fn validate_mesh(mesh: &MeshAsset) -> SoundResult<()> {
    crate::core::geometry::validate_triangle_mesh(
        &mesh.vertices,
        (!mesh.indices.is_empty()).then_some(mesh.indices.as_slice()),
    )
}

/// Checks finite acoustic coefficients in the authored unit interval.
pub(super) fn validate_material(material: &Material) -> SoundResult<()> {
    for (name, value) in [
        ("absorption", material.absorption),
        ("scattering", material.scattering),
        ("roughness", material.roughness),
    ] {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(SoundError::invalid_argument(format!(
                "Material {} {name} must be finite and in [0, 1]",
                material.id
            )));
        }
    }
    Ok(())
}

/// Checks finite edge geometry whose direction and bisector can be normalized in f32.
pub(super) fn validate_diffraction_edge(edge: &DiffractionEdge) -> SoundResult<()> {
    if !edge.start.is_finite() || !edge.end.is_finite() {
        return Err(SoundError::invalid_argument(format!(
            "Diffraction edge {} endpoints must be finite",
            edge.id
        )));
    }
    // Finite endpoints can still overflow when subtracted or squared for normalization.
    let direction = edge.end - edge.start;
    let direction_length_squared = direction.length_squared();
    if !direction_length_squared.is_finite() || direction_length_squared <= f32::EPSILON {
        return Err(SoundError::invalid_argument(format!(
            "Diffraction edge {} direction must have finite, nonzero length",
            edge.id
        )));
    }
    let bisector_length_squared = edge.bisector_dir.length_squared();
    if !bisector_length_squared.is_finite() || bisector_length_squared <= f32::EPSILON {
        return Err(SoundError::invalid_argument(format!(
            "Diffraction edge {} bisector must be finite and nonzero",
            edge.id
        )));
    }
    let edge_dir = direction.normalize();
    let bisector = edge.bisector_dir.normalize();
    if edge_dir.cross(bisector).length_squared() <= 1.0e-6 {
        return Err(SoundError::invalid_argument(format!(
            "Diffraction edge {} bisector must not be parallel to the edge",
            edge.id
        )));
    }
    if !edge.edge_angle_radians.is_finite()
        || !(std::f32::consts::PI..=std::f32::consts::TAU).contains(&edge.edge_angle_radians)
    {
        return Err(SoundError::invalid_argument(format!(
            "Diffraction edge {} angle must be finite and in [PI, 2*PI]",
            edge.id
        )));
    }
    if !edge.diffraction_radius.is_finite() || edge.diffraction_radius <= 0.0 {
        return Err(SoundError::invalid_argument(format!(
            "Diffraction edge {} radius must be finite and > 0",
            edge.id
        )));
    }
    Ok(())
}
