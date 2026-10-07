//! Host records for the scene shader interface, with CPU-only material and edge conversion.

use crate::gpu::rt::RtAabbSpec;
use crate::scene::{DiffractionEdge, Material};
use glam::{Mat4, Vec3};

/// Broadband pressure coefficients matching `SceneMaterial` in the acoustic shaders.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub(super) struct GpuMaterial {
    /// Pressure gain assigned to the smooth or glossy reflection lobe.
    pub(super) specular_pressure: f32,
    /// Pressure gain assigned to the rough reflection lobe.
    pub(super) rough_pressure: f32,
    /// Authored roughness clamped to the shader's unit interval.
    pub(super) roughness: f32,
}

impl From<&Material> for GpuMaterial {
    /// Converts authored energy fractions to normalized reflection pressure gains.
    fn from(material: &Material) -> Self {
        let reflection = 1.0 - material.absorption.clamp(0.0, 1.0);
        let scattering = material.scattering.clamp(0.0, 1.0);
        // The transport sums these lobes as pressure, not independent energies.
        // Normalize their relative amplitudes so branch sampling cannot multiply
        // reflected throughput by sqrt(1-s) + sqrt(s) at every interaction.
        let specular = (1.0 - scattering).sqrt();
        let rough = scattering.sqrt();
        let reflection_pressure = reflection.sqrt() / (specular + rough);
        Self {
            specular_pressure: reflection_pressure * specular,
            rough_pressure: reflection_pressure * rough,
            roughness: material.roughness.clamp(0.0, 1.0),
        }
    }
}

/// Object-table entry matching `SceneObject` in the surface closest-hit shader.
/// The scene owner must retain the buffers behind these borrowed device addresses.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub(super) struct GpuObject {
    /// Index into the compact material table, independent of the authored material ID.
    pub(super) material_index: u32,
    /// Nonzero when triangle vertices are selected through the index buffer.
    pub(super) indexed: u32,
    _pad: [u32; 2],
    /// Address of tightly packed three-float vertex positions.
    pub(super) vertex_address: u64,
    /// Address of 32-bit triangle indices, or zero for non-indexed meshes.
    pub(super) index_address: u64,
    /// Column-major transform used to reconstruct world-space surface normals.
    pub(super) object_to_world: [[f32; 4]; 4],
}

impl GpuObject {
    /// Encodes uploaded geometry addresses and initializes shader alignment padding.
    pub(super) fn new(
        material_index: u32,
        vertex_address: u64,
        index_address: Option<u64>,
        object_to_world: Mat4,
    ) -> Self {
        Self {
            material_index,
            indexed: u32::from(index_address.is_some()),
            _pad: [0; 2],
            vertex_address,
            index_address: index_address.unwrap_or(0),
            object_to_world: object_to_world.to_cols_array_2d(),
        }
    }
}

/// Wedge geometry matching `DiffractionEdge` in the acoustic shaders.
/// Derived directions are precomputed from the validated authored edge before upload.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub(super) struct GpuDiffractionEdge {
    start: [f32; 3],
    /// Half-width in meters of the procedural diffraction strip.
    pub(super) diffraction_radius: f32,
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

impl From<&DiffractionEdge> for GpuDiffractionEdge {
    /// Precomputes shader directions from a validated finite edge and zeroes padding.
    fn from(edge: &DiffractionEdge) -> Self {
        let edge_vector = edge.end - edge.start;
        let edge_length = edge_vector.length();
        let edge_dir = edge_vector / edge_length;
        let bisector_dir =
            (edge.bisector_dir - edge_dir * edge.bisector_dir.dot(edge_dir)).normalize();
        let positive_plane_normal = bisector_dir.cross(edge_dir).normalize();
        let delta = ((edge.edge_angle_radians - std::f32::consts::PI) * 0.5)
            .clamp(0.0, std::f32::consts::FRAC_PI_2);
        let (sin_delta, cos_delta) = delta.sin_cos();
        let positive_shadow_dir =
            (cos_delta * positive_plane_normal - sin_delta * bisector_dir).normalize();
        let negative_shadow_dir =
            (-cos_delta * positive_plane_normal - sin_delta * bisector_dir).normalize();
        let solid_wedge_cos = ((std::f32::consts::TAU - edge.edge_angle_radians) * 0.5).cos();

        Self {
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
}

impl GpuDiffractionEdge {
    /// Encloses the virtual strip with enough thickness for procedural ray traversal.
    pub(super) fn bounds(&self) -> RtAabbSpec {
        let radial = Vec3::from_array(self.bisector_dir);
        let plane_normal = Vec3::from_array(self.positive_plane_normal);
        let radius = self.diffraction_radius;
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
        for point in [Vec3::from_array(self.start), Vec3::from_array(self.end)] {
            for offset in offsets {
                let corner = point + offset;
                min = min.min(corner);
                max = max.max(corner);
            }
        }

        RtAabbSpec {
            min,
            max,
            opaque: true,
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/scene/gpu/records.rs"]
mod tests;
