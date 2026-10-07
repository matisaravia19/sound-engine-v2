use super::*;
use std::mem::{align_of, size_of};

#[test]
fn uploaded_reflection_mixture_has_no_scattering_gain() {
    for absorption in [0.0_f32, 0.2, 0.75, 1.0] {
        for scattering in [0.0_f32, 0.4, 1.0] {
            let material = Material {
                id: 1,
                absorption,
                scattering,
                roughness: 0.2,
            };
            let gpu = GpuMaterial::from(&material);
            let reflected = (gpu.specular_pressure + gpu.rough_pressure).powi(2);
            assert!((reflected - (1.0 - absorption)).abs() < 1e-6);
            let separate_energy = gpu.specular_pressure.powi(2) + gpu.rough_pressure.powi(2);
            assert!(separate_energy <= reflected + 1e-6);
            assert!((gpu.rough_pressure.powi(2) - separate_energy * scattering).abs() < 1e-6);
        }
    }
}

#[test]
fn material_gpu_record_matches_std430_layout() {
    assert_eq!(size_of::<GpuMaterial>(), 12);
    assert_eq!(align_of::<GpuMaterial>(), 4);
    assert_eq!(std::mem::offset_of!(GpuMaterial, roughness), 8);
}

#[test]
fn diffraction_edge_gpu_record_matches_std430_layout() {
    assert_eq!(size_of::<GpuDiffractionEdge>(), 112);
    assert_eq!(align_of::<GpuDiffractionEdge>(), 4);
    assert_eq!(std::mem::offset_of!(GpuDiffractionEdge, end), 16);
    assert_eq!(std::mem::offset_of!(GpuDiffractionEdge, edge_dir), 32);
    assert_eq!(
        std::mem::offset_of!(GpuDiffractionEdge, positive_plane_normal),
        64
    );
    assert_eq!(
        std::mem::offset_of!(GpuDiffractionEdge, solid_wedge_cos),
        108
    );
}

#[test]
fn object_gpu_record_matches_std430_layout_and_optional_index_address() {
    assert_eq!(size_of::<GpuObject>(), 96);
    assert_eq!(std::mem::offset_of!(GpuObject, vertex_address), 16);
    assert_eq!(std::mem::offset_of!(GpuObject, index_address), 24);
    assert_eq!(std::mem::offset_of!(GpuObject, object_to_world), 32);
    let transform = Mat4::from_translation(Vec3::new(2.0, 3.0, 4.0));
    for index_address in [None, Some(0x2000)] {
        let object = GpuObject::new(7, 0x1000, index_address, transform);
        assert_eq!(object.material_index, 7);
        assert_eq!(object.vertex_address, 0x1000);
        assert_eq!(object.object_to_world, transform.to_cols_array_2d());
        assert_eq!(object._pad, [0; 2]);
        assert_eq!(
            (object.indexed, object.index_address),
            if index_address.is_some() {
                (1, 0x2000)
            } else {
                (0, 0)
            }
        );
    }
}

#[test]
fn precomputes_shadow_dirs_for_exterior_corner() {
    let edge = test_edge();

    let gpu_edge = GpuDiffractionEdge::from(&edge);
    let diagonal = std::f32::consts::FRAC_1_SQRT_2;
    let positive_plane_normal = Vec3::from_array(gpu_edge.positive_plane_normal);
    let bisector_dir = Vec3::from_array(gpu_edge.bisector_dir);
    let positive_shadow_dir = Vec3::from_array(gpu_edge.positive_shadow_dir);
    let negative_shadow_dir = Vec3::from_array(gpu_edge.negative_shadow_dir);

    assert_eq!(positive_plane_normal, Vec3::Z);
    assert_eq!(bisector_dir, Vec3::X);
    assert!((positive_shadow_dir - Vec3::new(-diagonal, 0.0, diagonal)).length() <= 1.0e-6);
    assert!((negative_shadow_dir - Vec3::new(-diagonal, 0.0, -diagonal)).length() <= 1.0e-6);
    assert!((gpu_edge.solid_wedge_cos - diagonal).abs() <= 1.0e-6);
}

#[test]
fn diffraction_bounds_include_the_rotated_strip_and_plane_thickness() {
    let edge = DiffractionEdge {
        start: Vec3::new(1.0, 2.0, 3.0),
        end: Vec3::new(3.0, 4.0, 3.0),
        bisector_dir: Vec3::Z,
        ..test_edge()
    };
    let bounds = GpuDiffractionEdge::from(&edge).bounds();
    let thickness = 0.001 * std::f32::consts::FRAC_1_SQRT_2;
    assert!((bounds.min - Vec3::new(1.0 - thickness, 2.0 - thickness, 2.5)).length() < 1e-6);
    assert!((bounds.max - Vec3::new(3.0 + thickness, 4.0 + thickness, 3.5)).length() < 1e-6);
    assert!(bounds.opaque);
}

fn test_edge() -> DiffractionEdge {
    DiffractionEdge {
        id: 1,
        start: Vec3::ZERO,
        end: Vec3::Y,
        bisector_dir: Vec3::X,
        edge_angle_radians: 1.5 * std::f32::consts::PI,
        diffraction_radius: 0.5,
    }
}
