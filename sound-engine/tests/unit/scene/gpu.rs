use super::*;
use std::mem::{align_of, size_of};

#[test]
fn uploaded_reflection_mixture_has_no_scattering_gain() {
    for absorption in [0.0_f32, 0.2, 0.75, 1.0] {
        for scattering in [0.0_f32, 0.4, 1.0] {
            let material = super::super::types::Material {
                id: 1,
                absorption,
                scattering,
                roughness: 0.2,
            };
            let gpu = gpu_material(&material);
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
}

#[test]
fn precomputes_shadow_dirs_for_exterior_corner() {
    let edge = test_edge();

    let gpu_edge = gpu_diffraction_edge(&edge);
    let delta = std::f32::consts::FRAC_PI_4;
    let positive_plane_normal = Vec3::from_array(gpu_edge.positive_plane_normal);
    let bisector_dir = Vec3::from_array(gpu_edge.bisector_dir);
    let positive_shadow_dir = Vec3::from_array(gpu_edge.positive_shadow_dir);
    let negative_shadow_dir = Vec3::from_array(gpu_edge.negative_shadow_dir);
    let expected_positive =
        (delta.cos() * positive_plane_normal - delta.sin() * bisector_dir).normalize();
    let expected_negative =
        (-delta.cos() * positive_plane_normal - delta.sin() * bisector_dir).normalize();
    let expected_solid_wedge_cos = std::f32::consts::FRAC_PI_4.cos();

    assert!((positive_shadow_dir - expected_positive).length() <= 1.0e-6);
    assert!((negative_shadow_dir - expected_negative).length() <= 1.0e-6);
    assert!((gpu_edge.solid_wedge_cos - expected_solid_wedge_cos).abs() <= 1.0e-6);
}

#[test]
fn chooses_shadow_dir_from_projected_incoming_side() {
    let gpu_edge = gpu_diffraction_edge(&test_edge());
    let positive_plane_normal = Vec3::from_array(gpu_edge.positive_plane_normal);
    let edge_dir = Vec3::from_array(gpu_edge.edge_dir);
    let bisector_dir = Vec3::from_array(gpu_edge.bisector_dir);
    let positive_shadow_dir = Vec3::from_array(gpu_edge.positive_shadow_dir);
    let negative_shadow_dir = Vec3::from_array(gpu_edge.negative_shadow_dir);

    let positive = choose_shadow_dir(positive_plane_normal, &gpu_edge).unwrap();
    let negative = choose_shadow_dir(-positive_plane_normal, &gpu_edge).unwrap();
    let parallel_to_edge = choose_shadow_dir(edge_dir, &gpu_edge);
    let parallel_to_plane = choose_shadow_dir(bisector_dir, &gpu_edge);
    let inside_solid_wedge = choose_shadow_dir(-bisector_dir, &gpu_edge);

    assert!((positive - positive_shadow_dir).length() <= 1.0e-6);
    assert!((negative - negative_shadow_dir).length() <= 1.0e-6);
    assert!(parallel_to_edge.is_none());
    assert!(parallel_to_plane.is_none());
    assert!(inside_solid_wedge.is_none());
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

fn choose_shadow_dir(ray_direction: Vec3, edge: &GpuDiffractionEdge) -> Option<Vec3> {
    let edge_dir = Vec3::from_array(edge.edge_dir);
    let positive_plane_normal = Vec3::from_array(edge.positive_plane_normal);
    let mut d_perp = ray_direction - ray_direction.dot(edge_dir) * edge_dir;
    let d_perp_len = d_perp.length();
    if d_perp_len <= 1.0e-6 {
        return None;
    }
    d_perp /= d_perp_len;
    if d_perp.dot(-Vec3::from_array(edge.bisector_dir)) >= edge.solid_wedge_cos - 1.0e-6 {
        return None;
    }

    let side = d_perp.dot(positive_plane_normal);
    if side.abs() <= 1.0e-6 {
        return None;
    }

    Some(if side > 0.0 {
        Vec3::from_array(edge.positive_shadow_dir)
    } else {
        Vec3::from_array(edge.negative_shadow_dir)
    })
}
