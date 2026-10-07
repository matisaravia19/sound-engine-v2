use super::*;

#[test]
fn triangle_validation_rejects_out_of_bounds_indices_and_partial_triangles() {
    let vertices = [Vec3::ZERO, Vec3::X, Vec3::Y];
    assert!(validate_triangle_mesh(&vertices, None).is_ok());
    assert!(validate_triangle_mesh(&[], None).is_err());
    assert!(validate_triangle_mesh(&vertices, Some(&[])).is_err());
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
