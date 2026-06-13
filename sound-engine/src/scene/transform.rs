/// Returns a row-major 3x4 identity transform for TLAS instances.
pub fn identity_transform() -> [f32; 12] {
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
}

/// Returns a row-major 3x4 translation transform for TLAS instances.
pub fn translation_transform(x: f32, y: f32, z: f32) -> [f32; 12] {
    [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, y, 0.0, 0.0, 1.0, z]
}
