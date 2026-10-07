use super::*;

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn audit_verified_bundled_shader_skips_recompilation() {
    let gpu = crate::gpu::backend::VkBackend::new().unwrap();
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/auralization/shaders/multiply.comp.glsl");
    let first = gpu
        .shaders()
        .load_glsl_file(ShaderStage::Compute, &path)
        .unwrap();
    for _ in 0..4 {
        assert_eq!(
            gpu.shaders()
                .load_glsl_file(ShaderStage::Compute, &path)
                .unwrap(),
            first
        );
    }
    assert_eq!(
        gpu.shaders()
            .compile_count
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
}

#[test]
fn bundled_manifest_covers_shared_transport_and_current_sources() {
    assert!(
        BUNDLED_SHADER_SOURCES
            .iter()
            .any(|(path, _)| path.ends_with("/listener_transport.glsl"))
    );
    assert!(
        BUNDLED_SHADER_SOURCES
            .iter()
            .any(|(path, _)| path.ends_with("/diffraction.rint.glsl"))
    );
    validate_bundled_shader_sources(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/acoustics/shaders/listener_backward.rgen.glsl"),
    )
    .unwrap();
}
