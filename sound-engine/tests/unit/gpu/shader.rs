use super::*;

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn bundled_shaders_reuse_modules_and_preserve_handles_after_appends() {
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

    let first_module = gpu.shaders().shader_module(first).unwrap();
    let miss = gpu
        .shaders()
        .load_glsl_file(
            ShaderStage::RayMiss,
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("src/acoustics/shaders/visibility.rmiss.glsl"),
        )
        .unwrap();
    assert_ne!(first, miss);
    assert_ne!(first_module, gpu.shaders().shader_module(miss).unwrap());
    assert_eq!(
        gpu.shaders().shader_stage(miss).unwrap(),
        ShaderStage::RayMiss
    );
    assert_eq!(
        gpu.shaders().shader_stage(first).unwrap(),
        ShaderStage::Compute
    );
    assert_eq!(gpu.shaders().shader_module(first).unwrap(), first_module);
    assert_eq!(
        gpu.shaders()
            .load_glsl_file(ShaderStage::Compute, &path)
            .unwrap(),
        first
    );
    assert_eq!(
        gpu.shaders()
            .compile_count
            .load(std::sync::atomic::Ordering::Relaxed),
        2
    );

    let missing = ShaderId(usize::MAX);
    for error in [
        gpu.shaders().shader_module(missing).unwrap_err(),
        gpu.shaders().shader_stage(missing).unwrap_err(),
    ] {
        assert_eq!(error.code(), crate::core::error::ErrorCode::NotFound);
    }
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
