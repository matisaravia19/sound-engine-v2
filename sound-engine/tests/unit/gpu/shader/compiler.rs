use super::*;

#[test]
fn workspaces_are_distinct_and_cleanup_on_error() {
    let first = CompilerWorkspace::new().unwrap();
    let mut second_path = PathBuf::new();
    let result = (|| -> SoundResult<()> {
        let second = CompilerWorkspace::new()?;
        second_path = second.path.clone();
        assert_ne!(first.path, second.path);
        fs::create_dir(second.path.join("nested"))?;
        fs::write(second.path.join("nested/partial.spv"), b"partial")?;
        Err(SoundError::shader_compile_failed("test error"))
    })();
    assert!(result.is_err());
    assert!(!second_path.exists());
    assert!(first.path.is_dir());
    let first_path = first.path.clone();
    drop(first);
    assert!(!first_path.exists());
}

#[test]
fn bundled_snapshot_preserves_all_sources_and_include_paths() {
    let workspace = CompilerWorkspace::new().unwrap();
    let entry = workspace.write_bundled_sources(0).unwrap();
    assert_eq!(
        fs::read(entry).unwrap(),
        BUNDLED_SHADER_SOURCES[0].1.as_bytes()
    );
    for (relative, expected) in BUNDLED_SHADER_SOURCES {
        assert_eq!(
            fs::read(workspace.path.join(relative)).unwrap(),
            expected.as_bytes()
        );
    }
}

#[test]
fn edited_or_missing_shader_is_rejected_before_compilation() {
    let workspace = CompilerWorkspace::new().unwrap();
    let path = workspace.path.join("source.glsl");
    let expected = "layout(offset = 100) uint changed_layout;\n";
    fs::write(&path, expected).unwrap();
    assert!(super::super::verify_shader_source(&path, expected).is_ok());
    fs::write(&path, "layout(offset = 104) uint changed_layout;\n").unwrap();
    let error = super::super::verify_shader_source(&path, expected).unwrap_err();
    assert!(error.to_string().contains("Rebuild with cargo"));
    fs::remove_file(&path).unwrap();
    assert!(super::super::verify_shader_source(&path, expected).is_err());
}

#[test]
fn malformed_spirv_is_rejected_before_vulkan() {
    let valid = [0x0723_0203u32, 0x0001_0600, 0, 1, 0];
    let encode = |words: &[u32]| {
        words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>()
    };
    assert_eq!(decode_spirv(&encode(&valid)).unwrap(), valid);
    assert!(decode_spirv(&[]).is_err());
    assert!(decode_spirv(&encode(&valid)[..19]).is_err());
    assert!(decode_spirv(&[0; 20]).is_err());
    for (index, value) in [(0, 0), (3, 0), (4, 1)] {
        let mut invalid = valid;
        invalid[index] = value;
        assert!(decode_spirv(&encode(&invalid)).is_err());
    }
}

#[test]
#[ignore = "requires glslangValidator in PATH"]
fn compiler_handles_bundled_includes_custom_paths_and_failure() {
    let bundled = BUNDLED_SHADER_SOURCES
        .iter()
        .position(|(path, _)| path.ends_with("/listener_backward.rgen.glsl"))
        .unwrap();
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join(BUNDLED_SHADER_SOURCES[bundled].0);
    assert!(compile(&source, ShaderStage::RayGeneration, Some(bundled)).is_ok());
    let workspace = CompilerWorkspace::new().unwrap();
    let custom = workspace.path.join("-custom shader.comp.glsl");
    fs::write(
        &custom,
        "#version 450\nlayout(local_size_x=1) in; void main() {}\n",
    )
    .unwrap();
    assert!(compile(&custom, ShaderStage::Compute, None).is_ok());
    fs::write(&custom, "#version 450\ninvalid source").unwrap();
    let error = compile(&custom, ShaderStage::Compute, None).unwrap_err();
    assert!(error.to_string().contains("GLSL compilation failed"));
}
