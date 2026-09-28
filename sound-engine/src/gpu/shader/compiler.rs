use super::{BUNDLED_SHADER_SOURCES, SHADER_ENTRY_POINT, ShaderStage};
use crate::core::error::{ErrorCode, SoundError, SoundResult};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

/// Compiles verified bundled bytes or a caller-owned file in an owned workspace.
/// The workspace attempts cleanup on success and every error path.
pub(super) fn compile(
    source_path: &Path,
    stage: ShaderStage,
    bundled_source: Option<usize>,
) -> SoundResult<Vec<u32>> {
    let workspace = CompilerWorkspace::new()?;
    let compiler_source = match bundled_source {
        Some(index) => workspace.write_bundled_sources(index)?,
        // An absolute path cannot be interpreted as a compiler option. Includes
        // in custom shaders continue to resolve relative to their source file.
        None => source_path.canonicalize()?,
    };
    let output_path = workspace.path.join("module.spv");
    let mut command = Command::new("glslangValidator");
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let output = command
        .args(["-V", "--target-env", "vulkan1.3", "-S"])
        .arg(stage.glslang_suffix())
        .args(["-e", SHADER_ENTRY_POINT, "-o"])
        .arg(&output_path)
        .arg(&compiler_source)
        .output()
        .map_err(|error| {
            SoundError::with_source(
                ErrorCode::ShaderCompileFailed,
                format!(
                    "Failed to run glslangValidator for {}: {error}. Ensure Vulkan SDK tools are installed and glslangValidator is in PATH",
                    source_path.display()
                ),
                error,
            )
        })?;
    if !output.status.success() {
        return Err(SoundError::shader_compile_failed(format!(
            "GLSL compilation failed for {} ({stage:?}):\n{}\n{}",
            source_path.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    decode_spirv(&fs::read(output_path)?)
}

struct CompilerWorkspace {
    path: PathBuf,
    parent: PathBuf,
}

impl CompilerWorkspace {
    fn new() -> SoundResult<Self> {
        static NEXT_WORKSPACE: AtomicU64 = AtomicU64::new(0);
        let parent = std::env::temp_dir().canonicalize()?;
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        for _ in 0..64 {
            let serial = NEXT_WORKSPACE.fetch_add(1, Ordering::Relaxed);
            let path = parent.join(format!(
                "sound-engine-shader-{}-{timestamp}-{serial}",
                std::process::id()
            ));
            let builder = fs::DirBuilder::new();
            #[cfg(unix)]
            let mut builder = builder;
            #[cfg(unix)]
            {
                use std::os::unix::fs::DirBuilderExt;
                builder.mode(0o700);
            }
            // Exclusive creation refuses pre-existing files, directories and
            // links. The compiler never opens a predictable shared output file.
            match builder.create(&path) {
                Ok(()) => {
                    return Ok(Self {
                        path,
                        parent,
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Err(SoundError::io("Cannot reserve a shader compiler workspace"))
    }

    fn write_bundled_sources(&self, index: usize) -> SoundResult<PathBuf> {
        // Compile the same immutable source bytes that were checked against the
        // host manifest, even if a working-tree file changes after verification.
        for (relative, source) in BUNDLED_SHADER_SOURCES {
            let destination = self.path.join(relative);
            fs::create_dir_all(destination.parent().unwrap())?;
            fs::write(destination, source)?;
        }
        Ok(self.path.join(BUNDLED_SHADER_SOURCES[index].0))
    }
}

impl Drop for CompilerWorkspace {
    fn drop(&mut self) {
        // Delete only the exclusively created direct child of our canonical
        // temporary root; refuse a replaced path that resolves elsewhere.
        if let Ok(resolved) = self.path.canonicalize()
            && resolved == self.path
            && resolved.parent() == Some(self.parent.as_path())
        {
            let _ = fs::remove_dir_all(&resolved);
        }
    }
}

fn decode_spirv(bytes: &[u8]) -> SoundResult<Vec<u32>> {
    if bytes.len() < 5 * size_of::<u32>() || !bytes.len().is_multiple_of(size_of::<u32>()) {
        return Err(SoundError::shader_compile_failed("Invalid SPIR-V size"));
    }
    let words: Vec<_> = bytes
        .chunks_exact(size_of::<u32>())
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    if words[0] != 0x0723_0203 || words[3] == 0 || words[4] != 0 {
        return Err(SoundError::shader_compile_failed("Invalid SPIR-V header"));
    }
    Ok(words)
}

#[cfg(test)]
mod tests {
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
}
