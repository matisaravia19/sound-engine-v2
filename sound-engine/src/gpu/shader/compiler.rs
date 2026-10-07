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
#[path = "../../../tests/unit/gpu/shader/compiler.rs"]
mod tests;
