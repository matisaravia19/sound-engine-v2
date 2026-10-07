use crate::core::error::{SoundError, SoundResult};
use crate::gpu::backend::VkDeviceContext;
use ash::vk;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};

mod compiler;

// Cargo tracks every included GLSL file, including shared transport includes.
// The generated manifest is also used to reject stale host/source combinations.
include!(concat!(env!("OUT_DIR"), "/shader_sources.rs"));

/// Default shader entry point used when compiling GLSL modules.
pub const SHADER_ENTRY_POINT: &str = "main";

/// Owns compiled Vulkan shader modules and identifies them by `ShaderId`.
///
/// Shaders are compiled from GLSL files through `glslangValidator` and kept
/// alive until the library is dropped.
pub struct ShaderLibrary {
    device_context: Arc<VkDeviceContext>,
    // Records are append-only, so handles remain stable without a keyed registry.
    shaders: Mutex<Vec<ShaderRecord>>,
    #[cfg(test)]
    compile_count: std::sync::atomic::AtomicUsize,
}

/// Library-local handle to a shader module retained by its `ShaderLibrary`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShaderId(usize);

/// Shader stages supported by the engine shader compiler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderStage {
    /// Compute shader stage.
    Compute,
    /// Ray generation shader stage.
    RayGeneration,
    /// Ray miss shader stage.
    RayMiss,
    /// Ray closest-hit shader stage.
    RayClosestHit,
    /// Ray intersection shader stage.
    RayIntersection,
}

struct ShaderRecord {
    module: vk::ShaderModule,
    stage: ShaderStage,
    spirv: Vec<u32>,
    // Manifest entries are immutable after verification; custom files and their
    // includes still need compilation to determine whether their bytes changed.
    bundled_sources: Vec<usize>,
}

impl ShaderStage {
    fn glslang_suffix(self) -> &'static str {
        match self {
            ShaderStage::Compute => "comp",
            ShaderStage::RayGeneration => "rgen",
            ShaderStage::RayMiss => "rmiss",
            ShaderStage::RayClosestHit => "rchit",
            ShaderStage::RayIntersection => "rint",
        }
    }
}

impl ShaderLibrary {
    /// Shares device ownership; compiled modules live until this library is dropped.
    pub(super) fn new(device_context: Arc<VkDeviceContext>) -> Self {
        Self {
            device_context,
            shaders: Mutex::new(Vec::new()),
            #[cfg(test)]
            compile_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Compiles a GLSL file into SPIR-V and creates a Vulkan shader module.
    ///
    /// The returned ID remains valid until the shader library is dropped.
    /// Verified bundled sources reuse an existing module without recompilation.
    /// Bundled engine shaders must match the sources used to build this host
    /// executable; rebuild after editing GLSL, including shared include files.
    /// Compilation uses embedded bundled bytes so subsequent file edits cannot
    /// change the verified program. Custom shader files remain caller-controlled.
    pub fn load_glsl_file<P: AsRef<Path>>(
        &self,
        stage: ShaderStage,
        path: P,
    ) -> SoundResult<ShaderId> {
        let source_path = path.as_ref().to_path_buf();
        let bundled_source = validate_bundled_shader_sources(&source_path)?;
        if !source_path.exists() {
            return Err(SoundError::io(format!(
                "Shader source file does not exist: {}",
                source_path.display()
            )));
        }

        let mut shaders = self
            .shaders
            .lock()
            .map_err(|_| SoundError::poisoned_lock("Shader library lock is poisoned"))?;
        if let Some(source) = bundled_source
            && let Some(index) = shaders.iter().position(|shader| {
                shader.stage == stage && shader.bundled_sources.contains(&source)
            })
        {
            return Ok(ShaderId(index));
        }
        #[cfg(test)]
        self.compile_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let spirv_words = compiler::compile(&source_path, stage, bundled_source)?;
        // Recreating an engine on a shared backend must not accumulate identical
        // modules. Compiled bytes also reflect changes in included GLSL files.
        if let Some((index, shader)) = shaders
            .iter_mut()
            .enumerate()
            .find(|(_, shader)| shader.stage == stage && shader.spirv == spirv_words)
        {
            if let Some(source) = bundled_source {
                shader.bundled_sources.push(source);
            }
            return Ok(ShaderId(index));
        }
        let id = ShaderId(shaders.len());
        let module = create_shader_module(&self.device_context.device, &spirv_words)?;
        let record = ShaderRecord {
            module,
            stage,
            spirv: spirv_words,
            bundled_sources: bundled_source.into_iter().collect(),
        };

        shaders.push(record);
        Ok(id)
    }

    /// Returns the raw Vulkan shader module for pipeline creation.
    pub fn shader_module(&self, shader_id: ShaderId) -> SoundResult<vk::ShaderModule> {
        let shaders = self
            .shaders
            .lock()
            .map_err(|_| SoundError::poisoned_lock("Shader library lock is poisoned"))?;

        let shader = shaders
            .get(shader_id.0)
            .ok_or_else(|| SoundError::not_found(format!("Invalid shader id {}", shader_id.0)))?;

        Ok(shader.module)
    }

    /// Returns the stage recorded when the shader was loaded.
    pub fn shader_stage(&self, shader_id: ShaderId) -> SoundResult<ShaderStage> {
        let shaders = self
            .shaders
            .lock()
            .map_err(|_| SoundError::poisoned_lock("Shader library lock is poisoned"))?;

        let shader = shaders
            .get(shader_id.0)
            .ok_or_else(|| SoundError::not_found(format!("Invalid shader id {}", shader_id.0)))?;

        Ok(shader.stage)
    }
}

/// Rejects source edits before compiling any bundled shader against a stale
/// host ABI. Custom caller-owned shader files retain the ordinary loading path.
fn validate_bundled_shader_sources(source_path: &Path) -> SoundResult<Option<usize>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let requested = source_path.canonicalize().ok();
    let bundled = BUNDLED_SHADER_SOURCES.iter().position(|(relative, _)| {
        let expected = root.join(relative);
        source_path == expected
            || requested.as_ref().is_some_and(|path| {
                expected
                    .canonicalize()
                    .is_ok_and(|canonical| path == &canonical)
            })
    });
    if bundled.is_some() {
        // Validate includes as well as entry points. This is setup I/O, outside
        // the audio render loop; no filesystem reads are added to ray tracing.
        for (relative, expected) in BUNDLED_SHADER_SOURCES {
            verify_shader_source(&root.join(relative), expected)?;
        }
    }
    Ok(bundled)
}

/// Checks exact bytes so a host executable cannot silently adopt changed GLSL.
fn verify_shader_source(path: &Path, expected: &str) -> SoundResult<()> {
    let actual = fs::read(path).map_err(|error| {
        SoundError::shader_compile_failed(format!(
            "Cannot verify bundled shader {}: {error}. Restore the source files and rebuild the executable.",
            path.display()
        ))
    })?;
    if actual != expected.as_bytes() {
        return Err(SoundError::shader_compile_failed(format!(
            "Bundled shader {} differs from the source used to build this executable. Rebuild with cargo before running; mixing an old host with edited GLSL can invalidate acoustic results.",
            path.display()
        )));
    }
    Ok(())
}

impl Drop for ShaderLibrary {
    fn drop(&mut self) {
        let shaders = self
            .shaders
            .get_mut()
            .unwrap_or_else(|error| error.into_inner());

        unsafe {
            for shader in shaders.drain(..) {
                self.device_context
                    .device
                    .destroy_shader_module(shader.module, None);
            }
        }
    }
}

fn create_shader_module(
    device: &ash::Device,
    spirv_words: &[u32],
) -> SoundResult<vk::ShaderModule> {
    if spirv_words.is_empty() {
        return Err(SoundError::shader_compile_failed("SPIR-V module is empty"));
    }

    let create_info = vk::ShaderModuleCreateInfo::default().code(spirv_words);
    let module = unsafe { device.create_shader_module(&create_info, None)? };
    Ok(module)
}

#[cfg(test)]
#[path = "../../tests/unit/gpu/shader.rs"]
mod source_tests;
