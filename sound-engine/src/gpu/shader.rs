use crate::core::error::{ErrorCode, SoundError, SoundResult};
use crate::gpu::backend::VkDeviceContext;
use ash::vk;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

/// Default shader entry point used when compiling GLSL modules.
pub const SHADER_ENTRY_POINT: &str = "main";

/// Owns compiled Vulkan shader modules and identifies them by `ShaderId`.
///
/// Shaders are compiled from GLSL files through `glslangValidator` and kept
/// alive until the library is dropped.
pub struct ShaderLibrary {
    device_context: Arc<VkDeviceContext>,
    shaders: Mutex<HashMap<ShaderId, ShaderRecord>>,
}

/// Stable handle to a shader module stored in `ShaderLibrary`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShaderId(pub u32);

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
    /// Ray any-hit shader stage.
    RayAnyHit,
    /// Ray intersection shader stage.
    RayIntersection,
    /// Ray callable shader stage.
    RayCallable,
}

struct ShaderRecord {
    module: vk::ShaderModule,
    stage: ShaderStage,
    source_path: PathBuf,
}

impl ShaderStage {
    fn to_vk(self) -> vk::ShaderStageFlags {
        match self {
            ShaderStage::Compute => vk::ShaderStageFlags::COMPUTE,
            ShaderStage::RayGeneration => vk::ShaderStageFlags::RAYGEN_KHR,
            ShaderStage::RayMiss => vk::ShaderStageFlags::MISS_KHR,
            ShaderStage::RayClosestHit => vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            ShaderStage::RayAnyHit => vk::ShaderStageFlags::ANY_HIT_KHR,
            ShaderStage::RayIntersection => vk::ShaderStageFlags::INTERSECTION_KHR,
            ShaderStage::RayCallable => vk::ShaderStageFlags::CALLABLE_KHR,
        }
    }

    fn glslang_suffix(self) -> &'static str {
        match self {
            ShaderStage::Compute => "comp",
            ShaderStage::RayGeneration => "rgen",
            ShaderStage::RayMiss => "rmiss",
            ShaderStage::RayClosestHit => "rchit",
            ShaderStage::RayAnyHit => "rahit",
            ShaderStage::RayIntersection => "rint",
            ShaderStage::RayCallable => "rcall",
        }
    }
}

impl ShaderLibrary {
    pub(super) fn new(device_context: Arc<VkDeviceContext>) -> Self {
        Self {
            device_context,
            shaders: Mutex::new(HashMap::new()),
        }
    }

    /// Compiles a GLSL file into SPIR-V and creates a Vulkan shader module.
    ///
    /// The returned ID remains valid until the shader library is dropped.
    pub fn load_glsl_file<P: AsRef<Path>>(&self, stage: ShaderStage, path: P) -> SoundResult<ShaderId> {
        let source_path = path.as_ref().to_path_buf();
        if !source_path.exists() {
            return Err(SoundError::io(format!(
                "Shader source file does not exist: {}",
                source_path.display()
            )));
        }

        let spirv_words = compile_glsl_file_to_spirv_words(&source_path, stage, SHADER_ENTRY_POINT)?;
        let module = create_shader_module(&self.device_context.device, &spirv_words)?;
        let record = ShaderRecord {
            module,
            stage,
            source_path,
        };

        let mut shaders = self
            .shaders
            .lock()
            .map_err(|_| SoundError::poisoned_lock("Shader library lock is poisoned"))?;

        let id = ShaderId(shaders.len() as u32);
        shaders.insert(id, record);
        Ok(id)
    }

    /// Returns the raw Vulkan shader module for pipeline creation.
    pub fn shader_module(&self, shader_id: ShaderId) -> SoundResult<vk::ShaderModule> {
        let shaders = self
            .shaders
            .lock()
            .map_err(|_| SoundError::poisoned_lock("Shader library lock is poisoned"))?;

        let shader = shaders
            .get(&shader_id)
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
            .get(&shader_id)
            .ok_or_else(|| SoundError::not_found(format!("Invalid shader id {}", shader_id.0)))?;

        Ok(shader.stage)
    }
}

impl Drop for ShaderLibrary {
    fn drop(&mut self) {
        let mut shaders = self
            .shaders
            .lock()
            .expect("Shader library lock is poisoned during drop");

        unsafe {
            for (_, shader) in shaders.drain() {
                self.device_context.device.destroy_shader_module(shader.module, None);
            }
        }
    }
}

fn create_shader_module(device: &ash::Device, spirv_words: &[u32]) -> SoundResult<vk::ShaderModule> {
    if spirv_words.is_empty() {
        return Err(SoundError::shader_compile_failed("SPIR-V module is empty"));
    }

    let create_info = vk::ShaderModuleCreateInfo::default().code(spirv_words);
    let module = unsafe { device.create_shader_module(&create_info, None)? };
    Ok(module)
}

fn compile_glsl_file_to_spirv_words(source_path: &Path, stage: ShaderStage, entry: &str) -> SoundResult<Vec<u32>> {
    let temp_dir = std::env::temp_dir();
    let unique_tag = format!(
        "{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    );
    let spv_path = temp_dir.join(format!("sound_engine_shader_{unique_tag}.spv"));

    // Compile to a unique temp file because glslangValidator writes SPIR-V to disk.
    let output = Command::new("glslangValidator")
        .arg("-V")
        .arg("--target-env")
        .arg("vulkan1.3")
        .arg("-S")
        .arg(stage.glslang_suffix())
        .arg("-e")
        .arg(entry)
        .arg("-o")
        .arg(&spv_path)
        .arg(source_path)
        .output()
        .map_err(|e| {
            SoundError::with_source(
                ErrorCode::ShaderCompileFailed,
                format!(
                "Failed to run glslangValidator for {}: {e}. Ensure Vulkan SDK tools are installed and glslangValidator is in PATH",
                source_path.display()
                ),
                e,
            )
        })?;

    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let _ = fs::remove_file(&spv_path);
        return Err(SoundError::shader_compile_failed(format!(
            "GLSL compilation failed for {} ({stage:?}):\n{stdout}\n{stderr}",
            source_path.display()
        )));
    }

    let result = (|| -> SoundResult<Vec<u32>> {
        let spirv_bytes = fs::read(&spv_path)?;
        if spirv_bytes.len() % size_of::<u32>() != 0 {
            return Err(SoundError::shader_compile_failed(format!(
                "Invalid SPIR-V size for {}",
                source_path.display()
            )));
        }

        Ok(spirv_bytes
            .chunks_exact(size_of::<u32>())
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    })();

    let _ = fs::remove_file(&spv_path);
    result
}
