use crate::gpu::GpuError;
use crate::gpu::backend::VkDeviceContext;
use ash::vk;
use std::collections::HashMap;
use std::ffi::CString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

pub(super) const SHADER_ENTRY_POINT: &str = "main";

pub(crate) struct ShaderLibrary {
    device_context: Arc<VkDeviceContext>,
    shaders: Mutex<HashMap<ShaderId, ShaderRecord>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ShaderId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ShaderStage {
    Compute,
    RayGeneration,
    RayMiss,
    RayClosestHit,
    RayAnyHit,
    RayIntersection,
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

    pub fn load_glsl_file<P: AsRef<Path>>(&self, stage: ShaderStage, path: P) -> Result<ShaderId, GpuError> {
        let source_path = path.as_ref().to_path_buf();
        if !source_path.exists() {
            return Err(
                std::io::Error::other(format!("Shader source file does not exist: {}", source_path.display())).into(),
            );
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
            .map_err(|_| std::io::Error::other("Shader library lock is poisoned"))?;

        let id = ShaderId(shaders.len() as u32);
        shaders.insert(id, record);
        Ok(id)
    }

    // pub fn create_compute_pipeline(&self, shader: ShaderId) -> Result<PipelineId, GpuError> {
    //     let mut state = self
    //         .state
    //         .lock()
    //         .map_err(|_| std::io::Error::other("Shader library lock is poisoned"))?;

    //     let shader_record = state
    //         .shaders
    //         .get(shader.0 as usize)
    //         .ok_or_else(|| std::io::Error::other(format!("Invalid shader id {}", shader.0)))?;
    //     if shader_record.stage != ShaderStage::Compute {
    //         return Err(std::io::Error::other(format!(
    //             "create_compute_pipeline requires compute shader, got {:?}",
    //             shader_record.stage
    //         ))
    //         .into());
    //     }

    //     let layout_info = vk::PipelineLayoutCreateInfo::default();
    //     let layout = unsafe { self.device_context.device.create_pipeline_layout(&layout_info, None)? };

    //     let stage_info = vk::PipelineShaderStageCreateInfo::default()
    //         .stage(shader_record.stage.to_vk())
    //         .module(shader_record.module)
    //         .name(shader_record.entry.as_c_str());

    //     let compute_info = vk::ComputePipelineCreateInfo::default()
    //         .stage(stage_info)
    //         .layout(layout);

    //     let pipeline = unsafe {
    //         self.device_context
    //             .device
    //             .create_compute_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&compute_info), None)
    //             .map_err(|(_, err)| err)?[0]
    //     };

    //     let id = PipelineId(state.pipelines.len() as u32);
    //     state.pipelines.push(PipelineRecord { pipeline, layout });
    //     Ok(id)
    // }

    pub fn shader_module(&self, shader_id: ShaderId) -> Result<vk::ShaderModule, GpuError> {
        let shaders = self
            .shaders
            .lock()
            .map_err(|_| std::io::Error::other("Shader library lock is poisoned"))?;

        let shader = shaders
            .get(&shader_id)
            .ok_or_else(|| std::io::Error::other(format!("Invalid shader id {}", shader_id.0)))?;

        Ok(shader.module)
    }

    // pub fn pipeline(&self, pipeline: PipelineId) -> Result<vk::Pipeline, GpuError> {
    //     let state = self
    //         .state
    //         .lock()
    //         .map_err(|_| std::io::Error::other("Shader library lock is poisoned"))?;
    //     let pipeline_record = state
    //         .pipelines
    //         .get(pipeline.0 as usize)
    //         .ok_or_else(|| std::io::Error::other(format!("Invalid pipeline id {}", pipeline.0)))?;
    //     Ok(pipeline_record.pipeline)
    // }

    // pub fn pipeline_layout(&self, pipeline: PipelineId) -> Result<vk::PipelineLayout, GpuError> {
    //     let state = self
    //         .state
    //         .lock()
    //         .map_err(|_| std::io::Error::other("Shader library lock is poisoned"))?;
    //     let pipeline_record = state
    //         .pipelines
    //         .get(pipeline.0 as usize)
    //         .ok_or_else(|| std::io::Error::other(format!("Invalid pipeline id {}", pipeline.0)))?;
    //     Ok(pipeline_record.layout)
    // }

    // pub fn shader_source_path(&self, shader: ShaderId) -> Result<PathBuf, GpuError> {
    //     let state = self
    //         .state
    //         .lock()
    //         .map_err(|_| std::io::Error::other("Shader library lock is poisoned"))?;
    //     let shader_record = state
    //         .shaders
    //         .get(shader.0 as usize)
    //         .ok_or_else(|| std::io::Error::other(format!("Invalid shader id {}", shader.0)))?;
    //     Ok(shader_record.source_path.clone())
    // }
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

fn create_shader_module(device: &ash::Device, spirv_words: &[u32]) -> Result<vk::ShaderModule, GpuError> {
    if spirv_words.is_empty() {
        return Err(std::io::Error::other("SPIR-V module is empty").into());
    }

    let create_info = vk::ShaderModuleCreateInfo::default().code(spirv_words);
    let module = unsafe { device.create_shader_module(&create_info, None)? };
    Ok(module)
}

fn compile_glsl_file_to_spirv_words(source_path: &Path, stage: ShaderStage, entry: &str) -> Result<Vec<u32>, GpuError> {
    let temp_dir = std::env::temp_dir();
    let unique_tag = format!(
        "{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    );
    let spv_path = temp_dir.join(format!("sound_engine_shader_{unique_tag}.spv"));

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
            std::io::Error::other(format!(
                "Failed to run glslangValidator for {}: {e}. Ensure Vulkan SDK tools are installed and glslangValidator is in PATH",
                source_path.display()
            ))
        })?;

    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let _ = fs::remove_file(&spv_path);
        return Err(std::io::Error::other(format!(
            "GLSL compilation failed for {} ({stage:?}):\n{stdout}\n{stderr}",
            source_path.display()
        ))
        .into());
    }

    let result = (|| -> Result<Vec<u32>, GpuError> {
        let spirv_bytes = fs::read(&spv_path)?;
        if spirv_bytes.len() % size_of::<u32>() != 0 {
            return Err(std::io::Error::other(format!("Invalid SPIR-V size for {}", source_path.display())).into());
        }

        Ok(spirv_bytes
            .chunks_exact(size_of::<u32>())
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    })();

    let _ = fs::remove_file(&spv_path);
    result
}
