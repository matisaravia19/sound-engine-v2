use crate::gpu::context::GpuContext;
use crate::gpu::errors::GpuError;
use ash::vk;
use std::ffi::CString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderDomain {
    Compute,
    RayTracing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderStage {
    Compute,
    RayGeneration,
    RayMiss,
    RayClosestHit,
    RayAnyHit,
    RayIntersection,
    RayCallable,
}

impl ShaderStage {
    pub fn domain(self) -> ShaderDomain {
        match self {
            ShaderStage::Compute => ShaderDomain::Compute,
            ShaderStage::RayGeneration
            | ShaderStage::RayMiss
            | ShaderStage::RayClosestHit
            | ShaderStage::RayAnyHit
            | ShaderStage::RayIntersection
            | ShaderStage::RayCallable => ShaderDomain::RayTracing,
        }
    }

    pub fn vk_stage(self) -> vk::ShaderStageFlags {
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

    pub fn glslang_stage_name(self) -> &'static str {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShaderId(pub usize);

pub struct ShaderModule {
    module: vk::ShaderModule,
    stage: ShaderStage,
    entry_point: CString,
    source_path: Option<PathBuf>,
}

impl GpuContext {
    pub fn create_shader_from_spirv_bytes(
        &mut self,
        spirv_bytes: &[u8],
        stage: ShaderStage,
        entry_point: &str,
    ) -> Result<ShaderId, GpuError> {
        let shader = ShaderModule::from_spirv_bytes(self.device(), spirv_bytes, stage, entry_point)?;
        let shaders = self.shaders_mut();
        let id = ShaderId(shaders.len());
        shaders.push(shader);
        Ok(id)
    }

    pub fn create_shader_from_spirv_file<P: AsRef<Path>>(
        &mut self,
        path: P,
        stage: ShaderStage,
        entry_point: &str,
    ) -> Result<ShaderId, GpuError> {
        let shader = ShaderModule::from_spirv_file(self.device(), path, stage, entry_point)?;
        let shaders = self.shaders_mut();
        let id = ShaderId(shaders.len());
        shaders.push(shader);
        Ok(id)
    }

    pub fn create_shader_from_glsl_source(
        &mut self,
        source: &str,
        stage: ShaderStage,
        source_name: &str,
        entry_point: &str,
    ) -> Result<ShaderId, GpuError> {
        let shader = ShaderModule::from_glsl_source(self.device(), source, stage, source_name, entry_point)?;
        let shaders = self.shaders_mut();
        let id = ShaderId(shaders.len());
        shaders.push(shader);
        Ok(id)
    }

    pub fn create_shader_from_glsl_file<P: AsRef<Path>>(
        &mut self,
        path: P,
        stage: ShaderStage,
        entry_point: &str,
    ) -> Result<ShaderId, GpuError> {
        let shader = ShaderModule::from_glsl_file(self.device(), path, stage, entry_point)?;
        let shaders = self.shaders_mut();
        let id = ShaderId(shaders.len());
        shaders.push(shader);
        Ok(id)
    }

    pub fn shader(&self, id: ShaderId) -> Option<&ShaderModule> {
        self.shaders_ref().get(id.0)
    }
}

impl ShaderModule {
    pub fn from_spirv_words(
        device: &ash::Device,
        spirv_words: &[u32],
        stage: ShaderStage,
        entry_point: &str,
    ) -> Result<Self, GpuError> {
        if spirv_words.is_empty() {
            return Err(std::io::Error::other("SPIR-V module is empty").into());
        }

        let entry_point = CString::new(entry_point)?;
        let create_info = vk::ShaderModuleCreateInfo::default().code(spirv_words);
        let module = unsafe { device.create_shader_module(&create_info, None)? };

        Ok(Self {
            module,
            stage,
            entry_point,
            source_path: None,
        })
    }

    pub fn from_spirv_bytes(
        device: &ash::Device,
        spirv_bytes: &[u8],
        stage: ShaderStage,
        entry_point: &str,
    ) -> Result<Self, GpuError> {
        if spirv_bytes.len() % std::mem::size_of::<u32>() != 0 {
            return Err(std::io::Error::other("SPIR-V byte length must be a multiple of 4").into());
        }

        let words: Vec<u32> = spirv_bytes
            .chunks_exact(std::mem::size_of::<u32>())
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Self::from_spirv_words(device, &words, stage, entry_point)
    }

    pub fn from_spirv_file<P: AsRef<Path>>(
        device: &ash::Device,
        path: P,
        stage: ShaderStage,
        entry_point: &str,
    ) -> Result<Self, GpuError> {
        let path_ref = path.as_ref();
        let bytes = fs::read(path_ref)?;
        let mut module = Self::from_spirv_bytes(device, &bytes, stage, entry_point)?;
        module.source_path = Some(path_ref.to_path_buf());
        Ok(module)
    }

    pub fn from_glsl_source(
        device: &ash::Device,
        source: &str,
        stage: ShaderStage,
        source_name: &str,
        entry_point: &str,
    ) -> Result<Self, GpuError> {
        let spirv = Self::compile_glsl_to_spirv(source, stage, source_name, entry_point)?;
        Self::from_spirv_words(device, &spirv, stage, entry_point)
    }

    pub fn from_glsl_file<P: AsRef<Path>>(
        device: &ash::Device,
        path: P,
        stage: ShaderStage,
        entry_point: &str,
    ) -> Result<Self, GpuError> {
        let path_ref = path.as_ref();
        let source = fs::read_to_string(path_ref)?;
        let source_name = path_ref.to_string_lossy();
        let mut module = Self::from_glsl_source(device, &source, stage, &source_name, entry_point)?;
        module.source_path = Some(path_ref.to_path_buf());
        Ok(module)
    }

    pub fn compile_glsl_to_spirv(
        source: &str,
        stage: ShaderStage,
        source_name: &str,
        entry_point: &str,
    ) -> Result<Vec<u32>, GpuError> {
        let temp_dir = std::env::temp_dir();
        let unique_tag = format!(
            "{}_{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos()
        );
        let src_path = temp_dir.join(format!("sound_engine_shader_{unique_tag}.glsl"));
        let spv_path = temp_dir.join(format!("sound_engine_shader_{unique_tag}.spv"));

        fs::write(&src_path, source)?;

        let output = Command::new("glslangValidator")
            .arg("-V")
            .arg("--target-env")
            .arg("vulkan1.3")
            .arg("-S")
            .arg(stage.glslang_stage_name())
            .arg("-e")
            .arg(entry_point)
            .arg("-o")
            .arg(&spv_path)
            .arg(&src_path)
            .output();

        let result = (|| -> Result<Vec<u32>, GpuError> {
            let output = output.map_err(|e| {
                std::io::Error::other(format!(
                    "Failed to invoke glslangValidator while compiling '{source_name}': {e}. \
Install Vulkan SDK tools and ensure glslangValidator is in PATH."
                ))
            })?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let stdout = String::from_utf8_lossy(&output.stdout);
                return Err(std::io::Error::other(format!(
                    "GLSL compilation failed for '{source_name}' ({stage:?}):\n{stdout}\n{stderr}"
                ))
                .into());
            }

            let spirv_bytes = fs::read(&spv_path)?;
            if spirv_bytes.len() % std::mem::size_of::<u32>() != 0 {
                return Err(std::io::Error::other(format!(
                    "Invalid SPIR-V output size from glslangValidator for '{source_name}'"
                ))
                .into());
            }

            Ok(spirv_bytes
                .chunks_exact(std::mem::size_of::<u32>())
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect())
        })();

        let _ = fs::remove_file(&src_path);
        let _ = fs::remove_file(&spv_path);

        result
    }

    pub fn handle(&self) -> vk::ShaderModule {
        self.module
    }

    pub fn stage(&self) -> ShaderStage {
        self.stage
    }

    pub fn domain(&self) -> ShaderDomain {
        self.stage.domain()
    }

    pub fn entry_point(&self) -> &CString {
        &self.entry_point
    }

    pub fn source_path(&self) -> Option<&Path> {
        self.source_path.as_deref()
    }

    pub fn pipeline_stage_info(&self) -> vk::PipelineShaderStageCreateInfo<'_> {
        vk::PipelineShaderStageCreateInfo::default()
            .stage(self.stage.vk_stage())
            .module(self.module)
            .name(self.entry_point.as_c_str())
    }
}
