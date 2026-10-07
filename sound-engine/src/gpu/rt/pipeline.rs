use super::*;
use crate::core::error::{SoundError, SoundResult};
use crate::gpu::shader::{SHADER_ENTRY_POINT, ShaderId, ShaderStage};
use std::ffi::CString;

/// Stable handle to a ray tracing pipeline stored in `RtContext`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RtPipelineId(pub u32);

/// Description used to create a Vulkan ray tracing pipeline and SBT.
pub struct RtPipelineSpec {
    /// Loaded shader IDs in the order referenced by shader groups.
    pub shaders: Vec<ShaderId>,
    /// Shader groups used by the SBT.
    pub groups: Vec<RtShaderGroupSpec>,
    /// Descriptor set layout bindings used during trace recording.
    pub descriptor_bindings: Vec<RtDescriptorBindingSpec>,
    /// Push-constant ranges available to the RT pipeline.
    pub push_constant_ranges: Vec<RtPushConstantSpec>,
    /// Maximum ray recursion depth passed to Vulkan.
    pub max_ray_recursion_depth: u32,
}

/// Shader group entry used to build the RT pipeline and SBT regions.
#[derive(Debug, Clone, Copy)]
pub enum RtShaderGroupSpec {
    /// Ray-generation group referencing a general shader stage index.
    Raygen {
        /// Ray-generation stage index in the pipeline shader list.
        shader: u32,
    },
    /// Miss group referencing a general shader stage index.
    Miss {
        /// Miss stage index in the pipeline shader list.
        shader: u32,
    },
    /// Triangle hit group referencing a closest-hit shader stage index.
    TrianglesHit {
        /// Closest-hit stage index in the pipeline shader list.
        closest_hit_shader: u32,
    },
    /// Procedural hit group referencing closest-hit and intersection shader stage indices.
    ProceduralHit {
        /// Closest-hit shader stage index.
        closest_hit_shader: u32,
        /// Intersection shader stage index.
        intersection_shader: u32,
    },
}

/// Single-element descriptor binding kinds supported by RT pipelines.
#[derive(Debug, Clone, Copy)]
pub enum RtDescriptorBindingSpec {
    /// Acceleration-structure descriptor binding.
    AccelerationStructure {
        /// Descriptor binding number in the set.
        binding: u32,
        /// Shader stages allowed to access the binding.
        stage_flags: vk::ShaderStageFlags,
    },
    /// Storage-buffer descriptor binding.
    StorageBuffer {
        /// Descriptor binding number in the set.
        binding: u32,
        /// Shader stages allowed to access the binding.
        stage_flags: vk::ShaderStageFlags,
    },
}

/// Push-constant byte range for an RT pipeline.
#[derive(Debug, Clone, Copy)]
pub struct RtPushConstantSpec {
    /// Byte offset within the pipeline layout.
    pub offset: u32,
    /// Byte size of the range.
    pub size: u32,
    /// Shader stages allowed to read the range.
    pub stage_flags: vk::ShaderStageFlags,
}

impl RtDescriptorBindingSpec {
    /// Creates a single acceleration-structure binding visible to RT stages.
    pub fn acceleration_structure(binding: u32) -> Self {
        Self::AccelerationStructure {
            binding,
            stage_flags: default_rt_stage_flags(),
        }
    }

    /// Creates a single storage-buffer binding visible to RT stages.
    pub fn storage_buffer(binding: u32) -> Self {
        Self::StorageBuffer {
            binding,
            stage_flags: default_rt_stage_flags(),
        }
    }
}

impl RtPushConstantSpec {
    /// Creates a push-constant range visible to all RT shader stages.
    pub fn new(offset: u32, size: u32) -> Self {
        Self {
            offset,
            size,
            stage_flags: default_rt_stage_flags(),
        }
    }
}

impl RtContext {
    /// Creates a ray tracing pipeline and its shader binding table.
    pub fn create_pipeline(&self, spec: RtPipelineSpec) -> SoundResult<RtPipelineId> {
        if spec.shaders.is_empty() {
            return Err(SoundError::invalid_argument(
                "RT pipeline requires at least one shader",
            ));
        }

        if spec.groups.is_empty() {
            return Err(SoundError::invalid_argument(
                "RT pipeline requires at least one shader group",
            ));
        }

        let entry = CString::new(SHADER_ENTRY_POINT)?;
        let stages = spec
            .shaders
            .iter()
            .map(|shader| self.pipeline_shader_stage(*shader, &entry))
            .collect::<SoundResult<Vec<_>>>()?;
        validate_groups(&stages, &spec.groups)?;

        let groups = spec
            .groups
            .iter()
            .map(build_shader_group)
            .collect::<Vec<_>>();
        let descriptor_bindings = build_vk_descriptor_bindings(&spec.descriptor_bindings);
        let push_constant_ranges = build_vk_push_constant_ranges(&spec.push_constant_ranges);
        if spec.max_ray_recursion_depth == 0
            || spec.max_ray_recursion_depth
                > self
                    .device_context
                    .rt_properties
                    .ray_tracing_pipeline
                    .max_ray_recursion_depth
        {
            return Err(SoundError::invalid_argument(
                "Ray recursion depth exceeds device limits",
            ));
        }
        let mut resources = PipelineResources::new(
            self.device_context.clone(),
            &descriptor_bindings,
            &push_constant_ranges,
        )?;

        let pipeline_info = vk::RayTracingPipelineCreateInfoKHR::default()
            .stages(&stages)
            .groups(&groups)
            .max_pipeline_ray_recursion_depth(spec.max_ray_recursion_depth)
            .layout(resources.layout);
        resources.handle = unsafe {
            self.device_context
                .ray_tracing_pipeline
                .create_ray_tracing_pipelines(
                    vk::DeferredOperationKHR::null(),
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_info),
                    None,
                )
                .map_err(|(partial, err)| {
                    for handle in partial {
                        self.device_context.device.destroy_pipeline(handle, None);
                    }
                    err
                })?[0]
        };

        // The SBT is derived from created pipeline group handles.
        let sbt = ShaderBindingTable::new(
            &self.device_context,
            &self.memory,
            resources.handle,
            &spec.groups,
        )?;

        let mut pipelines = self
            .pipelines
            .lock()
            .map_err(|_| SoundError::poisoned_lock("RT pipelines lock is poisoned"))?;

        let id = RtPipelineId(
            self.next_pipeline_id
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
                .map_err(|_| {
                    SoundError::resource_allocation_failed("RT pipeline id space exhausted")
                })?,
        );

        pipelines.insert(
            id,
            RayTracingPipeline {
                resources,
                push_constant_ranges,
                descriptor_pools: DescriptorPools::new(
                    self.device_context.clone(),
                    &descriptor_bindings,
                ),
                descriptor_bindings,
                sbt,
            },
        );
        Ok(id)
    }

    fn pipeline_shader_stage<'a>(
        &self,
        shader_id: ShaderId,
        entry: &'a CString,
    ) -> SoundResult<vk::PipelineShaderStageCreateInfo<'a>> {
        let stage = self.shaders.shader_stage(shader_id)?;
        let vk_stage = shader_stage_to_vk(stage)?;
        Ok(vk::PipelineShaderStageCreateInfo::default()
            .stage(vk_stage)
            .module(self.shaders.shader_module(shader_id)?)
            .name(entry.as_c_str()))
    }
}

/// Shader binding table region containing a pipeline group.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ShaderGroupKind {
    /// Single ray-generation record selected at trace launch.
    Raygen,
    /// Miss records indexed by the ray's miss-shader selection.
    Miss,
    /// Surface and procedural records indexed by instance and trace offsets.
    Hit,
}

impl RtShaderGroupSpec {
    /// Selects the SBT region used by this shader group.
    pub(super) fn kind(self) -> ShaderGroupKind {
        match self {
            Self::Raygen {
                ..
            } => ShaderGroupKind::Raygen,
            Self::Miss {
                ..
            } => ShaderGroupKind::Miss,
            Self::TrianglesHit {
                ..
            } => ShaderGroupKind::Hit,
            Self::ProceduralHit {
                ..
            } => ShaderGroupKind::Hit,
        }
    }
}

fn build_shader_group(spec: &RtShaderGroupSpec) -> vk::RayTracingShaderGroupCreateInfoKHR<'static> {
    match *spec {
        RtShaderGroupSpec::Raygen {
            shader: general_shader,
        }
        | RtShaderGroupSpec::Miss {
            shader: general_shader,
        } => vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(general_shader)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR),
        RtShaderGroupSpec::TrianglesHit {
            closest_hit_shader,
        } => vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
            .general_shader(vk::SHADER_UNUSED_KHR)
            .closest_hit_shader(closest_hit_shader)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR),
        RtShaderGroupSpec::ProceduralHit {
            closest_hit_shader,
            intersection_shader,
        } => vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP)
            .general_shader(vk::SHADER_UNUSED_KHR)
            .closest_hit_shader(closest_hit_shader)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(intersection_shader),
    }
}

fn build_vk_descriptor_bindings(
    specs: &[RtDescriptorBindingSpec],
) -> Vec<vk::DescriptorSetLayoutBinding<'static>> {
    specs
        .iter()
        .map(|spec| match *spec {
            RtDescriptorBindingSpec::AccelerationStructure {
                binding,
                stage_flags,
            } => vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1)
                .stage_flags(stage_flags),
            RtDescriptorBindingSpec::StorageBuffer {
                binding,
                stage_flags,
            } => vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(stage_flags),
        })
        .collect()
}

fn build_vk_push_constant_ranges(specs: &[RtPushConstantSpec]) -> Vec<vk::PushConstantRange> {
    specs
        .iter()
        .map(|spec| {
            vk::PushConstantRange::default()
                .stage_flags(spec.stage_flags)
                .offset(spec.offset)
                .size(spec.size)
        })
        .collect()
}

fn validate_groups(
    stages: &[vk::PipelineShaderStageCreateInfo<'_>],
    groups: &[RtShaderGroupSpec],
) -> SoundResult<()> {
    if groups
        .iter()
        .filter(|group| group.kind() == ShaderGroupKind::Raygen)
        .count()
        != 1
    {
        return Err(SoundError::invalid_argument(
            "SBT requires exactly one ray-generation group",
        ));
    }
    let mut runs = Vec::new();
    for group in groups {
        let kind = group.kind();
        if runs.last() != Some(&kind) {
            if runs.contains(&kind) {
                return Err(SoundError::invalid_argument(
                    "SBT groups of each kind must form one contiguous run",
                ));
            }
            runs.push(kind);
        }
    }
    for group in groups {
        match *group {
            RtShaderGroupSpec::Raygen {
                shader,
            } => validate_group_stage(stages, shader, vk::ShaderStageFlags::RAYGEN_KHR)?,
            RtShaderGroupSpec::Miss {
                shader,
            } => validate_group_stage(stages, shader, vk::ShaderStageFlags::MISS_KHR)?,
            RtShaderGroupSpec::TrianglesHit {
                closest_hit_shader,
            } => validate_group_stage(
                stages,
                closest_hit_shader,
                vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            )?,
            RtShaderGroupSpec::ProceduralHit {
                closest_hit_shader,
                intersection_shader,
            } => {
                validate_group_stage(
                    stages,
                    closest_hit_shader,
                    vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                )?;
                validate_group_stage(
                    stages,
                    intersection_shader,
                    vk::ShaderStageFlags::INTERSECTION_KHR,
                )?;
            }
        }
    }
    Ok(())
}

fn validate_group_stage(
    stages: &[vk::PipelineShaderStageCreateInfo<'_>],
    index: u32,
    expected: vk::ShaderStageFlags,
) -> SoundResult<()> {
    let stage = stages.get(index as usize).ok_or_else(|| {
        SoundError::invalid_argument(format!(
            "RT shader group references invalid shader index {index}"
        ))
    })?;
    if stage.stage != expected {
        return Err(SoundError::invalid_argument(format!(
            "RT shader group index {index} expects {expected:?}, got {:?}",
            stage.stage
        )));
    }
    Ok(())
}

fn shader_stage_to_vk(stage: ShaderStage) -> SoundResult<vk::ShaderStageFlags> {
    match stage {
        ShaderStage::RayGeneration => Ok(vk::ShaderStageFlags::RAYGEN_KHR),
        ShaderStage::RayMiss => Ok(vk::ShaderStageFlags::MISS_KHR),
        ShaderStage::RayClosestHit => Ok(vk::ShaderStageFlags::CLOSEST_HIT_KHR),
        ShaderStage::RayIntersection => Ok(vk::ShaderStageFlags::INTERSECTION_KHR),
        ShaderStage::Compute => Err(SoundError::invalid_argument(
            "Compute shader cannot be used in an RT pipeline",
        )),
    }
}

fn default_rt_stage_flags() -> vk::ShaderStageFlags {
    vk::ShaderStageFlags::RAYGEN_KHR
        | vk::ShaderStageFlags::MISS_KHR
        | vk::ShaderStageFlags::CLOSEST_HIT_KHR
        | vk::ShaderStageFlags::INTERSECTION_KHR
}
