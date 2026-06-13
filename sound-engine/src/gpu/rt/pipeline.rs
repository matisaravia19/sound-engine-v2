use super::descriptor::aggregate_pool_sizes;
use super::*;
use crate::error::{SoundError, SoundResult};
use crate::gpu::memory::GpuAllocator;
use crate::gpu::shader::{ShaderId, ShaderStage, SHADER_ENTRY_POINT};
use std::ffi::CString;

/// Stable handle to a ray tracing pipeline stored in `RtContext`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RtPipelineId(pub u32);

/// Description used to create a Vulkan ray tracing pipeline and SBT.
pub struct RtPipelineSpec {
    /// Shader stages in the order referenced by shader groups.
    pub shaders: Vec<RtShaderStageSpec>,
    /// Shader groups used by the SBT.
    pub groups: Vec<RtShaderGroupSpec>,
    /// Descriptor set layout bindings used during trace recording.
    pub descriptor_bindings: Vec<RtDescriptorBindingSpec>,
    /// Push-constant ranges available to the RT pipeline.
    pub push_constant_ranges: Vec<RtPushConstantSpec>,
    /// Maximum ray recursion depth passed to Vulkan.
    pub max_ray_recursion_depth: u32,
}

/// One shader stage in an RT pipeline.
#[derive(Debug, Clone, Copy)]
pub struct RtShaderStageSpec {
    /// Shader module ID previously loaded through `ShaderLibrary`.
    pub shader_id: ShaderId,
}

/// Shader group entry used to build the RT pipeline and SBT regions.
#[derive(Debug, Clone, Copy)]
pub enum RtShaderGroupSpec {
    /// Ray-generation group referencing a general shader stage index.
    Raygen { shader: u32 },
    /// Miss group referencing a general shader stage index.
    Miss { shader: u32 },
    /// Triangle hit group referencing a closest-hit shader stage index.
    TrianglesHit { closest_hit_shader: u32 },
    /// Callable group referencing a general shader stage index.
    Callable { shader: u32 },
}

/// Descriptor binding kinds supported by RT pipelines.
#[derive(Debug, Clone, Copy)]
pub enum RtDescriptorBindingSpec {
    /// Acceleration-structure descriptor binding.
    AccelerationStructure {
        /// Descriptor binding number in the set.
        binding: u32,
        /// Number of descriptors in the binding.
        descriptor_count: u32,
        /// Shader stages allowed to access the binding.
        stage_flags: vk::ShaderStageFlags,
    },
    /// Storage-buffer descriptor binding.
    StorageBuffer {
        /// Descriptor binding number in the set.
        binding: u32,
        /// Number of descriptors in the binding.
        descriptor_count: u32,
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

impl RtShaderStageSpec {
    /// Creates a shader stage spec from an existing shader ID.
    pub fn new(shader_id: ShaderId) -> Self {
        Self { shader_id }
    }
}

impl RtDescriptorBindingSpec {
    /// Creates a single acceleration-structure binding visible to RT stages.
    pub fn acceleration_structure(binding: u32) -> Self {
        Self::AccelerationStructure {
            binding,
            descriptor_count: 1,
            stage_flags: default_rt_stage_flags(),
        }
    }

    /// Creates a single storage-buffer binding visible to RT stages.
    pub fn storage_buffer(binding: u32) -> Self {
        Self::StorageBuffer {
            binding,
            descriptor_count: 1,
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
    pub fn create_pipeline(&self, memory: &GpuAllocator, spec: RtPipelineSpec) -> SoundResult<RtPipelineId> {
        if spec.shaders.is_empty() {
            return Err(SoundError::invalid_argument("RT pipeline requires at least one shader"));
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
            .map(|stage| self.pipeline_shader_stage(*stage, &entry))
            .collect::<SoundResult<Vec<_>>>()?;
        validate_groups(&stages, &spec.groups)?;

        let groups = spec.groups.iter().map(build_shader_group).collect::<Vec<_>>();
        let descriptor_bindings = build_vk_descriptor_bindings(&spec.descriptor_bindings);
        let push_constant_ranges = build_vk_push_constant_ranges(&spec.push_constant_ranges);

        // RT pipelines use one descriptor set layout for now, matching compute.
        let descriptor_set_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_bindings);
        let descriptor_set_layout = unsafe {
            self.device_context
                .device
                .create_descriptor_set_layout(&descriptor_set_layout_info, None)?
        };

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(&push_constant_ranges);
        let layout = unsafe { self.device_context.device.create_pipeline_layout(&layout_info, None)? };

        let pipeline_info = vk::RayTracingPipelineCreateInfoKHR::default()
            .stages(&stages)
            .groups(&groups)
            .max_pipeline_ray_recursion_depth(spec.max_ray_recursion_depth)
            .layout(layout);
        let handle = unsafe {
            self.device_context
                .ray_tracing_pipeline
                .create_ray_tracing_pipelines(
                    vk::DeferredOperationKHR::null(),
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_info),
                    None,
                )
                .map_err(|(_, err)| err)?[0]
        };

        // The SBT is derived from created pipeline group handles.
        let (sbt, raygen_region, miss_region, hit_region, callable_region) =
            build_sbt(memory, self.device_context.clone(), handle, &spec.groups)?;

        let mut pipelines = self
            .pipelines
            .lock()
            .map_err(|_| SoundError::poisoned_lock("RT pipelines lock is poisoned"))?;

        let id = RtPipelineId(pipelines.len() as u32);

        pipelines.insert(
            id,
            RayTracingPipeline {
                handle,
                layout,
                descriptor_set_layout,
                push_constant_ranges,
                pool_sizes_template: aggregate_pool_sizes(&descriptor_bindings),
                descriptor_pools: Vec::new(),
                sbt,
                raygen_region,
                miss_region,
                hit_region,
                callable_region,
            },
        );
        Ok(id)
    }

    fn pipeline_shader_stage<'a>(
        &self,
        spec: RtShaderStageSpec,
        entry: &'a CString,
    ) -> SoundResult<vk::PipelineShaderStageCreateInfo<'a>> {
        let stage = self.shaders.shader_stage(spec.shader_id)?;
        let vk_stage = shader_stage_to_vk(stage)?;
        Ok(vk::PipelineShaderStageCreateInfo::default()
            .stage(vk_stage)
            .module(self.shaders.shader_module(spec.shader_id)?)
            .name(entry.as_c_str()))
    }
}

fn build_sbt(
    memory: &GpuAllocator,
    device: Arc<VkDeviceContext>,
    pipeline: vk::Pipeline,
    groups: &[RtShaderGroupSpec],
) -> SoundResult<(
    BufferHandle,
    vk::StridedDeviceAddressRegionKHR,
    vk::StridedDeviceAddressRegionKHR,
    vk::StridedDeviceAddressRegionKHR,
    vk::StridedDeviceAddressRegionKHR,
)> {
    let props = &device.rt_properties.ray_tracing_pipeline;
    let handle_size = props.shader_group_handle_size as usize;
    let handle_alignment = props.shader_group_handle_alignment as usize;
    let base_alignment = props.shader_group_base_alignment as usize;
    let record_stride = align_up(handle_size, handle_alignment);
    let group_count = groups.len() as u32;
    let region_size = align_up(record_stride, base_alignment);
    let total_size = region_size * groups.len();

    let handles = unsafe {
        device.ray_tracing_pipeline.get_ray_tracing_shader_group_handles(
            pipeline,
            0,
            group_count,
            handle_size * groups.len(),
        )?
    };

    // Each group handle is copied into a base-aligned record slot.
    let mut sbt_bytes = vec![0_u8; total_size];
    for group_idx in 0..groups.len() {
        let src_offset = group_idx * handle_size;
        let dst_offset = group_idx * region_size;
        sbt_bytes[dst_offset..dst_offset + handle_size].copy_from_slice(&handles[src_offset..src_offset + handle_size]);
    }

    // Host-visible SBT is fine for now and avoids a staging copy.
    let sbt = memory.create_host_device_address_buffer(
        sbt_bytes.len() as vk::DeviceSize,
        vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
    )?;
    memory.write_mapped_bytes(&sbt, &sbt_bytes)?;
    let base_address = memory.buffer_device_address(&sbt);

    Ok((
        sbt,
        region_for(groups, base_address, record_stride, region_size, RtGroupKind::Raygen),
        region_for(groups, base_address, record_stride, region_size, RtGroupKind::Miss),
        region_for(groups, base_address, record_stride, region_size, RtGroupKind::Hit),
        region_for(groups, base_address, record_stride, region_size, RtGroupKind::Callable),
    ))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RtGroupKind {
    Raygen,
    Miss,
    Hit,
    Callable,
}

fn region_for(
    groups: &[RtShaderGroupSpec],
    base_address: vk::DeviceAddress,
    record_stride: usize,
    region_size: usize,
    kind: RtGroupKind,
) -> vk::StridedDeviceAddressRegionKHR {
    // SBT regions are contiguous runs by group kind.
    let start = groups.iter().position(|group| group.kind() == kind);
    let Some(start) = start else {
        return vk::StridedDeviceAddressRegionKHR::default();
    };
    let count = groups[start..].iter().take_while(|group| group.kind() == kind).count();

    vk::StridedDeviceAddressRegionKHR {
        device_address: base_address + (start * region_size) as vk::DeviceAddress,
        stride: record_stride as vk::DeviceSize,
        size: (count * record_stride) as vk::DeviceSize,
    }
}

impl RtShaderGroupSpec {
    fn kind(self) -> RtGroupKind {
        match self {
            Self::Raygen { .. } => RtGroupKind::Raygen,
            Self::Miss { .. } => RtGroupKind::Miss,
            Self::TrianglesHit { .. } => RtGroupKind::Hit,
            Self::Callable { .. } => RtGroupKind::Callable,
        }
    }
}

fn build_shader_group(spec: &RtShaderGroupSpec) -> vk::RayTracingShaderGroupCreateInfoKHR<'static> {
    match *spec {
        RtShaderGroupSpec::Raygen { shader: general_shader }
        | RtShaderGroupSpec::Miss { shader: general_shader }
        | RtShaderGroupSpec::Callable { shader: general_shader } => vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(general_shader)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR),
        RtShaderGroupSpec::TrianglesHit { closest_hit_shader } => vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
            .general_shader(vk::SHADER_UNUSED_KHR)
            .closest_hit_shader(closest_hit_shader)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR),
    }
}

fn build_vk_descriptor_bindings(specs: &[RtDescriptorBindingSpec]) -> Vec<vk::DescriptorSetLayoutBinding<'static>> {
    specs
        .iter()
        .map(|spec| match *spec {
            RtDescriptorBindingSpec::AccelerationStructure {
                binding,
                descriptor_count,
                stage_flags,
            } => vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(descriptor_count)
                .stage_flags(stage_flags),
            RtDescriptorBindingSpec::StorageBuffer {
                binding,
                descriptor_count,
                stage_flags,
            } => vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(descriptor_count)
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

fn validate_groups(stages: &[vk::PipelineShaderStageCreateInfo<'_>], groups: &[RtShaderGroupSpec]) -> SoundResult<()> {
    for group in groups {
        match *group {
            RtShaderGroupSpec::Raygen { shader } => {
                validate_group_stage(stages, shader, vk::ShaderStageFlags::RAYGEN_KHR)?
            }
            RtShaderGroupSpec::Miss { shader } => validate_group_stage(stages, shader, vk::ShaderStageFlags::MISS_KHR)?,
            RtShaderGroupSpec::Callable { shader } => {
                validate_group_stage(stages, shader, vk::ShaderStageFlags::CALLABLE_KHR)?
            }
            RtShaderGroupSpec::TrianglesHit { closest_hit_shader } => {
                validate_group_stage(stages, closest_hit_shader, vk::ShaderStageFlags::CLOSEST_HIT_KHR)?
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
        SoundError::invalid_argument(format!("RT shader group references invalid shader index {index}"))
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
        ShaderStage::RayAnyHit => Ok(vk::ShaderStageFlags::ANY_HIT_KHR),
        ShaderStage::RayIntersection => Ok(vk::ShaderStageFlags::INTERSECTION_KHR),
        ShaderStage::RayCallable => Ok(vk::ShaderStageFlags::CALLABLE_KHR),
        ShaderStage::Compute => Err(SoundError::invalid_argument(
            "Compute shader cannot be used in an RT pipeline",
        )),
    }
}

fn default_rt_stage_flags() -> vk::ShaderStageFlags {
    vk::ShaderStageFlags::RAYGEN_KHR
        | vk::ShaderStageFlags::MISS_KHR
        | vk::ShaderStageFlags::CLOSEST_HIT_KHR
        | vk::ShaderStageFlags::ANY_HIT_KHR
        | vk::ShaderStageFlags::INTERSECTION_KHR
        | vk::ShaderStageFlags::CALLABLE_KHR
}
