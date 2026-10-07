use super::*;
use crate::gpu::shader::{SHADER_ENTRY_POINT, ShaderId, ShaderStage};
use std::ffi::CString;

/// Description used to create a compute pipeline and its descriptor layout.
pub struct ComputePipelineSpec {
    /// Shader module ID; it must refer to a compute shader.
    pub shader_id: ShaderId,
    /// Descriptor bindings exposed to the compute shader.
    pub descriptor_bindings: Vec<DescriptorBindingSpec>,
    /// Push-constant ranges available to dispatches.
    pub push_constant_ranges: Vec<PushConstantSpec>,
}

/// Descriptor binding kinds currently supported by compute pipelines.
pub enum DescriptorBindingSpec {
    /// Storage buffer binding with the given descriptor count.
    StorageBuffer {
        /// Descriptor binding number in the set.
        binding: u32,
        /// Number of storage-buffer descriptors in the binding.
        descriptor_count: u32,
    },
}

impl DescriptorBindingSpec {
    /// Creates a single storage-buffer binding.
    pub fn storage_buffer(binding: u32) -> Self {
        Self::StorageBuffer {
            binding,
            descriptor_count: 1,
        }
    }
}

/// Push-constant byte range available to a compute shader.
pub struct PushConstantSpec {
    /// Byte offset within the pipeline layout.
    pub offset: u32,
    /// Byte size of the range.
    pub size: u32,
}

impl PushConstantSpec {
    /// Creates a push-constant range.
    pub fn new(offset: u32, size: u32) -> Self {
        Self {
            offset,
            size,
        }
    }
}

impl ComputeContext {
    /// Creates and stores a Vulkan compute pipeline.
    pub fn create_pipeline(&self, spec: ComputePipelineSpec) -> SoundResult<PipelineId> {
        let shader_stage = self.shaders.shader_stage(spec.shader_id)?;
        if shader_stage != ShaderStage::Compute {
            return Err(SoundError::invalid_argument(format!(
                "create_pipeline requires compute shader, got {shader_stage:?}"
            )));
        }

        let shader_module = self.shaders.shader_module(spec.shader_id)?;
        let descriptor_bindings = build_vk_descriptor_bindings(&spec.descriptor_bindings);
        let push_constant_ranges = build_vk_push_constant_ranges(&spec.push_constant_ranges);
        crate::gpu::resource::validate_pipeline_layout(
            &descriptor_bindings,
            &push_constant_ranges,
            &self.device_context.limits,
        )?;
        let mut cleanup =
            crate::gpu::resource::PipelineBuildGuard::new(self.device_context.clone());

        // Pipeline layout is built from exactly one descriptor set layout for now.
        let descriptor_set_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_bindings);
        let descriptor_set_layout = unsafe {
            self.device_context
                .device
                .create_descriptor_set_layout(&descriptor_set_layout_info, None)?
        };

        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(&push_constant_ranges);
        cleanup.descriptor_set_layout = descriptor_set_layout;
        let layout = unsafe {
            self.device_context
                .device
                .create_pipeline_layout(&layout_info, None)?
        };

        let entry = CString::new(SHADER_ENTRY_POINT)?;
        cleanup.layout = layout;
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(entry.as_c_str());

        let compute_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(layout);

        let pipeline = unsafe {
            self.device_context
                .device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&compute_info),
                    None,
                )
                .map_err(|(partial, err)| {
                    for handle in partial {
                        self.device_context.device.destroy_pipeline(handle, None);
                    }
                    err
                })?[0]
        };
        cleanup.handle = pipeline;

        let mut pipelines = self
            .pipelines
            .lock()
            .map_err(|_| SoundError::poisoned_lock("Compute pipelines lock is poisoned"))?;

        let descriptor_pools =
            DescriptorPools::new(self.device_context.clone(), &descriptor_bindings);

        let id = PipelineId(
            self.next_pipeline_id
                .fetch_update(
                    std::sync::atomic::Ordering::Relaxed,
                    std::sync::atomic::Ordering::Relaxed,
                    |id| id.checked_add(1),
                )
                .map_err(|_| {
                    SoundError::resource_allocation_failed("Compute pipeline id space exhausted")
                })?,
        );
        pipelines.insert(
            id,
            ComputePipeline {
                device_context: self.device_context.clone(),
                descriptor_bindings,
                handle: pipeline,
                layout,
                descriptor_set_layout,
                push_constant_ranges,
                descriptor_pools,
            },
        );
        cleanup.disarm();
        Ok(id)
    }
}

fn build_vk_descriptor_bindings(
    specs: &[DescriptorBindingSpec],
) -> Vec<vk::DescriptorSetLayoutBinding<'static>> {
    let mut bindings = Vec::with_capacity(specs.len());
    for spec in specs {
        match spec {
            DescriptorBindingSpec::StorageBuffer {
                binding,
                descriptor_count,
            } => bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(*binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(*descriptor_count)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            ),
        }
    }
    bindings
}

fn build_vk_push_constant_ranges(specs: &[PushConstantSpec]) -> Vec<vk::PushConstantRange> {
    let mut ranges = Vec::with_capacity(specs.len());
    for spec in specs {
        ranges.push(
            vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(spec.offset)
                .size(spec.size),
        );
    }
    ranges
}
