use super::*;
use crate::error::{ErrorCode, SoundError, SoundResult};

/// One ray tracing dispatch recorded into a command buffer.
pub struct RtTraceSpec {
    /// Pipeline to bind before `vkCmdTraceRaysKHR`.
    pub pipeline_id: RtPipelineId,
    /// Descriptor writes used for this trace.
    pub descriptor_writes: Vec<RtDescriptorWrite>,
    /// Raw push-constant bytes to upload before tracing.
    pub push_constants: Vec<u8>,
    /// Offset where `push_constants` are written.
    pub push_constant_offset: u32,
    /// Trace dimensions passed to `vkCmdTraceRaysKHR`.
    pub dimensions: [u32; 3],
    /// Whether to add a shader-write to host-read barrier after tracing.
    pub barrier_after_trace: bool,
}

/// Descriptor write supported by RT trace dispatches.
pub enum RtDescriptorWrite {
    /// Writes a TLAS acceleration-structure descriptor.
    AccelerationStructure {
        /// Descriptor binding number to update.
        binding: u32,
        /// TLAS to bind at `binding`.
        tlas_id: TlasId,
    },
    /// Writes a storage-buffer descriptor.
    StorageBuffer {
        /// Descriptor binding number to update.
        binding: u32,
        /// Vulkan buffer handle to bind.
        buffer: vk::Buffer,
        /// Byte offset into the buffer.
        offset: vk::DeviceSize,
        /// Byte range exposed to the shader.
        range: vk::DeviceSize,
    },
}

impl RtDescriptorWrite {
    /// Creates an acceleration-structure descriptor write.
    pub fn acceleration_structure(binding: u32, tlas_id: TlasId) -> Self {
        Self::AccelerationStructure { binding, tlas_id }
    }

    /// Creates a storage-buffer descriptor write covering the whole buffer.
    pub fn storage_buffer(binding: u32, buffer: vk::Buffer) -> Self {
        Self::StorageBuffer {
            binding,
            buffer,
            offset: 0,
            range: vk::WHOLE_SIZE,
        }
    }
}

impl RtContext {
    /// Records descriptor updates, pipeline bind, push constants, and ray trace.
    pub fn record_trace(&self, command_buffer: vk::CommandBuffer, spec: &RtTraceSpec) -> SoundResult<()> {
        crate::debug_validate!(
            spec.dimensions.iter().all(|dimension| *dimension > 0),
            ErrorCode::InvalidArgument,
            "RT trace dimensions must all be non-zero, got {:?}",
            spec.dimensions
        );

        let mut pipelines = self
            .pipelines
            .lock()
            .map_err(|_| SoundError::poisoned_lock("RT pipelines lock is poisoned"))?;

        let pipeline = pipelines
            .get_mut(&spec.pipeline_id)
            .ok_or_else(|| SoundError::not_found(format!("Invalid RT pipeline id {}", spec.pipeline_id.0)))?;

        crate::debug_validate!(
            push_constants_fit_pipeline(spec, pipeline),
            ErrorCode::InvalidArgument,
            "RT push constants offset={} size={} do not fit pipeline ranges",
            spec.push_constant_offset,
            spec.push_constants.len()
        );

        // Descriptor sets are allocated per trace from growable pools.
        let descriptor_set = self.acquire_descriptor_set(pipeline)?;
        self.update_descriptor_set(descriptor_set, &spec.descriptor_writes)?;

        unsafe {
            self.device_context.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                pipeline.handle,
            );
            self.device_context.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                pipeline.layout,
                0,
                std::slice::from_ref(&descriptor_set),
                &[],
            );

            if !spec.push_constants.is_empty() {
                self.device_context.device.cmd_push_constants(
                    command_buffer,
                    pipeline.layout,
                    default_trace_push_constant_stages(),
                    spec.push_constant_offset,
                    &spec.push_constants,
                );
            }

            self.device_context.ray_tracing_pipeline.cmd_trace_rays(
                command_buffer,
                &pipeline.raygen_region,
                &pipeline.miss_region,
                &pipeline.hit_region,
                &pipeline.callable_region,
                spec.dimensions[0],
                spec.dimensions[1],
                spec.dimensions[2],
            );

            if spec.barrier_after_trace {
                // Required when the host reads storage-buffer results immediately.
                let barrier = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ);
                self.device_context.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    std::slice::from_ref(&barrier),
                    &[],
                    &[],
                );
            }
        }

        Ok(())
    }

    fn update_descriptor_set(
        &self,
        descriptor_set: vk::DescriptorSet,
        writes: &[RtDescriptorWrite],
    ) -> SoundResult<()> {
        let tlas = self
            .tlas
            .lock()
            .map_err(|_| SoundError::poisoned_lock("RT TLAS lock is poisoned"))?;

        for write in writes {
            match *write {
                RtDescriptorWrite::AccelerationStructure { binding, tlas_id } => {
                    let tlas = tlas
                        .get(&tlas_id)
                        .ok_or_else(|| SoundError::not_found(format!("Invalid TLAS id {}", tlas_id.0)))?;
                    let mut as_write = vk::WriteDescriptorSetAccelerationStructureKHR::default()
                        .acceleration_structures(std::slice::from_ref(&tlas.handle));
                    let descriptor_write = vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(binding)
                        .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                        .descriptor_count(1)
                        .push_next(&mut as_write);
                    unsafe {
                        self.device_context
                            .device
                            .update_descriptor_sets(std::slice::from_ref(&descriptor_write), &[]);
                    }
                }
                RtDescriptorWrite::StorageBuffer {
                    binding,
                    buffer,
                    offset,
                    range,
                } => {
                    let buffer_info = vk::DescriptorBufferInfo::default()
                        .buffer(buffer)
                        .offset(offset)
                        .range(range);
                    let descriptor_write = vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&buffer_info));
                    unsafe {
                        self.device_context
                            .device
                            .update_descriptor_sets(std::slice::from_ref(&descriptor_write), &[]);
                    }
                }
            }
        }

        Ok(())
    }
}

fn push_constants_fit_pipeline(spec: &RtTraceSpec, pipeline: &RayTracingPipeline) -> bool {
    if spec.push_constants.is_empty() {
        return true;
    }

    let start = spec.push_constant_offset;
    let Some(end) = start.checked_add(spec.push_constants.len() as u32) else {
        return false;
    };

    pipeline.push_constant_ranges.iter().any(|range| {
        let range_end = range.offset.saturating_add(range.size);
        start >= range.offset && end <= range_end
    })
}

fn default_trace_push_constant_stages() -> vk::ShaderStageFlags {
    vk::ShaderStageFlags::RAYGEN_KHR
        | vk::ShaderStageFlags::MISS_KHR
        | vk::ShaderStageFlags::CLOSEST_HIT_KHR
        | vk::ShaderStageFlags::ANY_HIT_KHR
        | vk::ShaderStageFlags::INTERSECTION_KHR
        | vk::ShaderStageFlags::CALLABLE_KHR
}
