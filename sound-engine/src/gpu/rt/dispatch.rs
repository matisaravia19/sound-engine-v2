use super::*;
use crate::core::error::{ErrorCode, SoundError, SoundResult};

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
        Self::AccelerationStructure {
            binding,
            tlas_id,
        }
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
    ///
    /// # Safety
    /// Record inside this context's compute submission. Buffers, TLAS/BLAS and
    /// device addresses must belong to this device and remain alive through
    /// completion. Their sizes, contents and SBT offsets must match the shaders;
    /// callers must synchronize all accesses to the referenced resources.
    pub unsafe fn record_trace(
        &self,
        command_buffer: vk::CommandBuffer,
        spec: &RtTraceSpec,
    ) -> SoundResult<()> {
        let invocations = spec
            .dimensions
            .iter()
            .try_fold(1_u64, |count, size| count.checked_mul(*size as u64));
        if invocations.is_none_or(|count| {
            count
                > self
                    .device_context
                    .rt_properties
                    .ray_tracing_pipeline
                    .max_ray_dispatch_invocation_count as u64
        }) {
            return Err(SoundError::invalid_argument(
                "RT invocation count exceeds device limits",
            ));
        }
        crate::validate!(
            spec.dimensions.iter().all(|dimension| *dimension > 0),
            ErrorCode::InvalidArgument,
            "RT trace dimensions must all be non-zero, got {:?}",
            spec.dimensions
        );

        let mut pipelines = self
            .pipelines
            .lock()
            .map_err(|_| SoundError::poisoned_lock("RT pipelines lock is poisoned"))?;

        let pipeline = pipelines.get_mut(&spec.pipeline_id).ok_or_else(|| {
            SoundError::not_found(format!("Invalid RT pipeline id {}", spec.pipeline_id.0))
        })?;

        let stages = crate::gpu::resource::push_constant_stages(
            &pipeline.push_constant_ranges,
            spec.push_constant_offset,
            spec.push_constants.len(),
        )?;
        let mut writes = Vec::with_capacity(spec.descriptor_writes.len());
        for write in &spec.descriptor_writes {
            match *write {
                RtDescriptorWrite::AccelerationStructure {
                    binding,
                    ..
                } => writes.push((binding, vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)),
                RtDescriptorWrite::StorageBuffer {
                    binding,
                    buffer,
                    offset,
                    range,
                } => {
                    crate::gpu::resource::validate_storage_descriptor(
                        buffer,
                        offset,
                        range,
                        &self.device_context.limits,
                    )?;
                    writes.push((binding, vk::DescriptorType::STORAGE_BUFFER));
                }
            }
        }
        crate::gpu::resource::validate_descriptor_writes(&pipeline.descriptor_bindings, &writes)?;

        // Descriptor sets are allocated per trace from growable pools.
        let _pool_access = self.compute.descriptor_pool_guard();
        let descriptor_allocation = pipeline
            .descriptor_pools
            .allocate(pipeline.resources.descriptor_set_layout)?;
        let descriptor_set = descriptor_allocation.set;
        self.compute
            .track_descriptor_set(command_buffer, descriptor_allocation);
        self.update_descriptor_set(descriptor_set, &spec.descriptor_writes)?;

        unsafe {
            self.device_context.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                pipeline.resources.handle,
            );
            self.device_context.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                pipeline.resources.layout,
                0,
                std::slice::from_ref(&descriptor_set),
                &[],
            );

            if !spec.push_constants.is_empty() {
                self.device_context.device.cmd_push_constants(
                    command_buffer,
                    pipeline.resources.layout,
                    stages,
                    spec.push_constant_offset,
                    &spec.push_constants,
                );
            }

            self.device_context.ray_tracing_pipeline.cmd_trace_rays(
                command_buffer,
                &pipeline.sbt.raygen,
                &pipeline.sbt.miss,
                &pipeline.sbt.hit,
                &vk::StridedDeviceAddressRegionKHR::default(),
                spec.dimensions[0],
                spec.dimensions[1],
                spec.dimensions[2],
            );
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
                RtDescriptorWrite::AccelerationStructure {
                    binding,
                    tlas_id,
                } => {
                    let tlas = tlas.get(&tlas_id).ok_or_else(|| {
                        SoundError::not_found(format!("Invalid TLAS id {}", tlas_id.0))
                    })?;
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
