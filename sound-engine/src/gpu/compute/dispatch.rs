use super::*;
use crate::core::error::ErrorCode;

/// One compute dispatch recorded into a command buffer.
pub struct DispatchSpec {
    /// Pipeline to bind before dispatch.
    pub pipeline_id: PipelineId,
    /// Descriptor writes used for this dispatch.
    pub descriptor_writes: Vec<DescriptorWrite>,
    /// Raw push-constant bytes to upload before dispatch.
    pub push_constants: Vec<u8>,
    /// Offset where `push_constants` are written.
    pub push_constant_offset: u32,
    /// Workgroup counts passed to `vkCmdDispatch`.
    pub groups: [u32; 3],
}

/// Descriptor write supported by compute dispatches.
pub enum DescriptorWrite {
    /// Storage buffer write for one descriptor binding.
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

impl DescriptorWrite {
    /// Creates a storage-buffer write covering the whole buffer.
    pub fn storage_buffer(binding: u32, buffer: vk::Buffer) -> Self {
        Self::StorageBuffer {
            binding,
            buffer,
            offset: 0,
            range: vk::WHOLE_SIZE,
        }
    }
}

impl ComputeContext {
    /// Records descriptor updates, pipeline bind, push constants, and dispatch.
    ///
    /// # Safety
    /// The command buffer must be recording inside this context's submission.
    /// Buffers must belong to this device, cover every shader access, and remain
    /// alive and synchronized through completion. Raw handles cannot prove this.
    pub unsafe fn record_dispatch(
        &self,
        command_buffer: vk::CommandBuffer,
        spec: &DispatchSpec,
    ) -> SoundResult<()> {
        if spec
            .groups
            .iter()
            .zip(self.device_context.limits.max_compute_work_group_count)
            .any(|(count, limit)| *count > limit)
        {
            return Err(SoundError::invalid_argument(
                "Compute group count exceeds device limits",
            ));
        }
        crate::validate!(
            spec.groups.iter().all(|group| *group > 0),
            ErrorCode::InvalidArgument,
            "Compute dispatch groups must all be non-zero, got {:?}",
            spec.groups
        );

        let mut pipelines = self
            .pipelines
            .lock()
            .map_err(|_| SoundError::poisoned_lock("Compute pipelines lock is poisoned"))?;

        let pipeline = pipelines.get_mut(&spec.pipeline_id).ok_or_else(|| {
            SoundError::not_found(format!("Invalid pipeline id {}", spec.pipeline_id.0))
        })?;

        let stages = crate::gpu::resource::push_constant_stages(
            &pipeline.push_constant_ranges,
            spec.push_constant_offset,
            spec.push_constants.len(),
        )?;
        let mut writes = Vec::with_capacity(spec.descriptor_writes.len());
        for write in &spec.descriptor_writes {
            let DescriptorWrite::StorageBuffer {
                binding,
                buffer,
                offset,
                range,
            } = *write;
            crate::gpu::resource::validate_storage_descriptor(
                buffer,
                offset,
                range,
                &self.device_context.limits,
            )?;
            writes.push((binding, vk::DescriptorType::STORAGE_BUFFER));
        }
        crate::gpu::resource::validate_descriptor_writes(&pipeline.descriptor_bindings, &writes)?;

        // Descriptor sets are short-lived and allocated from growable pools.
        let _pool_access = self.descriptor_pool_guard();
        let descriptor_allocation = self.acquire_descriptor_set(pipeline)?;
        let descriptor_set = descriptor_allocation.set;
        self.track_descriptor_set(command_buffer, descriptor_allocation)?;

        let buffer_infos = build_buffer_infos(spec);
        let writes = build_descriptor_writes(spec, descriptor_set, &buffer_infos);

        unsafe {
            self.device_context
                .device
                .update_descriptor_sets(&writes, &[]);
            self.device_context.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.handle,
            );

            self.device_context.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout,
                0,
                std::slice::from_ref(&descriptor_set),
                &[],
            );

            if !spec.push_constants.is_empty() {
                self.device_context.device.cmd_push_constants(
                    command_buffer,
                    pipeline.layout,
                    stages,
                    spec.push_constant_offset,
                    &spec.push_constants,
                );
            }

            self.device_context.device.cmd_dispatch(
                command_buffer,
                spec.groups[0],
                spec.groups[1],
                spec.groups[2],
            );
        }

        Ok(())
    }
}
fn build_buffer_infos(spec: &DispatchSpec) -> Vec<vk::DescriptorBufferInfo> {
    let mut buffer_infos = Vec::new();
    for write in &spec.descriptor_writes {
        match write {
            DescriptorWrite::StorageBuffer {
                buffer,
                offset,
                range,
                ..
            } => {
                buffer_infos.push(
                    vk::DescriptorBufferInfo::default()
                        .buffer(*buffer)
                        .offset(*offset)
                        .range(*range),
                );
            }
        }
    }
    buffer_infos
}

fn build_descriptor_writes<'a>(
    spec: &DispatchSpec,
    descriptor_set: vk::DescriptorSet,
    buffer_infos: &'a Vec<vk::DescriptorBufferInfo>,
) -> Vec<vk::WriteDescriptorSet<'a>> {
    let mut writes = Vec::new();
    for (idx, write) in spec.descriptor_writes.iter().enumerate() {
        match write {
            DescriptorWrite::StorageBuffer {
                binding,
                ..
            } => {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set)
                        .dst_binding(*binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&buffer_infos[idx])),
                );
            }
        }
    }
    writes
}
