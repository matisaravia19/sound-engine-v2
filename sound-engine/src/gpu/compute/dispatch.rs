use super::*;

pub(crate) struct DispatchSpec {
    pub pipeline_id: PipelineId,
    pub descriptor_writes: Vec<DescriptorWrite>,
    pub push_constants: Vec<u8>,
    pub push_constant_offset: u32,
    pub groups: [u32; 3],
}

pub(crate) enum DescriptorWrite {
    StorageBuffer {
        binding: u32,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
    },
}

impl DescriptorWrite {
    pub fn storage_buffer(binding: u32, buffer: vk::Buffer) -> Self {
        Self::StorageBuffer {
            binding,
            buffer,
            offset: 0,
            range: vk::WHOLE_SIZE,
        }
    }

    fn binding(&self) -> u32 {
        match self {
            DescriptorWrite::StorageBuffer { binding, .. } => *binding,
        }
    }
}

impl ComputeContext {
    pub fn record_dispatch(&self, command_buffer: vk::CommandBuffer, spec: &DispatchSpec) -> Result<(), GpuError> {
        let mut pipelines = self
            .pipelines
            .lock()
            .map_err(|_| std::io::Error::other("Compute pipelines lock is poisoned"))?;

        let pipeline = pipelines
            .get_mut(&spec.pipeline_id)
            .ok_or_else(|| std::io::Error::other(format!("Invalid pipeline id {}", spec.pipeline_id.0)))?;

        let descriptor_set = self.acquire_descriptor_set(pipeline)?;

        let buffer_infos = build_buffer_infos(spec);
        let writes = build_descriptor_writes(spec, descriptor_set, &buffer_infos);

        unsafe {
            self.device_context.device.update_descriptor_sets(&writes, &[]);
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
                    vk::ShaderStageFlags::COMPUTE,
                    spec.push_constant_offset,
                    &spec.push_constants,
                );
            }

            self.device_context
                .device
                .cmd_dispatch(command_buffer, spec.groups[0], spec.groups[1], spec.groups[2]);
        }

        Ok(())
    }
}

fn build_buffer_infos(spec: &DispatchSpec) -> Vec<vk::DescriptorBufferInfo> {
    let mut buffer_infos = Vec::new();
    for write in &spec.descriptor_writes {
        match write {
            DescriptorWrite::StorageBuffer {
                buffer, offset, range, ..
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
            DescriptorWrite::StorageBuffer { binding, .. } => {
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
