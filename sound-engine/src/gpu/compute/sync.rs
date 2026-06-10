use super::*;

/// Pipeline barrier for memory dependencies that are not buffer-specific.
pub struct MemoryBarrierSpec {
    /// Pipeline stages that must complete before the barrier.
    pub src_stage: vk::PipelineStageFlags,
    /// Pipeline stages that wait for the barrier.
    pub dst_stage: vk::PipelineStageFlags,
    /// Accesses that must be made available.
    pub src_access: vk::AccessFlags,
    /// Accesses that must be made visible.
    pub dst_access: vk::AccessFlags,
}

impl MemoryBarrierSpec {
    /// Synchronizes compute shader writes with later compute shader reads/writes.
    pub fn compute_shader_write_to_compute_shader_read_write() -> Self {
        Self {
            src_stage: vk::PipelineStageFlags::COMPUTE_SHADER,
            dst_stage: vk::PipelineStageFlags::COMPUTE_SHADER,
            src_access: vk::AccessFlags::SHADER_WRITE,
            dst_access: vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        }
    }
}

/// Pipeline barrier for dependencies on a single buffer.
pub struct BufferBarrierSpec {
    /// Pipeline stages that must complete before the barrier.
    pub src_stage: vk::PipelineStageFlags,
    /// Pipeline stages that wait for the barrier.
    pub dst_stage: vk::PipelineStageFlags,
    /// Accesses that must be made available.
    pub src_access: vk::AccessFlags,
    /// Accesses that must be made visible.
    pub dst_access: vk::AccessFlags,
    /// Queue family that releases ownership, or `QUEUE_FAMILY_IGNORED`.
    pub src_queue_family: u32,
    /// Queue family that acquires ownership, or `QUEUE_FAMILY_IGNORED`.
    pub dst_queue_family: u32,
    /// Buffer covered by the barrier.
    pub buffer: vk::Buffer,
    /// Byte offset into `buffer`.
    pub offset: vk::DeviceSize,
    /// Byte range covered by the barrier.
    pub size: vk::DeviceSize,
}

impl BufferBarrierSpec {
    /// Synchronizes compute shader writes with later compute shader reads/writes on the same buffer.
    pub fn compute_shader_write_to_compute_shader_read_write(buffer: vk::Buffer) -> Self {
        Self {
            src_stage: vk::PipelineStageFlags::COMPUTE_SHADER,
            dst_stage: vk::PipelineStageFlags::COMPUTE_SHADER,
            src_access: vk::AccessFlags::SHADER_WRITE,
            dst_access: vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            src_queue_family: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family: vk::QUEUE_FAMILY_IGNORED,
            buffer,
            offset: 0,
            size: vk::WHOLE_SIZE,
        }
    }
}

impl ComputeContext {
    /// Records a memory barrier into an already recording compute command buffer.
    pub fn record_memory_barrier(&self, command_buffer: vk::CommandBuffer, spec: &MemoryBarrierSpec) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(spec.src_access)
            .dst_access_mask(spec.dst_access);

        unsafe {
            self.device_context.device.cmd_pipeline_barrier(
                command_buffer,
                spec.src_stage,
                spec.dst_stage,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );
        }
    }

    /// Records a buffer barrier into an already recording compute command buffer.
    pub fn record_buffer_barrier(&self, command_buffer: vk::CommandBuffer, spec: &BufferBarrierSpec) {
        let barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(spec.src_access)
            .dst_access_mask(spec.dst_access)
            .src_queue_family_index(spec.src_queue_family)
            .dst_queue_family_index(spec.dst_queue_family)
            .buffer(spec.buffer)
            .offset(spec.offset)
            .size(spec.size);

        unsafe {
            self.device_context.device.cmd_pipeline_barrier(
                command_buffer,
                spec.src_stage,
                spec.dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                std::slice::from_ref(&barrier),
                &[],
            );
        }
    }
}
