use crate::gpu::GpuError;
use crate::gpu::backend::{QueueSet, VkDeviceContext};
use crate::gpu::shader::{SHADER_ENTRY_POINT, ShaderId, ShaderLibrary, ShaderStage};
use ash::vk;
use std::collections::HashMap;
use std::ffi::CString;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

const COMMAND_BUFFER_POOL_SIZE: u32 = 12;

pub(crate) struct ComputeContext {
    device_context: Arc<VkDeviceContext>,
    shaders: Arc<ShaderLibrary>,
    next_token: AtomicU64,
    in_flight: Mutex<HashMap<u64, InFlightSubmission>>,
    queue: RwLock<QueueSet>,
    available_command_buffers: Mutex<Vec<vk::CommandBuffer>>,
    pipelines: Mutex<HashMap<PipelineId, ComputePipeline>>,
}

pub(crate) struct FrameToken(pub u64);

struct InFlightSubmission {
    fence: vk::Fence,
    command_buffer: vk::CommandBuffer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PipelineId(u32);

struct ComputePipeline {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
}

impl ComputeContext {
    pub(super) fn new(
        device_context: Arc<VkDeviceContext>,
        shaders: Arc<ShaderLibrary>,
        compute_queue: QueueSet,
    ) -> Result<Self, GpuError> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(compute_queue.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(COMMAND_BUFFER_POOL_SIZE);
        let preallocated_command_buffers = unsafe { device_context.device.allocate_command_buffers(&alloc_info)? };

        Ok(Self {
            device_context,
            shaders,
            next_token: AtomicU64::new(1),
            in_flight: Mutex::new(HashMap::new()),
            queue: RwLock::new(compute_queue),
            available_command_buffers: Mutex::new(preallocated_command_buffers),
            pipelines: Mutex::new(HashMap::new()),
        })
    }

    pub fn submit_compute<F>(&self, record: F) -> Result<FrameToken, GpuError>
    where
        F: FnOnce(vk::CommandBuffer) -> Result<(), GpuError>,
    {
        let token = self.next_token.fetch_add(1, Ordering::Relaxed);
        let fence = self.device_context.create_fence()?;

        let queue = self
            .queue
            .write()
            .map_err(|_| std::io::Error::other("Compute queue lock is poisoned"))?;

        let command_buffer = self.acquire_command_buffer(&queue)?;

        unsafe {
            self.device_context
                .device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;

            let begin_info = vk::CommandBufferBeginInfo::default();
            self.device_context
                .device
                .begin_command_buffer(command_buffer, &begin_info)?;
        }

        record(command_buffer)?;

        unsafe {
            self.device_context.device.end_command_buffer(command_buffer)?;
        }

        let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));
        unsafe {
            self.device_context
                .device
                .queue_submit(queue.handle, std::slice::from_ref(&submit_info), fence)?;
        }

        self.in_flight
            .lock()
            .map_err(|_| std::io::Error::other("Compute in_flight lock is poisoned"))?
            .insert(token, InFlightSubmission { fence, command_buffer });

        Ok(FrameToken(token))
    }

    fn acquire_command_buffer(&self, queue: &QueueSet) -> Result<vk::CommandBuffer, GpuError> {
        if let Some(command_buffer) = self
            .available_command_buffers
            .lock()
            .map_err(|_| std::io::Error::other("Compute command buffer pool lock is poisoned"))?
            .pop()
        {
            return Ok(command_buffer);
        }

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(queue.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe { self.device_context.device.allocate_command_buffers(&alloc_info)? }[0];
        Ok(command_buffer)
    }

    pub fn submit_compute_and_wait<F>(&self, record: F) -> Result<(), GpuError>
    where
        F: FnOnce(vk::CommandBuffer) -> Result<(), GpuError>,
    {
        let token = self.submit_compute(record)?;
        self.wait_for(token)
    }

    pub fn wait_for(&self, token: FrameToken) -> Result<(), GpuError> {
        let in_flight = self
            .in_flight
            .lock()
            .map_err(|_| std::io::Error::other("Compute in_flight lock is poisoned"))?
            .remove(&token.0)
            .ok_or_else(|| std::io::Error::other(format!("Unknown frame token {}", token.0)))?;

        unsafe {
            self.device_context
                .device
                .wait_for_fences(std::slice::from_ref(&in_flight.fence), true, u64::MAX)?;
            self.device_context.device.destroy_fence(in_flight.fence, None);
        }

        self.available_command_buffers
            .lock()
            .map_err(|_| std::io::Error::other("Compute command buffer pool lock is poisoned"))?
            .push(in_flight.command_buffer);

        Ok(())
    }

    pub fn create_pipeline(
        &self,
        shader_id: ShaderId,
        bindings: &[vk::DescriptorSetLayoutBinding],
    ) -> Result<PipelineId, GpuError> {
        let shader_module = self.shaders.shader_module(shader_id)?;

        let descriptor_set_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(bindings);
        let descriptor_set_layout = unsafe {
            self.device_context
                .device
                .create_descriptor_set_layout(&descriptor_set_layout_info, None)?
        };

        let layout_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&descriptor_set_layout));
        let layout = unsafe { self.device_context.device.create_pipeline_layout(&layout_info, None)? };

        let entry = CString::new(SHADER_ENTRY_POINT)?;
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
                .create_compute_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&compute_info), None)
                .map_err(|(_, err)| err)?[0]
        };

        let mut pipelines = self
            .pipelines
            .lock()
            .map_err(|_| std::io::Error::other("Compute pipelines lock is poisoned"))?;

        let id = PipelineId(pipelines.len() as u32);
        pipelines.insert(
            id,
            ComputePipeline {
                pipeline,
                layout,
                descriptor_set_layout,
            },
        );
        Ok(id)
    }

    pub fn dispatch_multiply(
        &self,
        pipeline_id: PipelineId,
        signal_buffer: vk::Buffer,
        ir_buffer: vk::Buffer,
        count: u32,
    ) -> Result<(), GpuError> {
        let (pipeline, layout, descriptor_set_layout) = {
            let pipelines = self
                .pipelines
                .lock()
                .map_err(|_| std::io::Error::other("Compute pipelines lock is poisoned"))?;
            let p = pipelines
                .get(&pipeline_id)
                .ok_or_else(|| std::io::Error::other(format!("Invalid pipeline id {}", pipeline_id.0)))?;
            (p.pipeline, p.layout, p.descriptor_set_layout)
        };

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(2)];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe {
            self.device_context
                .device
                .create_descriptor_pool(&descriptor_pool_info, None)?
        };

        let set_layouts = [descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);
        let descriptor_set = unsafe { self.device_context.device.allocate_descriptor_sets(&alloc_info)?[0] };

        let signal_info = [vk::DescriptorBufferInfo::default()
            .buffer(signal_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let ir_info = [vk::DescriptorBufferInfo::default()
            .buffer(ir_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&signal_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&ir_info),
        ];
        unsafe {
            self.device_context.device.update_descriptor_sets(&writes, &[]);
        }

        let dispatch_x = count.div_ceil(256);
        let push_data = count.to_ne_bytes();

        let result = self.submit_compute_and_wait(|command_buffer| {
            unsafe {
                self.device_context
                    .device
                    .cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
                self.device_context.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    layout,
                    0,
                    std::slice::from_ref(&descriptor_set),
                    &[],
                );
                self.device_context.device.cmd_push_constants(
                    command_buffer,
                    layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &push_data,
                );
                self.device_context
                    .device
                    .cmd_dispatch(command_buffer, dispatch_x, 1, 1);
            }
            Ok(())
        });

        unsafe {
            self.device_context
                .device
                .destroy_descriptor_pool(descriptor_pool, None);
        }

        result
    }

    pub unsafe fn queue_handle(&self) -> Result<vk::Queue, GpuError> {
        let queue = self
            .queue
            .read()
            .map_err(|_| std::io::Error::other("Compute queue lock is poisoned"))?;
        Ok(queue.handle)
    }

    pub unsafe fn command_pool_handle(&self) -> Result<vk::CommandPool, GpuError> {
        let queue = self
            .queue
            .read()
            .map_err(|_| std::io::Error::other("Compute queue lock is poisoned"))?;
        Ok(queue.command_pool)
    }
}

impl Drop for ComputeContext {
    fn drop(&mut self) {
        let queue = self
            .queue
            .write()
            .expect("Compute queue lock is poisoned during ComputeContext drop");

        if let Ok(mut in_flight) = self.in_flight.lock() {
            for (_, submission) in in_flight.drain() {
                unsafe {
                    let _ = self.device_context.device.wait_for_fences(
                        std::slice::from_ref(&submission.fence),
                        true,
                        u64::MAX,
                    );
                    self.device_context.device.destroy_fence(submission.fence, None);
                }
            }
        }

        if let Ok(mut available) = self.available_command_buffers.lock() {
            available.clear();
        }

        if let Ok(mut pipelines) = self.pipelines.lock() {
            unsafe {
                for (_, pipeline) in pipelines.drain() {
                    self.device_context.device.destroy_pipeline(pipeline.pipeline, None);
                    self.device_context
                        .device
                        .destroy_pipeline_layout(pipeline.layout, None);
                    self.device_context
                        .device
                        .destroy_descriptor_set_layout(pipeline.descriptor_set_layout, None);
                }
            }
        }

        unsafe {
            self.device_context.device.queue_wait_idle(queue.handle).ok();
            self.device_context
                .device
                .destroy_command_pool(queue.command_pool, None);
        }
    }
}
