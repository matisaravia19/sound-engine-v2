use crate::gpu::buffer::BufferContext;
use crate::gpu::errors::GpuError;
use crate::gpu::shader::ShaderModule;
use ash::vk::{self, CommandPool, PhysicalDevice, Queue};
use ash::{Device, Entry, Instance};
use std::ffi::CString;

pub struct GpuContext {
    entry: Entry,
    instance: Instance,
    physical_device: PhysicalDevice,
    device: Device,
    queues: QueueContext,
    buffers: Option<BufferContext>,
    shaders: Vec<ShaderModule>,
}

struct QueueContext {
    compute_family_index: u32,
    compute_queue: Queue,
    compute_command_pool: CommandPool,
}

impl GpuContext {
    pub fn init() -> Result<Self, GpuError> {
        let entry = unsafe { Entry::load()? };
        let instance = Self::create_instance(&entry)?;
        let (physical_device, compute_family_index) = Self::select_physical_device_and_queue_family(&instance)?;
        let (device, queues) = Self::create_device_and_queues(&instance, physical_device, compute_family_index)?;
        let buffers = BufferContext::init(&instance, physical_device, &device)?;

        Ok(Self {
            entry,
            instance,
            physical_device,
            device,
            queues,
            buffers: Some(buffers),
            shaders: Vec::new(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn physical_device(&self) -> &PhysicalDevice {
        &self.physical_device
    }

    pub fn compute_queue(&self) -> &Queue {
        &self.queues.compute_queue
    }

    pub fn compute_command_pool(&self) -> &CommandPool {
        &self.queues.compute_command_pool
    }

    pub fn create_fence(&self) -> Result<vk::Fence, GpuError> {
        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe { self.device.create_fence(&fence_info, None)? };
        Ok(fence)
    }

    pub fn reset_fence(&self, fence: vk::Fence) -> Result<(), GpuError> {
        unsafe {
            self.device.reset_fences(std::slice::from_ref(&fence))?;
        }
        Ok(())
    }

    pub fn wait_for_fence(&self, fence: vk::Fence) -> Result<(), GpuError> {
        unsafe {
            self.device
                .wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)?;
        }
        Ok(())
    }

    pub fn destroy_fence(&self, fence: vk::Fence) {
        unsafe {
            self.device.destroy_fence(fence, None);
        }
    }

    pub fn immediate_submit<F>(&self, record: F) -> Result<(), GpuError>
    where
        F: FnOnce(vk::CommandBuffer) -> Result<(), GpuError>,
    {
        let allocation_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.queues.compute_command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer = unsafe { self.device.allocate_command_buffers(&allocation_info)? }[0];

        let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe { self.device.create_fence(&fence_info, None)? };

        let result = (|| -> Result<(), GpuError> {
            unsafe {
                self.device.begin_command_buffer(command_buffer, &begin_info)?;
            }

            record(command_buffer)?;

            unsafe {
                self.device.end_command_buffer(command_buffer)?;
            }

            let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));

            unsafe {
                self.device
                    .queue_submit(self.queues.compute_queue, std::slice::from_ref(&submit_info), fence)?;
                self.device
                    .wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)?;
            }

            Ok(())
        })();

        unsafe {
            self.device.destroy_fence(fence, None);
            self.device
                .free_command_buffers(self.queues.compute_command_pool, std::slice::from_ref(&command_buffer));
        }

        result
    }

    pub(super) fn buffers_ref(&self) -> Result<&BufferContext, GpuError> {
        self.buffers
            .as_ref()
            .ok_or_else(|| std::io::Error::other("Buffer context is not available").into())
    }

    pub(super) fn buffers_mut(&mut self) -> Result<&mut BufferContext, GpuError> {
        self.buffers
            .as_mut()
            .ok_or_else(|| std::io::Error::other("Buffer context is not available").into())
    }

    pub(super) fn shaders_ref(&self) -> &Vec<ShaderModule> {
        &self.shaders
    }

    pub(super) fn shaders_mut(&mut self) -> &mut Vec<ShaderModule> {
        &mut self.shaders
    }

    fn create_instance(entry: &Entry) -> Result<Instance, GpuError> {
        let app_name = CString::new("sound-engine")?;
        let engine_name = CString::new("sound-engine")?;

        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::make_api_version(0, 1, 3, 0));

        let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&instance_info, None)? };

        Ok(instance)
    }

    fn select_physical_device_and_queue_family(instance: &Instance) -> Result<(PhysicalDevice, u32), GpuError> {
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };

        physical_devices
            .into_iter()
            .filter_map(|physical_device| {
                let compute_family_index = Self::get_compute_queue_family_index(instance, physical_device)?;
                let score = Self::score_physical_device(instance, physical_device);
                Some((score, physical_device, compute_family_index))
            })
            .max_by_key(|(score, _, _)| *score)
            .map(|(_, physical_device, compute_family_index)| (physical_device, compute_family_index))
            .ok_or_else(|| {
                "No Vulkan physical device with a compute queue family was found"
                    .to_string()
                    .into()
            })
    }

    fn get_compute_queue_family_index(instance: &Instance, physical_device: PhysicalDevice) -> Option<u32> {
        let queue_families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        queue_families
            .iter()
            .enumerate()
            .find(|(_, family)| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(index, _)| index as u32)
    }

    fn score_physical_device(instance: &Instance, physical_device: PhysicalDevice) -> u64 {
        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let device_type_score = match properties.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 5_000_000_u64,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 4_000_000_u64,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 3_000_000_u64,
            vk::PhysicalDeviceType::CPU => 2_000_000_u64,
            _ => 1_000_000_u64,
        };

        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let device_local_memory_bytes = memory_properties.memory_heaps[..memory_properties.memory_heap_count as usize]
            .iter()
            .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|heap| heap.size)
            .sum::<u64>();

        let limits = properties.limits;

        device_type_score
            + (device_local_memory_bytes / (1024 * 1024))
            + limits.max_compute_shared_memory_size as u64
            + (limits.max_compute_work_group_invocations as u64 * 1024)
    }

    fn create_device_and_queues(
        instance: &Instance,
        physical_device: PhysicalDevice,
        compute_family_index: u32,
    ) -> Result<(Device, QueueContext), GpuError> {
        let queue_priorities = [1.0_f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(compute_family_index)
            .queue_priorities(&queue_priorities);

        let queue_create_infos = [queue_create_info];
        let device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_create_infos);
        let device = unsafe { instance.create_device(physical_device, &device_info, None)? };

        let compute_queue = unsafe { device.get_device_queue(compute_family_index, 0) };

        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(compute_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let compute_command_pool = unsafe { device.create_command_pool(&command_pool_info, None)? };

        let queue_context = QueueContext {
            compute_family_index,
            compute_queue,
            compute_command_pool,
        };

        Ok((device, queue_context))
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        for shader in self.shaders.drain(..) {
            unsafe {
                self.device.destroy_shader_module(shader.handle(), None);
            }
        }

        if let Some(buffers) = self.buffers.take() {
            drop(buffers);
        }

        unsafe {
            self.device.destroy_command_pool(self.queues.compute_command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
