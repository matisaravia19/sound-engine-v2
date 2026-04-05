use crate::gpu::GpuError;
use crate::gpu::memory::GpuAllocator;
use crate::gpu::shader::ShaderLibrary;
use crate::gpu::sync::SyncContext;
use ash::vk;
use ash::{Device, Entry, Instance};
use std::ffi::CString;
use std::sync::Arc;

pub(crate) struct VkBackend {
    entry: Entry,
    memory: GpuAllocator,
    shaders: ShaderLibrary,
    sync: SyncContext,
    device: Arc<VkDeviceContext>,
}

pub(super) struct VkDeviceContext {
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queues: QueueSet,
}

pub(super) struct QueueSet {
    pub compute_queue: vk::Queue,
    pub compute_family: u32,
    pub compute_command_pool: vk::CommandPool,
    pub transfer_queue: vk::Queue,
    pub transfer_family: u32,
    pub transfer_command_pool: vk::CommandPool,
}

impl VkBackend {
    pub fn new() -> Result<Self, GpuError> {
        let entry = unsafe { Entry::load()? };
        let instance = Self::create_instance(&entry)?;

        let (physical_device, compute_family) = Self::select_physical_device_and_compute_family(&instance)?;
        let (device, queues) = Self::create_device_and_queues(&instance, physical_device, compute_family)?;

        let device = Arc::new(VkDeviceContext {
            instance,
            physical_device,
            device,
            queues,
        });
        let memory = GpuAllocator::new(device.clone())?;

        Ok(Self {
            entry,
            memory,
            shaders: ShaderLibrary::new(device.clone()),
            sync: SyncContext::new(device.clone()),
            device,
        })
    }

    pub(super) fn device(&self) -> &VkDeviceContext {
        self.device.as_ref()
    }

    pub fn memory(&self) -> &GpuAllocator {
        &self.memory
    }

    pub fn memory_mut(&mut self) -> &mut GpuAllocator {
        &mut self.memory
    }

    pub fn shaders(&self) -> &ShaderLibrary {
        &self.shaders
    }

    pub fn shaders_mut(&mut self) -> &mut ShaderLibrary {
        &mut self.shaders
    }

    pub fn sync(&self) -> &SyncContext {
        &self.sync
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

    fn select_physical_device_and_compute_family(instance: &Instance) -> Result<(vk::PhysicalDevice, u32), GpuError> {
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };

        physical_devices
            .into_iter()
            .filter_map(|physical_device| {
                let compute_family = Self::find_queue_family(instance, physical_device, vk::QueueFlags::COMPUTE)?;
                let score = Self::score_physical_device(instance, physical_device);
                Some((score, physical_device, compute_family))
            })
            .max_by_key(|(score, _, _)| *score)
            .map(|(_, physical_device, compute_family)| (physical_device, compute_family))
            .ok_or_else(|| std::io::Error::other("No Vulkan physical device with compute support was found").into())
    }

    fn create_device_and_queues(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        compute_family: u32,
    ) -> Result<(Device, QueueSet), GpuError> {
        let transfer_family =
            Self::find_queue_family(instance, physical_device, vk::QueueFlags::TRANSFER).unwrap_or(compute_family);

        let queue_priorities = [1.0_f32];
        let mut queue_infos = vec![
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(compute_family)
                .queue_priorities(&queue_priorities),
        ];

        if transfer_family != compute_family {
            queue_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(transfer_family)
                    .queue_priorities(&queue_priorities),
            );
        }

        let device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_infos);
        let device = unsafe { instance.create_device(physical_device, &device_info, None)? };

        let compute_queue = unsafe { device.get_device_queue(compute_family, 0) };
        let transfer_queue = unsafe { device.get_device_queue(transfer_family, 0) };

        let compute_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(compute_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let compute_command_pool = unsafe { device.create_command_pool(&compute_pool_info, None)? };

        let transfer_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(transfer_family)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let transfer_command_pool = unsafe { device.create_command_pool(&transfer_pool_info, None)? };

        let queues = QueueSet {
            compute_queue,
            compute_family,
            compute_command_pool,
            transfer_queue,
            transfer_family,
            transfer_command_pool,
        };

        Ok((device, queues))
    }

    fn find_queue_family(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        flags: vk::QueueFlags,
    ) -> Option<u32> {
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        families
            .iter()
            .enumerate()
            .find(|(_, family)| family.queue_flags.contains(flags))
            .map(|(idx, _)| idx as u32)
    }

    fn score_physical_device(instance: &Instance, physical_device: vk::PhysicalDevice) -> u64 {
        let properties = unsafe { instance.get_physical_device_properties(physical_device) };

        let type_score = match properties.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 5_000_000_u64,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 4_000_000_u64,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 3_000_000_u64,
            vk::PhysicalDeviceType::CPU => 2_000_000_u64,
            _ => 1_000_000_u64,
        };

        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let device_local_memory_bytes = mem_props.memory_heaps[..mem_props.memory_heap_count as usize]
            .iter()
            .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|heap| heap.size)
            .sum::<u64>();

        type_score + (device_local_memory_bytes / (1024 * 1024))
    }
}

impl Drop for VkBackend {
    fn drop(&mut self) {
        let _ = &self.entry;
    }
}

impl Drop for VkDeviceContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device
                .destroy_command_pool(self.queues.transfer_command_pool, None);
            self.device.destroy_command_pool(self.queues.compute_command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
