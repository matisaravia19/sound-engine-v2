use crate::error::{SoundError, SoundResult};
use crate::gpu::compute::ComputeContext;
use crate::gpu::memory::GpuAllocator;
use crate::gpu::rt::RtContext;
use crate::gpu::shader::ShaderLibrary;
use ash::vk;
use ash::{Entry, Instance};
use std::ffi::CString;
use std::sync::Arc;

/// Owns the Vulkan device and the GPU subsystems built on top of it.
///
/// The backend selects a ray-tracing-capable device at construction time and
/// tears down child contexts before the device is destroyed.
pub struct VkBackend {
    entry: Entry,
    memory: GpuAllocator,
    shaders: Arc<ShaderLibrary>,
    compute: Arc<ComputeContext>,
    rt: RtContext,
    device: Arc<VkDeviceContext>,
}

/// Shared Vulkan handles used by GPU subsystems.
///
/// This is intentionally low-level: modules such as memory, compute, and RT
/// use it to create Vulkan objects while still centralizing device lifetime.
pub struct VkDeviceContext {
    /// Vulkan instance that owns the selected physical device.
    pub instance: ash::Instance,
    /// Physical device chosen for compute and KHR ray tracing support.
    pub physical_device: vk::PhysicalDevice,
    /// Logical Vulkan device used by all engine GPU work.
    pub device: ash::Device,
    /// KHR acceleration-structure function loader for this device.
    pub acceleration_structure: ash::khr::acceleration_structure::Device,
    /// KHR ray-tracing-pipeline function loader for this device.
    pub ray_tracing_pipeline: ash::khr::ray_tracing_pipeline::Device,
    /// Ray tracing limits and alignment requirements queried from the device.
    pub rt_properties: RayTracingProperties,
}

/// Device properties needed by acceleration-structure and RT pipeline code.
pub struct RayTracingProperties {
    /// Acceleration-structure limits reported by Vulkan.
    pub acceleration_structure: vk::PhysicalDeviceAccelerationStructurePropertiesKHR<'static>,
    /// Shader group handle sizes and SBT alignment requirements.
    pub ray_tracing_pipeline: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'static>,
}

/// Queue handle and command pool pair used by one GPU subsystem.
pub(super) struct QueueSet {
    /// Vulkan queue used for submissions.
    pub handle: vk::Queue,
    /// Queue family index used to create the queue and command pool.
    pub queue_family: u32,
    /// Command pool tied to `queue_family`.
    pub command_pool: vk::CommandPool,
}

struct DeviceCreationResult {
    device: ash::Device,
    compute_queue: QueueSet,
    transfer_queue: QueueSet,
}

impl VkBackend {
    /// Creates a Vulkan backend with compute, transfer, shader, and RT support.
    ///
    /// Device selection requires KHR acceleration structures, ray tracing
    /// pipelines, deferred host operations, and buffer device addresses.
    pub fn new() -> SoundResult<Self> {
        let entry = unsafe { Entry::load()? };
        let instance = Self::create_instance(&entry)?;

        // Pick a single compute-capable queue family that can also run RT work.
        let (physical_device, compute_family, rt_properties) =
            Self::select_physical_device_and_compute_family(&instance)?;
        let device_creation_result = Self::create_device_and_queues(&instance, physical_device, compute_family)?;
        let acceleration_structure =
            ash::khr::acceleration_structure::Device::new(&instance, &device_creation_result.device);
        let ray_tracing_pipeline =
            ash::khr::ray_tracing_pipeline::Device::new(&instance, &device_creation_result.device);

        let device = Arc::new(VkDeviceContext {
            instance,
            physical_device,
            device: device_creation_result.device,
            acceleration_structure,
            ray_tracing_pipeline,
            rt_properties,
        });

        let shaders = Arc::new(ShaderLibrary::new(device.clone()));

        let compute = Arc::new(ComputeContext::new(
            device.clone(),
            shaders.clone(),
            device_creation_result.compute_queue,
        )?);

        Ok(Self {
            entry,
            memory: GpuAllocator::new(device.clone(), device_creation_result.transfer_queue)?,
            shaders: shaders.clone(),
            compute: compute.clone(),
            rt: RtContext::new(device.clone(), shaders.clone(), compute),
            device,
        })
    }

    /// Returns the shared Vulkan device context.
    pub fn device(&self) -> &VkDeviceContext {
        self.device.as_ref()
    }

    /// Clones the shared device context for long-lived GPU resources.
    pub fn device_context(&self) -> Arc<VkDeviceContext> {
        self.device.clone()
    }

    /// Returns the GPU allocator and synchronous transfer helper.
    pub fn memory(&self) -> &GpuAllocator {
        &self.memory
    }

    /// Returns a mutable reference to the GPU allocator.
    pub fn memory_mut(&mut self) -> &mut GpuAllocator {
        &mut self.memory
    }

    /// Returns the shader library used to compile and own shader modules.
    pub fn shaders(&self) -> &ShaderLibrary {
        &self.shaders
    }

    /// Returns the compute context used for pipeline dispatch and submission.
    pub fn compute(&self) -> &ComputeContext {
        self.compute.as_ref()
    }

    /// Returns the ray tracing context used for AS builds and RT pipelines.
    pub fn rt(&self) -> &RtContext {
        &self.rt
    }

    fn create_instance(entry: &Entry) -> SoundResult<Instance> {
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

    fn select_physical_device_and_compute_family(
        instance: &Instance,
    ) -> SoundResult<(vk::PhysicalDevice, u32, RayTracingProperties)> {
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };

        physical_devices
            .into_iter()
            .filter_map(|physical_device| {
                let compute_family = Self::find_queue_family(instance, physical_device, vk::QueueFlags::COMPUTE)?;
                let rt_properties = Self::query_ray_tracing_support(instance, physical_device).ok()?;
                let score = Self::score_physical_device(instance, physical_device);
                Some((score, physical_device, compute_family, rt_properties))
            })
            .max_by_key(|(score, _, _, _)| *score)
            .map(|(_, physical_device, compute_family, rt_properties)| (physical_device, compute_family, rt_properties))
            .ok_or_else(|| {
                SoundError::backend_unavailable(
                    "No Vulkan physical device with compute and KHR ray tracing support was found",
                )
            })
    }

    fn create_device_and_queues(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        compute_family: u32,
    ) -> SoundResult<DeviceCreationResult> {
        let queue_priorities = [1.0_f32];
        let mut queue_infos = vec![
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(compute_family)
                .queue_priorities(&queue_priorities),
        ];

        let transfer_family =
            Self::find_queue_family(instance, physical_device, vk::QueueFlags::TRANSFER).unwrap_or(compute_family);
        if transfer_family != compute_family {
            queue_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(transfer_family)
                    .queue_priorities(&queue_priorities),
            );
        }

        // Enable the full set needed for device-address AS builds and RT traces.
        let extension_names = [
            vk::KHR_ACCELERATION_STRUCTURE_NAME.as_ptr(),
            vk::KHR_RAY_TRACING_PIPELINE_NAME.as_ptr(),
            vk::KHR_DEFERRED_HOST_OPERATIONS_NAME.as_ptr(),
            vk::KHR_BUFFER_DEVICE_ADDRESS_NAME.as_ptr(),
        ];

        // Features are chained through pNext; Vulkan only enables what is set here.
        let mut buffer_device_address_features =
            vk::PhysicalDeviceBufferDeviceAddressFeatures::default().buffer_device_address(true);
        let mut acceleration_structure_features =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default().acceleration_structure(true);
        let mut ray_tracing_pipeline_features =
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default().ray_tracing_pipeline(true);

        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&extension_names)
            .push_next(&mut buffer_device_address_features)
            .push_next(&mut acceleration_structure_features)
            .push_next(&mut ray_tracing_pipeline_features);
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

        Ok(DeviceCreationResult {
            device,
            compute_queue: QueueSet {
                handle: compute_queue,
                queue_family: compute_family,
                command_pool: compute_command_pool,
            },
            transfer_queue: QueueSet {
                handle: transfer_queue,
                queue_family: transfer_family,
                command_pool: transfer_command_pool,
            },
        })
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

    fn query_ray_tracing_support(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
    ) -> SoundResult<RayTracingProperties> {
        Self::require_device_extensions(instance, physical_device)?;

        let mut buffer_device_address_features = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
        let mut acceleration_structure_features = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
        let mut ray_tracing_pipeline_features = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default();
        // Query the boolean feature chain separately from property limits.
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut buffer_device_address_features)
            .push_next(&mut acceleration_structure_features)
            .push_next(&mut ray_tracing_pipeline_features);

        unsafe {
            instance.get_physical_device_features2(physical_device, &mut features);
        }

        if buffer_device_address_features.buffer_device_address != vk::TRUE
            || acceleration_structure_features.acceleration_structure != vk::TRUE
            || ray_tracing_pipeline_features.ray_tracing_pipeline != vk::TRUE
        {
            return Err(SoundError::unsupported_feature(
                "Required Vulkan ray tracing features are not supported",
            ));
        }

        let mut acceleration_structure_properties = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
        let mut ray_tracing_pipeline_properties = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
        // SBT construction later depends on these alignment properties.
        let mut properties = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut acceleration_structure_properties)
            .push_next(&mut ray_tracing_pipeline_properties);

        unsafe {
            instance.get_physical_device_properties2(physical_device, &mut properties);
        }

        Ok(RayTracingProperties {
            acceleration_structure: acceleration_structure_properties,
            ray_tracing_pipeline: ray_tracing_pipeline_properties,
        })
    }

    fn require_device_extensions(instance: &Instance, physical_device: vk::PhysicalDevice) -> SoundResult<()> {
        let extensions = unsafe { instance.enumerate_device_extension_properties(physical_device)? };
        let required = [
            vk::KHR_ACCELERATION_STRUCTURE_NAME,
            vk::KHR_RAY_TRACING_PIPELINE_NAME,
            vk::KHR_DEFERRED_HOST_OPERATIONS_NAME,
            vk::KHR_BUFFER_DEVICE_ADDRESS_NAME,
        ];

        for required_extension in required {
            let found = extensions.iter().any(|extension| {
                let name = unsafe { std::ffi::CStr::from_ptr(extension.extension_name.as_ptr()) };
                name == required_extension
            });

            if !found {
                return Err(SoundError::unsupported_feature(format!(
                    "Required Vulkan device extension is not supported: {}",
                    required_extension.to_string_lossy()
                )));
            }
        }

        Ok(())
    }
}

impl VkDeviceContext {
    /// Creates an unsignaled fence on the backend device.
    pub fn create_fence(&self) -> SoundResult<vk::Fence> {
        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe { self.device.create_fence(&fence_info, None)? };
        Ok(fence)
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
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
