//! Vulkan device selection, creation, and shared lifetime ownership.

use crate::core::error::{SoundError, SoundResult};
use ash::vk;
use ash::{Entry, Instance};
use std::ffi::CStr;
use std::sync::OnceLock;

// Use the same requirements for selection and logical-device creation.
const REQUIRED_DEVICE_EXTENSIONS: [&CStr; 5] = [
    vk::KHR_ACCELERATION_STRUCTURE_NAME,
    vk::KHR_RAY_TRACING_PIPELINE_NAME,
    vk::KHR_DEFERRED_HOST_OPERATIONS_NAME,
    vk::KHR_BUFFER_DEVICE_ADDRESS_NAME,
    vk::EXT_SHADER_ATOMIC_FLOAT_NAME,
];

/// Shared Vulkan handles used by GPU subsystems.
///
/// This is intentionally low-level: modules such as memory, compute, and RT
/// use it to create Vulkan objects while still centralizing device lifetime.
pub struct VkDeviceContext {
    handles: VkDeviceHandles,
    // Device contexts can outlive VkBackend. Keep its function pointers loaded
    // until the final device/instance destructor has run.
    _entry: Entry,
}

/// Read-only Vulkan handles and limits borrowed from an owning device context.
/// Raw ash operations still require Vulkan's lifetime and synchronization contracts.
pub struct VkDeviceHandles {
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
    /// Core limits used to reject invalid allocations, layouts and dispatches.
    pub limits: vk::PhysicalDeviceLimits,
}

impl std::ops::Deref for VkDeviceContext {
    type Target = VkDeviceHandles;

    fn deref(&self) -> &VkDeviceHandles {
        &self.handles
    }
}

/// Device properties needed by acceleration-structure and RT pipeline code.
pub struct RayTracingProperties {
    /// Acceleration-structure limits reported by Vulkan.
    pub acceleration_structure: vk::PhysicalDeviceAccelerationStructurePropertiesKHR<'static>,
    /// Shader group handle sizes and SBT alignment requirements.
    pub ray_tracing_pipeline: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'static>,
}

impl VkDeviceContext {
    /// Creates the shared device owner and returns its selected compute queue family.
    pub(super) fn new() -> SoundResult<(Self, u32)> {
        // Keep the loader resident between backend lifetimes. Unloading and
        // reloading it can stall later device creation on the Windows RT driver.
        static ENTRY: OnceLock<Entry> = OnceLock::new();
        let entry = match ENTRY.get() {
            Some(entry) => entry.clone(),
            None => {
                let loaded = unsafe { Entry::load()? };
                ENTRY.get_or_init(|| loaded).clone()
            }
        };
        let instance = Self::create_instance(&entry)?;

        // Pick a single compute-capable queue family that can also run RT work.
        let setup = (|| {
            let (physical_device, compute_family, rt_properties) =
                Self::select_physical_device_and_compute_family(&instance)?;
            let created = Self::create_device(&instance, physical_device, compute_family)?;
            Ok::<_, SoundError>((physical_device, compute_family, rt_properties, created))
        })();
        let (physical_device, compute_family, rt_properties, logical_device) = match setup {
            Ok(created) => created,
            Err(error) => {
                unsafe { instance.destroy_instance(None) };
                return Err(error);
            }
        };
        let acceleration_structure =
            ash::khr::acceleration_structure::Device::new(&instance, &logical_device);
        let ray_tracing_pipeline =
            ash::khr::ray_tracing_pipeline::Device::new(&instance, &logical_device);

        let limits = unsafe { instance.get_physical_device_properties(physical_device) }.limits;
        let device = Self {
            handles: VkDeviceHandles {
                instance,
                physical_device,
                device: logical_device,
                acceleration_structure,
                ray_tracing_pipeline,
                rt_properties,
                limits,
            },
            _entry: entry,
        };
        Ok((device, compute_family))
    }

    /// Creates an unsignaled fence on the backend device.
    pub fn create_fence(&self) -> SoundResult<vk::Fence> {
        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe { self.device.create_fence(&fence_info, None)? };
        Ok(fence)
    }

    fn create_instance(entry: &Entry) -> SoundResult<Instance> {
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"sound-engine")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"sound-engine")
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);

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
                let compute_family =
                    Self::find_queue_family(instance, physical_device, vk::QueueFlags::COMPUTE)?;
                let rt_properties =
                    Self::query_ray_tracing_support(instance, physical_device).ok()?;
                let score = Self::score_physical_device(instance, physical_device);
                Some((score, physical_device, compute_family, rt_properties))
            })
            .max_by_key(|(score, _, _, _)| *score)
            .map(|(_, physical_device, compute_family, rt_properties)| {
                (physical_device, compute_family, rt_properties)
            })
            .ok_or_else(|| {
                SoundError::backend_unavailable(
                    "No Vulkan physical device with compute and KHR ray tracing support was found",
                )
            })
    }

    fn create_device(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        compute_family: u32,
    ) -> SoundResult<ash::Device> {
        let queue_priorities = [1.0_f32];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(compute_family)
            .queue_priorities(&queue_priorities)];

        // Enable the full set needed for device-address AS builds and RT traces.
        let extension_names = REQUIRED_DEVICE_EXTENSIONS.map(|name| name.as_ptr());

        // Features are chained through pNext; Vulkan only enables what is set here.
        let mut buffer_device_address_features =
            vk::PhysicalDeviceBufferDeviceAddressFeatures::default().buffer_device_address(true);
        let mut acceleration_structure_features =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
                .acceleration_structure(true);
        let mut ray_tracing_pipeline_features =
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default().ray_tracing_pipeline(true);
        let mut shader_atomic_float_features =
            vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default()
                .shader_buffer_float32_atomic_add(true);
        // Surface hit shaders dereference 64-bit buffer device addresses. The
        // corresponding core feature must be enabled in addition to BDA.
        let core_features = vk::PhysicalDeviceFeatures::default().shader_int64(true);

        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&extension_names)
            .enabled_features(&core_features)
            .push_next(&mut buffer_device_address_features)
            .push_next(&mut acceleration_structure_features)
            .push_next(&mut ray_tracing_pipeline_features)
            .push_next(&mut shader_atomic_float_features);
        let device = unsafe { instance.create_device(physical_device, &device_info, None)? };

        Ok(device)
    }

    fn find_queue_family(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        flags: vk::QueueFlags,
    ) -> Option<u32> {
        let families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
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
        let device_local_memory_bytes = mem_props.memory_heaps
            [..mem_props.memory_heap_count as usize]
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
        if unsafe { instance.get_physical_device_properties(physical_device) }.api_version
            < vk::API_VERSION_1_3
        {
            return Err(SoundError::unsupported_feature(
                "Vulkan 1.3 is required by the bundled shaders and allocator",
            ));
        }
        Self::require_device_extensions(instance, physical_device)?;

        let mut buffer_device_address_features =
            vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
        let mut acceleration_structure_features =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
        let mut ray_tracing_pipeline_features =
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default();
        let mut shader_atomic_float_features =
            vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default();
        // Query the boolean feature chain separately from property limits.
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut buffer_device_address_features)
            .push_next(&mut acceleration_structure_features)
            .push_next(&mut ray_tracing_pipeline_features)
            .push_next(&mut shader_atomic_float_features);

        unsafe {
            instance.get_physical_device_features2(physical_device, &mut features);
        }
        let shader_int64_supported = features.features.shader_int64 == vk::TRUE;

        if buffer_device_address_features.buffer_device_address != vk::TRUE
            || acceleration_structure_features.acceleration_structure != vk::TRUE
            || ray_tracing_pipeline_features.ray_tracing_pipeline != vk::TRUE
            || shader_atomic_float_features.shader_buffer_float32_atomic_add != vk::TRUE
            || !shader_int64_supported
        {
            return Err(SoundError::unsupported_feature(
                "Required Vulkan ray tracing, shaderInt64, or shader float atomic features are not supported",
            ));
        }

        let mut acceleration_structure_properties =
            vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
        let mut ray_tracing_pipeline_properties =
            vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
        // SBT construction later depends on these alignment properties.
        let mut properties = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut acceleration_structure_properties)
            .push_next(&mut ray_tracing_pipeline_properties);

        unsafe {
            instance.get_physical_device_properties2(physical_device, &mut properties);
        }

        // The queried pNext chain points at stack locals and must not escape.
        acceleration_structure_properties.p_next = std::ptr::null_mut();
        ray_tracing_pipeline_properties.p_next = std::ptr::null_mut();
        Ok(RayTracingProperties {
            acceleration_structure: acceleration_structure_properties,
            ray_tracing_pipeline: ray_tracing_pipeline_properties,
        })
    }

    fn require_device_extensions(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
    ) -> SoundResult<()> {
        let extensions =
            unsafe { instance.enumerate_device_extension_properties(physical_device)? };
        for required_extension in REQUIRED_DEVICE_EXTENSIONS {
            let found = extensions.iter().any(|extension| {
                let name = unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) };
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

impl Drop for VkDeviceContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
