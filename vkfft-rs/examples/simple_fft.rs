use ash::{Entry, vk};
use std::{error::Error, ffi::CString};
use vkfft_rs::plan::{DeviceHandles, PlanBuilder};

const N: usize = 8;

fn main() -> Result<(), Box<dyn Error>> {
    run()?;
    Ok(())
}

fn run() -> Result<(), Box<dyn Error>> {
    let entry = unsafe { Entry::load()? };

    let app_name = CString::new("vkfft-rs-simple-fft")?;
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(0)
        .engine_name(&app_name)
        .engine_version(0)
        .api_version(vk::API_VERSION_1_1);

    let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = unsafe { entry.create_instance(&instance_info, None)? };

    let physical_devices = unsafe { instance.enumerate_physical_devices()? };
    let physical_device = *physical_devices.first().ok_or("No Vulkan physical device found")?;

    let queue_family_props = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let compute_queue_family_index = queue_family_props
        .iter()
        .enumerate()
        .find_map(|(index, props)| {
            if props.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                Some(index as u32)
            } else {
                None
            }
        })
        .ok_or("No compute-capable queue family found")?;

    let queue_priority = [1.0_f32];
    let queue_info = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(compute_queue_family_index)
        .queue_priorities(&queue_priority)];

    let device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_info);
    let device = unsafe { instance.create_device(physical_device, &device_info, None)? };
    let queue = unsafe { device.get_device_queue(compute_queue_family_index, 0) };

    let command_pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(compute_queue_family_index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let command_pool = unsafe { device.create_command_pool(&command_pool_info, None)? };

    let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };

    let element_count = N * 2;
    let buffer_bytes = (element_count * std::mem::size_of::<f32>()) as u64;

    let buffer_info = vk::BufferCreateInfo::default()
        .size(buffer_bytes)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { device.create_buffer(&buffer_info, None)? };

    let req = unsafe { device.get_buffer_memory_requirements(buffer) };
    let mem_index = find_memory_type_index(
        &instance,
        physical_device,
        req.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .ok_or("No suitable HOST_VISIBLE|HOST_COHERENT memory type found")?;

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(mem_index);
    let memory = unsafe { device.allocate_memory(&alloc_info, None)? };
    unsafe { device.bind_buffer_memory(buffer, memory, 0)? };

    {
        let mapped = unsafe { device.map_memory(memory, 0, buffer_bytes, vk::MemoryMapFlags::empty())? };
        let slice = unsafe { std::slice::from_raw_parts_mut(mapped.cast::<f32>(), element_count) };
        slice.fill(0.0);
        slice[0] = 1.0;
        unsafe { device.unmap_memory(memory) };
    }

    let mut builder = PlanBuilder::new(DeviceHandles {
        physical_device,
        device: device.handle(),
        queue,
        command_pool,
        fence,
    })
    .map_err(|e| format!("PlanBuilder::new failed: {e:?}"))?;

    builder.with_dimensions(&[N as u64]).with_buffer(buffer, buffer_bytes);

    let plan = builder
        .build()
        .map_err(|e| format!("PlanBuilder::build failed: {e:?}"))?;

    let cmd_alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let command_buffer = unsafe { device.allocate_command_buffers(&cmd_alloc_info)?[0] };

    let begin_info = vk::CommandBufferBeginInfo::default();
    unsafe { device.begin_command_buffer(command_buffer, &begin_info)? };
    plan.launch(command_buffer)
        .map_err(|e| format!("Plan::launch failed: {e:?}"))?;
    unsafe { device.end_command_buffer(command_buffer)? };

    let submit_infos = [vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer))];
    unsafe { device.reset_fences(std::slice::from_ref(&fence))? };
    unsafe { device.queue_submit(queue, &submit_infos, fence)? };
    unsafe { device.wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)? };

    {
        let mapped = unsafe { device.map_memory(memory, 0, buffer_bytes, vk::MemoryMapFlags::empty())? };
        let slice = unsafe { std::slice::from_raw_parts(mapped.cast::<f32>(), element_count) };

        let mut ok = true;
        for k in 0..N {
            let re = slice[2 * k];
            let im = slice[2 * k + 1];
            if (re - 1.0).abs() > 1e-2 || im.abs() > 1e-2 {
                ok = false;
                break;
            }
        }

        unsafe { device.unmap_memory(memory) };

        if !ok {
            return Err("FFT output check failed for impulse input".into());
        }
    }

    unsafe { device.device_wait_idle()? };

    drop(plan);
    drop(builder);

    unsafe {
        device.free_command_buffers(command_pool, std::slice::from_ref(&command_buffer));
        device.destroy_buffer(buffer, None);
        device.free_memory(memory, None);
        device.destroy_fence(fence, None);
        device.destroy_command_pool(command_pool, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
    }

    println!("VkFFT simple FFT example passed.");
    Ok(())
}

fn find_memory_type_index(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };

    for index in 0..mem_props.memory_type_count {
        let supported = (type_bits & (1 << index)) != 0;
        let has_flags = mem_props.memory_types[index as usize].property_flags.contains(required);
        if supported && has_flags {
            return Some(index);
        }
    }

    None
}
