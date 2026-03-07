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

    let app_name = CString::new("vkfft-rs-convolution")?;
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(0)
        .engine_name(&app_name)
        .engine_version(0)
        .api_version(vk::API_VERSION_1_1);

    let instance_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = unsafe { entry.create_instance(&instance_info, None)? };

    let physical_devices = unsafe { instance.enumerate_physical_devices()? };
    let physical_device = *physical_devices
        .first()
        .ok_or("No Vulkan physical device found")?;

    let queue_family_props = unsafe {
        instance.get_physical_device_queue_family_properties(physical_device)
    };
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

    let signal_buffer = unsafe { device.create_buffer(&buffer_info, None)? };
    let kernel_buffer = unsafe { device.create_buffer(&buffer_info, None)? };

    let req = unsafe { device.get_buffer_memory_requirements(signal_buffer) };
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

    let signal_memory = unsafe { device.allocate_memory(&alloc_info, None)? };
    let kernel_memory = unsafe { device.allocate_memory(&alloc_info, None)? };

    unsafe { device.bind_buffer_memory(signal_buffer, signal_memory, 0)? };
    unsafe { device.bind_buffer_memory(kernel_buffer, kernel_memory, 0)? };

    let signal_real = [1.0_f32, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0];
    let kernel_real = [1.0_f32, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let expected = circular_convolution(&signal_real, &kernel_real);

    {
        let mapped = unsafe {
            device.map_memory(signal_memory, 0, buffer_bytes, vk::MemoryMapFlags::empty())?
        };
        let slice = unsafe { std::slice::from_raw_parts_mut(mapped.cast::<f32>(), element_count) };
        slice.fill(0.0);
        for i in 0..N {
            slice[2 * i] = signal_real[i];
        }
        unsafe { device.unmap_memory(signal_memory) };
    }

    {
        let mapped = unsafe {
            device.map_memory(kernel_memory, 0, buffer_bytes, vk::MemoryMapFlags::empty())?
        };
        let slice = unsafe { std::slice::from_raw_parts_mut(mapped.cast::<f32>(), element_count) };
        slice.fill(0.0);
        for i in 0..N {
            slice[2 * i] = kernel_real[i];
        }
        unsafe { device.unmap_memory(kernel_memory) };
    }

    let device_handles = DeviceHandles {
        physical_device,
        device: device.handle(),
        queue,
        command_pool,
        fence,
    };

    let mut kernel_builder = PlanBuilder::new(DeviceHandles { ..device_handles })
        .map_err(|e| format!("kernel PlanBuilder::new failed: {e:?}"))?;
    kernel_builder
        .with_dimensions(&[N as u64])
        .with_buffer(kernel_buffer, buffer_bytes);

    let kernel_plan = kernel_builder
        .build()
        .map_err(|e| format!("kernel PlanBuilder::build failed: {e:?}"))?;

    let mut signal_builder = PlanBuilder::new(DeviceHandles { ..device_handles })
        .map_err(|e| format!("signal PlanBuilder::new failed: {e:?}"))?;
    signal_builder
        .with_dimensions(&[N as u64])
        .with_buffer(signal_buffer, buffer_bytes)
        .with_normalization();

    let signal_plan = signal_builder
        .build()
        .map_err(|e| format!("signal PlanBuilder::build failed: {e:?}"))?;

    let cmd_alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let command_buffer = unsafe { device.allocate_command_buffers(&cmd_alloc_info)?[0] };

    let begin_info = vk::CommandBufferBeginInfo::default();
    unsafe { device.begin_command_buffer(command_buffer, &begin_info)? };

    signal_plan
        .launch(command_buffer)
        .map_err(|e| format!("signal forward launch failed: {e:?}"))?;
    kernel_plan
        .launch(command_buffer)
        .map_err(|e| format!("kernel forward launch failed: {e:?}"))?;

    unsafe { device.end_command_buffer(command_buffer)? };

    let submit_infos = [vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer))];
    unsafe { device.reset_fences(std::slice::from_ref(&fence))? };
    unsafe { device.queue_submit(queue, &submit_infos, fence)? };
    unsafe { device.wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)? };

    {
        let mapped_signal = unsafe {
            device.map_memory(signal_memory, 0, buffer_bytes, vk::MemoryMapFlags::empty())?
        };
        let mapped_kernel = unsafe {
            device.map_memory(kernel_memory, 0, buffer_bytes, vk::MemoryMapFlags::empty())?
        };

        let signal_slice = unsafe { std::slice::from_raw_parts_mut(mapped_signal.cast::<f32>(), element_count) };
        let kernel_slice = unsafe { std::slice::from_raw_parts(mapped_kernel.cast::<f32>(), element_count) };

        for i in 0..N {
            let ar = signal_slice[2 * i];
            let ai = signal_slice[2 * i + 1];
            let br = kernel_slice[2 * i];
            let bi = kernel_slice[2 * i + 1];

            signal_slice[2 * i] = ar * br - ai * bi;
            signal_slice[2 * i + 1] = ar * bi + ai * br;
        }

        unsafe { device.unmap_memory(kernel_memory) };
        unsafe { device.unmap_memory(signal_memory) };
    }

    unsafe { device.reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())? };
    unsafe { device.begin_command_buffer(command_buffer, &begin_info)? };
    signal_plan
        .launch_inverse(command_buffer)
        .map_err(|e| format!("signal inverse launch failed: {e:?}"))?;
    unsafe { device.end_command_buffer(command_buffer)? };

    unsafe { device.reset_fences(std::slice::from_ref(&fence))? };
    unsafe { device.queue_submit(queue, &submit_infos, fence)? };
    unsafe { device.wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)? };

    {
        let mapped = unsafe {
            device.map_memory(signal_memory, 0, buffer_bytes, vk::MemoryMapFlags::empty())?
        };
        let slice = unsafe { std::slice::from_raw_parts(mapped.cast::<f32>(), element_count) };

        let mut ok = true;
        for i in 0..N {
            let re = slice[2 * i];
            let im = slice[2 * i + 1];
            if (re - expected[i]).abs() > 1e-2 || im.abs() > 1e-2 {
                ok = false;
                break;
            }
        }

        unsafe { device.unmap_memory(signal_memory) };

        if !ok {
            return Err("Convolution output check failed".into());
        }
    }

    unsafe { device.device_wait_idle()? };

    drop(signal_plan);
    drop(kernel_plan);
    drop(signal_builder);
    drop(kernel_builder);

    unsafe {
        device.free_command_buffers(command_pool, std::slice::from_ref(&command_buffer));
        device.destroy_buffer(signal_buffer, None);
        device.destroy_buffer(kernel_buffer, None);
        device.free_memory(signal_memory, None);
        device.free_memory(kernel_memory, None);
        device.destroy_fence(fence, None);
        device.destroy_command_pool(command_pool, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
    }

    println!("VkFFT convolution example passed.");
    Ok(())
}

fn circular_convolution(a: &[f32; N], b: &[f32; N]) -> [f32; N] {
    let mut out = [0.0_f32; N];
    for n in 0..N {
        let mut acc = 0.0_f32;
        for k in 0..N {
            let j = (n + N - k) % N;
            acc += a[k] * b[j];
        }
        out[n] = acc;
    }
    out
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
        let has_flags = mem_props.memory_types[index as usize]
            .property_flags
            .contains(required);
        if supported && has_flags {
            return Some(index);
        }
    }

    None
}
