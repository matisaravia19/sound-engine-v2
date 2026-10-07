use super::*;
use crate::gpu::backend::VkBackend;
use crate::gpu::resource::PipelineResources;

#[test]
fn storage_writes_reject_invalid_handles_offsets_and_ranges() {
    use ash::vk::Handle;
    // This handle is inspected as data only; no Vulkan call uses it.
    let buffer = vk::Buffer::from_raw(1);
    let limits = vk::PhysicalDeviceLimits {
        min_storage_buffer_offset_alignment: 8,
        max_storage_buffer_range: 64,
        ..Default::default()
    };
    let whole = StorageBufferWrite::new(0, buffer);
    assert!(whole.validate(&limits).is_ok());
    assert!(
        StorageBufferWrite {
            offset: 16,
            range: 64,
            ..whole
        }
        .validate(&limits)
        .is_ok()
    );
    for invalid in [
        StorageBufferWrite::new(0, vk::Buffer::null()),
        StorageBufferWrite {
            offset: 4,
            ..whole
        },
        StorageBufferWrite {
            range: 0,
            ..whole
        },
        StorageBufferWrite {
            range: 65,
            ..whole
        },
        StorageBufferWrite {
            offset: u64::MAX - 7,
            range: 16,
            ..whole
        },
    ] {
        assert!(invalid.validate(&limits).is_err());
    }
}

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn descriptor_pools_grow_and_reuse_freed_sets() {
    let gpu = VkBackend::new().unwrap();
    let device = gpu.device_context();
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_count(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
    ];
    let layout_owner = PipelineResources::new(device.clone(), &bindings, &[]).unwrap();
    let mut pools = DescriptorPools::new(device.clone(), &bindings);
    let _pool_access = gpu.compute().descriptor_pool_guard();
    let mut initial_pool_count = 0;
    for cycle in 0..3 {
        let allocations: Vec<_> = (0..DESCRIPTOR_SETS_PER_POOL * 2 + 1)
            .map(|_| pools.allocate(layout_owner.descriptor_set_layout).unwrap())
            .collect();
        if cycle == 0 {
            initial_pool_count = pools.pools.len();
        } else {
            assert_eq!(pools.pools.len(), initial_pool_count);
        }
        // No submission uses these sets; freeing them immediately is safe.
        for allocation in allocations {
            unsafe {
                device
                    .device
                    .free_descriptor_sets(allocation.pool, &[allocation.set])
            }
            .unwrap();
        }
    }
}
