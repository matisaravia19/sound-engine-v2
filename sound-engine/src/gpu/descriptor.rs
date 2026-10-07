//! Descriptor pool ownership shared by compute and ray tracing pipelines.

use super::backend::VkDeviceContext;
use crate::core::error::SoundResult;
use ash::vk;
use std::sync::Arc;

/// Number of sets reserved when a pipeline exhausts its existing pools.
pub(super) const DESCRIPTOR_SETS_PER_POOL: u32 = 64;

/// Descriptor set retained until the command buffer using it has completed.
#[derive(Clone, Copy)]
pub(super) struct DescriptorSetAllocation {
    /// Pool that owns the descriptor set.
    pub(super) pool: vk::DescriptorPool,
    /// Set referenced by commands recorded into one submission.
    pub(super) set: vk::DescriptorSet,
}

/// Owns the growable descriptor pools for one pipeline and retains their device.
///
/// Allocation and freeing must hold the shared compute context's pool guard.
/// The pipeline owner must wait for every referencing submission before drop.
pub(super) struct DescriptorPools {
    device: Arc<VkDeviceContext>,
    pool_sizes: Vec<vk::DescriptorPoolSize>,
    pools: Vec<vk::DescriptorPool>,
}

impl DescriptorPools {
    /// Prepares pool sizes from bindings already checked by layout validation.
    pub(super) fn new(
        device: Arc<VkDeviceContext>,
        bindings: &[vk::DescriptorSetLayoutBinding<'_>],
    ) -> Self {
        let mut pool_sizes = Vec::<vk::DescriptorPoolSize>::new();
        for binding in bindings {
            let count = binding.descriptor_count * DESCRIPTOR_SETS_PER_POOL;
            if let Some(size) = pool_sizes
                .iter_mut()
                .find(|size| size.ty == binding.descriptor_type)
            {
                size.descriptor_count += count;
            } else {
                pool_sizes.push(
                    vk::DescriptorPoolSize::default()
                        .ty(binding.descriptor_type)
                        .descriptor_count(count),
                );
            }
        }

        Self {
            device,
            pool_sizes,
            pools: Vec::new(),
        }
    }

    /// Allocates a set while the caller holds the shared descriptor-pool guard.
    pub(super) fn allocate(
        &mut self,
        layout: vk::DescriptorSetLayout,
    ) -> SoundResult<DescriptorSetAllocation> {
        for &pool in self.pools.iter().rev() {
            match self.allocate_from(pool, layout) {
                Ok(allocation) => return Ok(allocation),
                Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY | vk::Result::ERROR_FRAGMENTED_POOL) => {}
                Err(error) => return Err(error.into()),
            }
        }

        let info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(DESCRIPTOR_SETS_PER_POOL)
            .pool_sizes(&self.pool_sizes);
        let pool = unsafe { self.device.device.create_descriptor_pool(&info, None)? };
        // Retain ownership even if the following allocation fails.
        self.pools.push(pool);
        Ok(self.allocate_from(pool, layout)?)
    }

    fn allocate_from(
        &self,
        pool: vk::DescriptorPool,
        layout: vk::DescriptorSetLayout,
    ) -> Result<DescriptorSetAllocation, vk::Result> {
        let layouts = [layout];
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        let set = unsafe { self.device.device.allocate_descriptor_sets(&info)?[0] };
        Ok(DescriptorSetAllocation {
            pool,
            set,
        })
    }
}

impl Drop for DescriptorPools {
    fn drop(&mut self) {
        for &pool in &self.pools {
            unsafe { self.device.device.destroy_descriptor_pool(pool, None) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::backend::VkBackend;
    use crate::gpu::resource::PipelineResources;

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
}
