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
#[path = "../../tests/unit/gpu/descriptor.rs"]
mod tests;
