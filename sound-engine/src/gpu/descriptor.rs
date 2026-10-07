//! Storage-buffer bindings and descriptor-pool ownership shared by GPU pipelines.

use super::backend::VkDeviceContext;
use crate::core::error::{SoundError, SoundResult};
use ash::vk;
use std::sync::Arc;

/// One storage-buffer descriptor written before a compute or ray dispatch.
///
/// This is a borrowed raw handle: the caller retains and synchronizes the buffer
/// through command completion. The binding exposes either the whole buffer or
/// an explicitly aligned byte range.
#[derive(Debug, Clone, Copy)]
pub struct StorageBufferWrite {
    /// Descriptor binding number to update.
    pub binding: u32,
    /// Vulkan buffer handle borrowed for the dispatch.
    pub buffer: vk::Buffer,
    /// Byte offset into the buffer.
    pub offset: vk::DeviceSize,
    /// Byte range exposed to the shader, or `vk::WHOLE_SIZE`.
    pub range: vk::DeviceSize,
}

impl StorageBufferWrite {
    /// Binds the whole buffer at the given descriptor index.
    pub fn new(binding: u32, buffer: vk::Buffer) -> Self {
        Self {
            binding,
            buffer,
            offset: 0,
            range: vk::WHOLE_SIZE,
        }
    }

    /// Checks handle presence, range arithmetic, and the device's alignment limits.
    /// Allocation bounds and resource lifetime remain the dispatch caller's responsibility.
    pub(super) fn validate(&self, limits: &vk::PhysicalDeviceLimits) -> SoundResult<()> {
        if self.buffer == vk::Buffer::null()
            || !self
                .offset
                .is_multiple_of(limits.min_storage_buffer_offset_alignment.max(1))
            || self.range == 0
            || (self.range != vk::WHOLE_SIZE
                && (self.range > limits.max_storage_buffer_range as u64
                    || self.offset.checked_add(self.range).is_none()))
        {
            return Err(SoundError::invalid_argument(
                "Invalid storage buffer descriptor handle, alignment or range",
            ));
        }
        Ok(())
    }

    /// Builds the Vulkan view borrowed by a descriptor write.
    pub(super) fn buffer_info(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::default()
            .buffer(self.buffer)
            .offset(self.offset)
            .range(self.range)
    }
}

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
