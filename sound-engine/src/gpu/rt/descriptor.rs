use super::*;
use crate::gpu::GpuError;

impl RtContext {
    pub(super) fn acquire_descriptor_set(
        &self,
        pipeline: &mut RayTracingPipeline,
    ) -> Result<vk::DescriptorSet, GpuError> {
        for pool in pipeline.descriptor_pools.iter().rev() {
            if let Ok(set) = self.try_allocate_descriptor_set(*pool, pipeline.descriptor_set_layout) {
                return Ok(set);
            }
        }

        let descriptor_pool = self.create_descriptor_pool(pipeline.pool_sizes_template.as_slice())?;
        let descriptor_set = self.try_allocate_descriptor_set(descriptor_pool, pipeline.descriptor_set_layout)?;
        pipeline.descriptor_pools.push(descriptor_pool);
        Ok(descriptor_set)
    }

    fn create_descriptor_pool(
        &self,
        pool_sizes_template: &[vk::DescriptorPoolSize],
    ) -> Result<vk::DescriptorPool, GpuError> {
        let pool_sizes = pool_sizes_template
            .iter()
            .map(|size| {
                vk::DescriptorPoolSize::default()
                    .ty(size.ty)
                    .descriptor_count(size.descriptor_count * DESCRIPTOR_SETS_PER_POOL)
            })
            .collect::<Vec<_>>();

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(DESCRIPTOR_SETS_PER_POOL)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe { self.device_context.device.create_descriptor_pool(&pool_info, None)? };
        Ok(descriptor_pool)
    }

    fn try_allocate_descriptor_set(
        &self,
        descriptor_pool: vk::DescriptorPool,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> Result<vk::DescriptorSet, GpuError> {
        let set_layouts = [descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);
        let descriptor_set = unsafe { self.device_context.device.allocate_descriptor_sets(&alloc_info)?[0] };
        Ok(descriptor_set)
    }
}

/// Aggregates descriptor counts by type for RT descriptor-pool creation.
pub(super) fn aggregate_pool_sizes(bindings: &[vk::DescriptorSetLayoutBinding<'_>]) -> Vec<vk::DescriptorPoolSize> {
    let mut pool_sizes = Vec::<vk::DescriptorPoolSize>::new();
    for binding in bindings {
        if let Some(existing) = pool_sizes.iter_mut().find(|entry| entry.ty == binding.descriptor_type) {
            existing.descriptor_count += binding.descriptor_count;
        } else {
            pool_sizes.push(
                vk::DescriptorPoolSize::default()
                    .ty(binding.descriptor_type)
                    .descriptor_count(binding.descriptor_count),
            );
        }
    }
    pool_sizes
}

#[cfg(test)]
mod tests {
    use super::aggregate_pool_sizes;
    use ash::vk;

    #[test]
    fn aggregate_pool_sizes_merges_same_descriptor_types() {
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1),
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(2),
        ];

        let sizes = aggregate_pool_sizes(&bindings);

        assert_eq!(sizes.len(), 1);
        assert_eq!(sizes[0].ty, vk::DescriptorType::STORAGE_BUFFER);
        assert_eq!(sizes[0].descriptor_count, 3);
    }
}
