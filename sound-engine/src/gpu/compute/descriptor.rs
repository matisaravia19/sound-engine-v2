use super::*;

impl ComputeContext {
    pub(super) fn acquire_descriptor_set(
        &self,
        pipeline: &mut ComputePipeline,
    ) -> SoundResult<DescriptorSetAllocation> {
        for pool in pipeline.descriptor_pools.iter().rev() {
            match self.try_allocate_descriptor_set(*pool, pipeline.descriptor_set_layout) {
                Ok(set) => {
                    return Ok(DescriptorSetAllocation {
                        pool: *pool,
                        set,
                    });
                }
                Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY | vk::Result::ERROR_FRAGMENTED_POOL) => {}
                Err(error) => return Err(error.into()),
            }
        }

        let descriptor_pool =
            self.create_descriptor_pool(pipeline.pool_sizes_template.as_slice())?;
        pipeline.descriptor_pools.push(descriptor_pool);
        let descriptor_set =
            self.try_allocate_descriptor_set(descriptor_pool, pipeline.descriptor_set_layout)?;
        Ok(DescriptorSetAllocation {
            pool: descriptor_pool,
            set: descriptor_set,
        })
    }

    fn create_descriptor_pool(
        &self,
        pool_sizes_template: &[vk::DescriptorPoolSize],
    ) -> SoundResult<vk::DescriptorPool> {
        let pool_sizes = pool_sizes_template
            .iter()
            .map(|size| {
                vk::DescriptorPoolSize::default()
                    .ty(size.ty)
                    .descriptor_count(size.descriptor_count * DESCRIPTOR_SETS_PER_POOL)
            })
            .collect::<Vec<_>>();

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(DESCRIPTOR_SETS_PER_POOL)
            .pool_sizes(&pool_sizes);

        let descriptor_pool = unsafe {
            self.device_context
                .device
                .create_descriptor_pool(&pool_info, None)?
        };
        Ok(descriptor_pool)
    }

    fn try_allocate_descriptor_set(
        &self,
        descriptor_pool: vk::DescriptorPool,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> Result<vk::DescriptorSet, vk::Result> {
        let set_layouts = [descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);
        let descriptor_set = unsafe {
            self.device_context
                .device
                .allocate_descriptor_sets(&alloc_info)?[0]
        };
        Ok(descriptor_set)
    }
}
