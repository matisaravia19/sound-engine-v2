use super::backend::VkDeviceContext;
use crate::core::error::{SoundError, SoundResult};
use ash::vk;
use std::sync::Arc;

/// Owns an initialization fence and keeps its Vulkan device alive.
pub(crate) struct OwnedFence {
    /// Fence available for serialized initialization work.
    pub(crate) handle: vk::Fence,
    device: Arc<VkDeviceContext>,
}

impl OwnedFence {
    /// Creates an unsignaled fence; initialization must finish before destruction.
    pub(crate) fn new(device: Arc<VkDeviceContext>) -> SoundResult<Self> {
        Ok(Self {
            handle: device.create_fence()?,
            device,
        })
    }
}

impl Drop for OwnedFence {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_fence(self.handle, None);
        }
    }
}

/// Owns a pipeline and its single descriptor-set layout from construction through teardown.
/// Retains the Vulkan device; submitted work must finish before this owner is dropped.
pub(crate) struct PipelineResources {
    /// Pipeline created so far, or null before successful pipeline creation.
    pub(crate) handle: vk::Pipeline,
    /// Pipeline layout used by dispatches and push constants.
    pub(crate) layout: vk::PipelineLayout,
    /// Descriptor-set layout used by per-dispatch descriptor allocations.
    pub(crate) descriptor_set_layout: vk::DescriptorSetLayout,
    device: Arc<VkDeviceContext>,
}

impl PipelineResources {
    /// Validates and creates both layouts, releasing partial allocations on failure.
    /// The caller then installs the created pipeline in `handle` before registration.
    pub(crate) fn new(
        device: Arc<VkDeviceContext>,
        bindings: &[vk::DescriptorSetLayoutBinding<'_>],
        ranges: &[vk::PushConstantRange],
    ) -> SoundResult<Self> {
        validate_pipeline_layout(bindings, ranges, &device.limits)?;
        let mut resources = Self {
            handle: vk::Pipeline::null(),
            layout: vk::PipelineLayout::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            device,
        };
        let set_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(bindings);
        resources.descriptor_set_layout = unsafe {
            resources
                .device
                .device
                .create_descriptor_set_layout(&set_info, None)?
        };
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&resources.descriptor_set_layout))
            .push_constant_ranges(ranges);
        resources.layout = unsafe {
            resources
                .device
                .device
                .create_pipeline_layout(&layout_info, None)?
        };
        Ok(resources)
    }
}

impl Drop for PipelineResources {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_pipeline(self.handle, None);
            self.device
                .device
                .destroy_pipeline_layout(self.layout, None);
            self.device
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

/// Validates the layout subset supported by the engine before invoking Vulkan.
fn validate_pipeline_layout(
    bindings: &[vk::DescriptorSetLayoutBinding<'_>],
    ranges: &[vk::PushConstantRange],
    limits: &vk::PhysicalDeviceLimits,
) -> SoundResult<()> {
    let mut seen = std::collections::HashSet::new();
    let mut storage_count = 0_u64;
    let mut descriptor_count = 0_u64;
    for binding in bindings {
        if binding.descriptor_count == 0
            || binding.stage_flags.is_empty()
            || !seen.insert(binding.binding)
        {
            return Err(SoundError::invalid_argument(
                "Descriptor bindings must be unique, non-empty and visible to a shader stage",
            ));
        }
        descriptor_count += binding.descriptor_count as u64;
        if binding.descriptor_type == vk::DescriptorType::STORAGE_BUFFER {
            storage_count += binding.descriptor_count as u64;
        }
    }
    if descriptor_count > u32::MAX as u64 / super::descriptor::DESCRIPTOR_SETS_PER_POOL as u64
        || storage_count > limits.max_descriptor_set_storage_buffers as u64
    {
        return Err(SoundError::invalid_argument(
            "Descriptor counts exceed pool or device capacity",
        ));
    }
    let mut stages = vk::ShaderStageFlags::empty();
    for range in ranges {
        if range.size == 0
            || range.size % 4 != 0
            || range.offset % 4 != 0
            || range.stage_flags.is_empty()
            || stages.intersects(range.stage_flags)
            || range
                .offset
                .checked_add(range.size)
                .is_none_or(|end| end > limits.max_push_constants_size)
        {
            return Err(SoundError::invalid_argument(
                "Invalid push constant range or repeated shader stage",
            ));
        }
        stages |= range.stage_flags;
    }
    Ok(())
}

/// Requires a complete, single-element descriptor set; partial writes leave new sets undefined.
pub(crate) fn validate_descriptor_writes(
    bindings: &[vk::DescriptorSetLayoutBinding<'_>],
    writes: &[(u32, vk::DescriptorType)],
) -> SoundResult<()> {
    if writes.len() != bindings.len()
        || bindings.iter().any(|binding| {
            binding.descriptor_count != 1
                || writes
                    .iter()
                    .filter(|(index, ty)| {
                        *index == binding.binding && *ty == binding.descriptor_type
                    })
                    .count()
                    != 1
        })
    {
        return Err(SoundError::invalid_argument(
            "Dispatch must initialize each single-element descriptor binding exactly once with the correct type",
        ));
    }
    Ok(())
}

/// Validates storage descriptor alignment and range; buffer lifetime remains caller-owned.
pub(crate) fn validate_storage_descriptor(
    buffer: vk::Buffer,
    offset: u64,
    range: u64,
    limits: &vk::PhysicalDeviceLimits,
) -> SoundResult<()> {
    if buffer == vk::Buffer::null()
        || !offset.is_multiple_of(limits.min_storage_buffer_offset_alignment.max(1))
        || range == 0
        || (range != vk::WHOLE_SIZE
            && (range > limits.max_storage_buffer_range as u64
                || offset.checked_add(range).is_none()))
    {
        return Err(SoundError::invalid_argument(
            "Invalid storage buffer descriptor handle, alignment or range",
        ));
    }
    Ok(())
}

/// Finds the shader stages covering an aligned push-constant write in its entirety.
pub(crate) fn push_constant_stages(
    ranges: &[vk::PushConstantRange],
    offset: u32,
    size: usize,
) -> SoundResult<vk::ShaderStageFlags> {
    if size == 0 {
        return Ok(vk::ShaderStageFlags::empty());
    }
    let end = u32::try_from(size)
        .ok()
        .and_then(|size| offset.checked_add(size));
    if !offset.is_multiple_of(4) || !size.is_multiple_of(4) || end.is_none() {
        return Err(SoundError::invalid_argument(
            "Push constant writes must be aligned and fit 32-bit offsets",
        ));
    }
    let end = end.unwrap();
    let mut stages = vk::ShaderStageFlags::empty();
    for range in ranges {
        let range_end = range.offset.saturating_add(range.size);
        if offset < range_end && end > range.offset {
            if offset < range.offset || end > range_end {
                return Err(SoundError::invalid_argument(
                    "Push constant write only partially overlaps a shader range",
                ));
            }
            stages |= range.stage_flags;
        }
    }
    if stages.is_empty() {
        return Err(SoundError::invalid_argument(
            "Push constant write is outside the pipeline layout",
        ));
    }
    Ok(stages)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_writes_check_alignment_overflow_bounds_and_stage_visibility() {
        let ranges = [vk::PushConstantRange::default()
            .offset(4)
            .size(8)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)];
        assert_eq!(
            push_constant_stages(&ranges, 4, 8).unwrap(),
            vk::ShaderStageFlags::RAYGEN_KHR
        );
        for (offset, size) in [
            (0, 8),
            (4, 12),
            (5, 4),
            (4, 3),
            (u32::MAX - 3, 8),
            (0, usize::MAX),
        ] {
            assert!(push_constant_stages(&ranges, offset, size).is_err());
        }
    }

    #[test]
    fn descriptors_reject_missing_duplicate_wrong_type_and_uninitialized_arrays() {
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER);
        let write = (0, vk::DescriptorType::STORAGE_BUFFER);
        assert!(validate_descriptor_writes(&[binding], &[write]).is_ok());
        assert!(validate_descriptor_writes(&[binding], &[]).is_err());
        assert!(validate_descriptor_writes(&[binding], &[write, write]).is_err());
        assert!(
            validate_descriptor_writes(&[binding], &[(0, vk::DescriptorType::UNIFORM_BUFFER)])
                .is_err()
        );
        assert!(validate_descriptor_writes(&[binding.descriptor_count(2)], &[write]).is_err());
    }
}
