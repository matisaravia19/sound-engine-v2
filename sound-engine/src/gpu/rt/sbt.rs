//! Shader group handle packing and device-address regions for ray dispatch.
use super::pipeline::{RtShaderGroupSpec, ShaderGroupKind};
use crate::core::error::{SoundError, SoundResult};
use crate::gpu::backend::VkDeviceContext;
use crate::gpu::memory::{BufferHandle, GpuAllocator};
use ash::vk;

/// Owns the buffer backing all three trace regions for one ray-tracing pipeline.
/// Regions remain valid until this table drops; every referencing trace must finish first.
pub(super) struct ShaderBindingTable {
    _buffer: BufferHandle,
    /// Exactly one ray-generation record.
    pub(super) raygen: vk::StridedDeviceAddressRegionKHR,
    /// Contiguous miss records, or an empty region when none are configured.
    pub(super) miss: vk::StridedDeviceAddressRegionKHR,
    /// Contiguous surface/procedural hit records, or an empty region.
    pub(super) hit: vk::StridedDeviceAddressRegionKHR,
}

impl ShaderBindingTable {
    /// Packs handles from an existing pipeline whose group order has been validated.
    pub(super) fn new(
        device: &VkDeviceContext,
        memory: &GpuAllocator,
        pipeline: vk::Pipeline,
        groups: &[RtShaderGroupSpec],
    ) -> SoundResult<Self> {
        let props = &device.rt_properties.ray_tracing_pipeline;
        let handle_size = props.shader_group_handle_size as usize;
        let handle_alignment = props.shader_group_handle_alignment as usize;
        let base_alignment = props.shader_group_base_alignment as usize;
        if handle_size == 0
            || !handle_alignment.is_power_of_two()
            || !base_alignment.is_power_of_two()
        {
            return Err(SoundError::invalid_state(
                "Invalid Vulkan SBT alignment properties",
            ));
        }
        let handle_stride = align_up(handle_size, handle_alignment);
        let group_count = groups.len() as u32;
        // A base-aligned record stride lets every group-kind run start at a
        // valid trace-region address while preserving pipeline group order.
        let record_stride = align_up(handle_stride, base_alignment);
        if record_stride > props.max_shader_group_stride as usize {
            return Err(SoundError::invalid_argument(
                "SBT record stride exceeds the device limit",
            ));
        }
        let total_size = record_stride
            .checked_mul(groups.len())
            .and_then(|size| size.checked_add(base_alignment - 1))
            .ok_or_else(|| SoundError::invalid_argument("SBT allocation size overflow"))?;

        let handles = unsafe {
            device
                .ray_tracing_pipeline
                .get_ray_tracing_shader_group_handles(
                    pipeline,
                    0,
                    group_count,
                    handle_size * groups.len(),
                )?
        };

        // The allocation's address need not satisfy shaderGroupBaseAlignment.
        // Reserve a prefix and align the first record inside the actual buffer.
        let sbt = memory.create_host_device_address_buffer(
            total_size as vk::DeviceSize,
            vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
        )?;
        let allocation_address = memory.buffer_device_address(&sbt);
        let base_address = allocation_address
            .checked_add(base_alignment as u64 - 1)
            .map(|address| address & !(base_alignment as u64 - 1))
            .ok_or_else(|| SoundError::invalid_state("SBT device address overflow"))?;
        let prefix = (base_address - allocation_address) as usize;
        let mut sbt_bytes = vec![0_u8; total_size];
        for group_idx in 0..groups.len() {
            let src_offset = group_idx * handle_size;
            let dst_offset = prefix + group_idx * record_stride;
            sbt_bytes[dst_offset..dst_offset + handle_size]
                .copy_from_slice(&handles[src_offset..src_offset + handle_size]);
        }

        memory.write_mapped_bytes(&sbt, &sbt_bytes)?;

        Ok(Self {
            _buffer: sbt,
            raygen: region_for(groups, base_address, record_stride, ShaderGroupKind::Raygen),
            miss: region_for(groups, base_address, record_stride, ShaderGroupKind::Miss),
            hit: region_for(groups, base_address, record_stride, ShaderGroupKind::Hit),
        })
    }
}

fn region_for(
    groups: &[RtShaderGroupSpec],
    base_address: vk::DeviceAddress,
    record_stride: usize,
    kind: ShaderGroupKind,
) -> vk::StridedDeviceAddressRegionKHR {
    // SBT regions are contiguous runs by group kind.
    let start = groups.iter().position(|group| group.kind() == kind);
    let Some(start) = start else {
        return vk::StridedDeviceAddressRegionKHR::default();
    };
    let count = groups[start..]
        .iter()
        .take_while(|group| group.kind() == kind)
        .count();

    vk::StridedDeviceAddressRegionKHR {
        device_address: base_address + (start * record_stride) as vk::DeviceAddress,
        stride: record_stride as vk::DeviceSize,
        size: (count * record_stride) as vk::DeviceSize,
    }
}

fn align_up(value: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two());
    (value + alignment - 1) & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_regions_match_group_order_and_omit_missing_kinds() {
        let groups = [
            RtShaderGroupSpec::Miss {
                shader: 0,
            },
            RtShaderGroupSpec::Miss {
                shader: 1,
            },
            RtShaderGroupSpec::Raygen {
                shader: 2,
            },
            RtShaderGroupSpec::ProceduralHit {
                closest_hit_shader: 3,
                intersection_shader: 4,
            },
            RtShaderGroupSpec::TrianglesHit {
                closest_hit_shader: 5,
            },
        ];
        // Pipeline group order is preserved even when the ray-generation
        // record comes after a multi-record miss region.
        for (kind, address, size) in [
            (ShaderGroupKind::Raygen, 0x1080, 64),
            (ShaderGroupKind::Miss, 0x1000, 128),
            (ShaderGroupKind::Hit, 0x10c0, 128),
        ] {
            let region = region_for(&groups, 0x1000, 64, kind);
            assert_eq!(region.device_address, address);
            assert_eq!(region.stride, 64);
            assert_eq!(region.size, size);
        }
        for kind in [ShaderGroupKind::Miss, ShaderGroupKind::Hit] {
            let region = region_for(&groups[2..3], 0x1000, 64, kind);
            assert_eq!(
                (region.device_address, region.stride, region.size),
                (0, 0, 0)
            );
        }
    }

    #[test]
    fn align_up_keeps_aligned_values() {
        assert_eq!(align_up(64, 32), 64);
    }

    #[test]
    fn align_up_rounds_to_next_boundary() {
        assert_eq!(align_up(65, 32), 96);
    }
}
