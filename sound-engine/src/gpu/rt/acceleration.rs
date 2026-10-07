use glam::{Mat4, Vec3};

use super::scene::rt_aabb_stride;
use super::*;
use crate::core::error::{SoundError, SoundResult};
use std::mem::size_of;

/// Stable handle to a bottom-level acceleration structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlasId(pub u32);

/// Stable handle to a top-level acceleration structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TlasId(pub u32);

/// Parameters for building a BLAS from uploaded triangle mesh buffers.
pub struct BlasBuildSpec<'a> {
    /// Uploaded mesh buffers used as build input.
    pub mesh: &'a RtMeshBuffers,
    /// Vulkan build flags, such as `PREFER_FAST_TRACE`.
    pub flags: vk::BuildAccelerationStructureFlagsKHR,
}

/// Parameters for building a BLAS from uploaded procedural AABB buffers.
pub struct AabbBlasBuildSpec<'a> {
    /// Uploaded AABB buffer used as build input.
    pub aabbs: &'a RtAabbBuffers,
    /// Vulkan build flags, such as `PREFER_FAST_TRACE`.
    pub flags: vk::BuildAccelerationStructureFlagsKHR,
}

/// Parameters for building a TLAS from BLAS instances.
pub struct TlasBuildSpec<'a> {
    /// Instances to include in the TLAS.
    pub instances: &'a [RtInstanceSpec],
    /// Vulkan build flags, such as `PREFER_FAST_TRACE`.
    pub flags: vk::BuildAccelerationStructureFlagsKHR,
}

impl RtContext {
    /// Builds a bottom-level acceleration structure from one triangle mesh.
    ///
    /// The build is submitted to the compute queue and completed before the new
    /// `BlasId` is returned.
    ///
    /// # Safety
    /// Buffers must have AS-build-input usage and contain finite triangle data
    /// with in-bounds indices matching the declared counts. Their contents must
    /// remain synchronized and unchanged until this synchronous build completes.
    /// Build flags must be supported and valid for triangle geometry.
    pub unsafe fn build_blas(&self, spec: BlasBuildSpec<'_>) -> SoundResult<BlasId> {
        self.memory.validate_owner(&spec.mesh.vertex_buffer)?;
        if let Some(buffer) = &spec.mesh.index_buffer {
            self.memory.validate_owner(buffer)?;
        }
        if spec.mesh.vertex_count == 0
            || spec.mesh.vertex_count as u64 * size_of::<Vec3>() as u64
                > spec.mesh.vertex_buffer.size
            || spec.mesh.index_buffer.as_ref().is_some_and(|buffer| {
                spec.mesh.index_count as u64 * 4 > buffer.size
                    || !spec.mesh.index_count.is_multiple_of(3)
            })
            || (spec.mesh.index_buffer.is_none() && !spec.mesh.vertex_count.is_multiple_of(3))
        {
            return Err(SoundError::invalid_argument(
                "BLAS geometry counts exceed input buffer bounds or contain incomplete triangles",
            ));
        }
        // Geometry build inputs reference buffers by device address, not descriptors.
        let vertex_address = self.memory.buffer_device_address(&spec.mesh.vertex_buffer);
        let mut triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .vertex_data(vk::DeviceOrHostAddressConstKHR {
                device_address: vertex_address,
            })
            .vertex_stride(size_of::<Vec3>() as vk::DeviceSize)
            .max_vertex(spec.mesh.vertex_count.saturating_sub(1));

        let primitive_count = if let Some(index_buffer) = &spec.mesh.index_buffer {
            triangles = triangles.index_type(vk::IndexType::UINT32).index_data(
                vk::DeviceOrHostAddressConstKHR {
                    device_address: self.memory.buffer_device_address(index_buffer),
                },
            );
            spec.mesh.index_count / 3
        } else {
            triangles = triangles.index_type(vk::IndexType::NONE_KHR);
            spec.mesh.vertex_count / 3
        };

        if primitive_count == 0 {
            return Err(SoundError::invalid_argument(
                "RT BLAS must contain at least one triangle",
            ));
        }

        let mut geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                triangles,
            });
        if spec.mesh.opaque {
            geometry = geometry.flags(vk::GeometryFlagsKHR::OPAQUE);
        }

        self.build_bottom_level(&geometry, primitive_count, spec.flags)
    }

    /// Builds a bottom-level acceleration structure from procedural AABBs.
    ///
    /// The build is submitted to the compute queue and completed before the new
    /// `BlasId` is returned. The returned BLAS requires a procedural hit group
    /// with an intersection shader when instanced in a traced TLAS.
    ///
    /// # Safety
    /// The buffer must have AS-build-input usage and contain finite, ordered
    /// AABBs matching the declared count. Synchronize its contents through this
    /// build, and use supported build flags valid for AABB geometry.
    pub unsafe fn build_aabb_blas(&self, spec: AabbBlasBuildSpec<'_>) -> SoundResult<BlasId> {
        self.memory.validate_owner(&spec.aabbs.aabb_buffer)?;
        if spec.aabbs.primitive_count as u64 * rt_aabb_stride() > spec.aabbs.aabb_buffer.size {
            return Err(SoundError::invalid_argument(
                "AABB count exceeds input buffer bounds",
            ));
        }
        if spec.aabbs.primitive_count == 0 {
            return Err(SoundError::invalid_argument(
                "RT AABB BLAS must contain at least one primitive",
            ));
        }

        let aabbs = vk::AccelerationStructureGeometryAabbsDataKHR::default()
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: self.memory.buffer_device_address(&spec.aabbs.aabb_buffer),
            })
            .stride(rt_aabb_stride());

        let mut geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::AABBS)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                aabbs,
            });
        if spec.aabbs.opaque {
            geometry = geometry.flags(vk::GeometryFlagsKHR::OPAQUE);
        }

        self.build_bottom_level(&geometry, spec.aabbs.primitive_count, spec.flags)
    }

    /// Builds a top-level acceleration structure from BLAS instances.
    ///
    /// The instance buffer is uploaded as host-visible device-address data and
    /// the build completes before the returned `TlasId` can be used for tracing.
    /// Referenced BLAS storage is retained through construction and the TLAS lifetime.
    pub fn build_tlas(&self, spec: TlasBuildSpec<'_>) -> SoundResult<TlasId> {
        if spec.instances.len() as u64
            > self
                .device_context
                .rt_properties
                .acceleration_structure
                .max_instance_count
            || spec.instances.len() > u32::MAX as usize
        {
            return Err(SoundError::invalid_argument(
                "TLAS instance count exceeds device limits",
            ));
        }
        if spec.instances.is_empty() {
            return Err(SoundError::invalid_argument(
                "RT TLAS must contain at least one instance",
            ));
        }

        // Acquire strong references under the registry lock so a concurrent scene
        // replacement cannot invalidate an address during construction or tracing.
        let (instances, referenced_blas) = self.resolve_instances(spec.instances)?;

        // Vulkan consumes TLAS instances through a device address.
        let instance_buffer = self.memory.create_host_device_address_buffer(
            std::mem::size_of_val(instances.as_slice()) as vk::DeviceSize,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        )?;
        self.memory.write_mapped_bytes(
            &instance_buffer,
            acceleration_instances_as_bytes(instances.as_slice()),
        )?;
        let instance_address = self.memory.buffer_device_address(&instance_buffer);

        let instances_data = vk::AccelerationStructureGeometryInstancesDataKHR::default()
            .array_of_pointers(false)
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: instance_address,
            });
        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: instances_data,
            });
        let mut handle = self.build_acceleration_structure(
            vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            spec.flags,
            &geometry,
            instances.len() as u32,
        )?;

        let mut tlas = self
            .tlas
            .lock()
            .map_err(|_| SoundError::poisoned_lock("RT TLAS lock is poisoned"))?;
        // Do not let a stale ID alias a later scene/query after removal.
        let raw_id = self
            .next_tlas_id
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| SoundError::resource_allocation_failed("RT TLAS id space exhausted"))?;
        let id = TlasId(raw_id);
        handle._referenced_blas = referenced_blas;
        tlas.insert(id, Arc::new(handle));
        Ok(id)
    }

    /// Destroys a TLAS that is no longer referenced by submitted GPU work.
    ///
    /// Removing the handle releases both the Vulkan acceleration structure and
    /// its backing allocation. Removed identifiers are never reused.
    ///
    /// # Safety
    /// The caller must own `id`, finish every submission referencing it, and
    /// prevent concurrent recording or submission that could use the TLAS.
    /// IDs borrowed from a scene or another scoped owner must not be destroyed.
    pub unsafe fn destroy_tlas(&self, id: TlasId) -> SoundResult<()> {
        let handle = self
            .tlas
            .lock()
            .map_err(|_| SoundError::poisoned_lock("RT TLAS lock is poisoned"))?
            .remove(&id)
            .ok_or_else(|| SoundError::not_found(format!("Invalid TLAS id {}", id.0)))?;
        drop(handle);
        Ok(())
    }

    fn build_bottom_level(
        &self,
        geometry: &vk::AccelerationStructureGeometryKHR<'_>,
        primitive_count: u32,
        flags: vk::BuildAccelerationStructureFlagsKHR,
    ) -> SoundResult<BlasId> {
        let handle = self.build_acceleration_structure(
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            flags,
            geometry,
            primitive_count,
        )?;

        let mut blas = self
            .blas
            .lock()
            .map_err(|_| SoundError::poisoned_lock("RT BLAS lock is poisoned"))?;
        let id = BlasId(
            self.next_blas_id
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
                .map_err(|_| {
                    SoundError::resource_allocation_failed("RT BLAS id space exhausted")
                })?,
        );
        blas.insert(id, Arc::new(handle));
        Ok(id)
    }

    fn build_acceleration_structure(
        &self,
        ty: vk::AccelerationStructureTypeKHR,
        flags: vk::BuildAccelerationStructureFlagsKHR,
        geometry: &vk::AccelerationStructureGeometryKHR<'_>,
        primitive_count: u32,
    ) -> SoundResult<AccelerationStructureHandle> {
        validate_build_flags(flags)?;
        if ty == vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL
            && primitive_count as u64
                > self
                    .device_context
                    .rt_properties
                    .acceleration_structure
                    .max_primitive_count
        {
            return Err(SoundError::invalid_argument(
                "BLAS primitive count exceeds device limits",
            ));
        }
        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(ty)
            .flags(flags)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(std::slice::from_ref(geometry));

        // Query Vulkan for AS storage and temporary scratch requirements.
        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            self.device_context
                .acceleration_structure
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &build_info,
                    &[primitive_count],
                    &mut size_info,
                );
        }

        let scratch_alignment =
            self.device_context
                .rt_properties
                .acceleration_structure
                .min_acceleration_structure_scratch_offset_alignment as vk::DeviceSize;
        if scratch_alignment == 0 || !scratch_alignment.is_power_of_two() {
            return Err(SoundError::invalid_state(format!(
                "Invalid acceleration-structure scratch alignment {scratch_alignment}"
            )));
        }
        let scratch_allocation_size = size_info
            .build_scratch_size
            .checked_add(scratch_alignment - 1)
            .ok_or_else(|| {
                SoundError::resource_allocation_failed("RT scratch buffer size overflow")
            })?;

        // The AS object is backed by a buffer owned by the returned handle.
        let storage = self.memory.create_device_address_buffer(
            size_info.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
        )?;

        // A VMA buffer's base device address is not guaranteed to meet the
        // stricter acceleration-structure scratch alignment. Reserve padding
        // and point Vulkan at an aligned address inside the allocation.
        // Scratch is temporary; the synchronous build below finishes before it drops.
        let scratch = self.memory.create_device_address_buffer(
            scratch_allocation_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
        let scratch_address = align_device_address(
            self.memory.buffer_device_address(&scratch),
            scratch_alignment,
        )
        .ok_or_else(|| {
            SoundError::invalid_state("Acceleration-structure scratch address overflow")
        })?;

        let create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(storage.buffer)
            .size(size_info.acceleration_structure_size)
            .ty(ty);
        let handle = unsafe {
            self.device_context
                .acceleration_structure
                .create_acceleration_structure(&create_info, None)?
        };
        // Own the Vulkan object immediately so failed submissions release it
        // before its backing buffer, just like a completed build.
        let mut acceleration_structure = AccelerationStructureHandle {
            handle,
            device_address: 0,
            _storage: storage,
            _referenced_blas: Vec::new(),
            device_context: self.device_context.clone(),
        };
        let build_info = build_info.dst_acceleration_structure(handle).scratch_data(
            vk::DeviceOrHostAddressKHR {
                device_address: scratch_address,
            },
        );
        let range =
            vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(primitive_count);

        self.compute
            .submit_compute_and_wait(|command_buffer| unsafe {
                self.device_context
                    .acceleration_structure
                    .cmd_build_acceleration_structures(
                        command_buffer,
                        std::slice::from_ref(&build_info),
                        &[std::slice::from_ref(&range)],
                    );

                Ok(())
            })?;

        let address_info =
            vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(handle);
        acceleration_structure.device_address = unsafe {
            self.device_context
                .acceleration_structure
                .get_acceleration_structure_device_address(&address_info)
        };

        Ok(acceleration_structure)
    }

    fn resolve_instances(
        &self,
        specs: &[RtInstanceSpec],
    ) -> SoundResult<(
        Vec<vk::AccelerationStructureInstanceKHR>,
        Vec<Arc<AccelerationStructureHandle>>,
    )> {
        let blas = self
            .blas
            .lock()
            .map_err(|_| SoundError::poisoned_lock("RT BLAS lock is poisoned"))?;

        let mut referenced_blas = HashMap::new();
        let instances = specs
            .iter()
            .map(|instance| {
                validate_instance_transform(instance.transform)?;
                if instance.custom_index > 0x00ff_ffff || instance.sbt_record_offset > 0x00ff_ffff {
                    return Err(SoundError::invalid_argument(
                        "TLAS custom index and SBT offset must fit 24 bits",
                    ));
                }
                let blas_handle = blas.get(&instance.blas_id).ok_or_else(|| {
                    SoundError::not_found(format!("Invalid BLAS id {}", instance.blas_id.0))
                })?;

                referenced_blas
                    .entry(instance.blas_id)
                    .or_insert_with(|| blas_handle.clone());
                Ok(vk::AccelerationStructureInstanceKHR {
                    transform: vk::TransformMatrixKHR {
                        matrix: matrix_to_tlas_transform(instance.transform),
                    },
                    instance_custom_index_and_mask: vk::Packed24_8::new(
                        instance.custom_index,
                        instance.mask,
                    ),
                    instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                        instance.sbt_record_offset,
                        0,
                    ),
                    acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                        device_handle: blas_handle.device_address,
                    },
                })
            })
            .collect::<SoundResult<Vec<_>>>()?;
        Ok((instances, referenced_blas.into_values().collect()))
    }
}

// The backend enables base KHR acceleration structures, without motion or micromaps.
fn validate_build_flags(flags: vk::BuildAccelerationStructureFlagsKHR) -> SoundResult<()> {
    let supported = vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE
        | vk::BuildAccelerationStructureFlagsKHR::ALLOW_COMPACTION
        | vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
        | vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_BUILD
        | vk::BuildAccelerationStructureFlagsKHR::LOW_MEMORY;
    if !(flags & !supported).is_empty() {
        return Err(SoundError::unsupported_feature(
            "Acceleration build flags require disabled extensions or contain unknown bits",
        ));
    }
    if flags.contains(
        vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
            | vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_BUILD,
    ) {
        return Err(SoundError::invalid_argument(
            "Acceleration builds cannot prefer fast trace and fast build simultaneously",
        ));
    }
    Ok(())
}

fn align_device_address(
    address: vk::DeviceAddress,
    alignment: vk::DeviceSize,
) -> Option<vk::DeviceAddress> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return None;
    }
    address
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
}

fn acceleration_instances_as_bytes(instances: &[vk::AccelerationStructureInstanceKHR]) -> &[u8] {
    // SAFETY:
    // - Vulkan defines VkAccelerationStructureInstanceKHR as the exact byte layout
    //   consumed for TLAS instance builds.
    // - ash represents it as a repr(C)-compatible Vulkan FFI type.
    // - The returned byte slice is only used immediately for copying into mapped memory.
    unsafe {
        std::slice::from_raw_parts(
            instances.as_ptr().cast::<u8>(),
            std::mem::size_of_val(instances),
        )
    }
}

// Converts a 4x4 column-major matrix to the 3x4 row-major format required for TLAS instance transforms.
fn matrix_to_tlas_transform(matrix: Mat4) -> [f32; 12] {
    [
        matrix.col(0).x,
        matrix.col(1).x,
        matrix.col(2).x,
        matrix.col(3).x,
        matrix.col(0).y,
        matrix.col(1).y,
        matrix.col(2).y,
        matrix.col(3).y,
        matrix.col(0).z,
        matrix.col(1).z,
        matrix.col(2).z,
        matrix.col(3).z,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat4, Vec3};

    #[test]
    fn build_flags_reject_unknown_features_and_conflicting_preferences() {
        use vk::BuildAccelerationStructureFlagsKHR as Flags;
        for flags in [
            Flags::empty(),
            Flags::PREFER_FAST_TRACE | Flags::ALLOW_UPDATE,
            Flags::PREFER_FAST_BUILD | Flags::LOW_MEMORY | Flags::ALLOW_COMPACTION,
        ] {
            assert!(validate_build_flags(flags).is_ok());
        }
        for flags in [
            Flags::MOTION_NV,
            Flags::from_raw(u32::MAX),
            Flags::PREFER_FAST_TRACE | Flags::PREFER_FAST_BUILD,
        ] {
            assert!(validate_build_flags(flags).is_err());
        }
    }

    #[test]
    fn scratch_device_address_alignment_handles_unaligned_allocations() {
        assert_eq!(align_device_address(0x1000, 128), Some(0x1000));
        assert_eq!(align_device_address(0x1001, 128), Some(0x1080));
        assert_eq!(align_device_address(0x107f, 128), Some(0x1080));
        assert_eq!(align_device_address(u64::MAX, 128), None);
        assert_eq!(align_device_address(0x1001, 0), None);
        assert_eq!(align_device_address(0x1001, 96), None);
    }

    #[test]
    fn translation_transform_is_converted_correctly() {
        let matrix = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));

        let result = matrix_to_tlas_transform(matrix);

        assert_eq!(
            result,
            [
                1.0, 0.0, 0.0, 1.0, // row 0
                0.0, 1.0, 0.0, 2.0, // row 1
                0.0, 0.0, 1.0, 3.0, // row 2
            ]
        );
    }
}
