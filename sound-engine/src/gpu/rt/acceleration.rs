use super::*;
use crate::error::{SoundError, SoundResult};
use crate::gpu::memory::GpuAllocator;
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
    pub fn build_blas(&self, memory: &GpuAllocator, spec: BlasBuildSpec<'_>) -> SoundResult<BlasId> {
        // Geometry build inputs reference buffers by device address, not descriptors.
        let vertex_address = memory.buffer_device_address(&spec.mesh.vertex_buffer);
        let mut triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .vertex_data(vk::DeviceOrHostAddressConstKHR {
                device_address: vertex_address,
            })
            .vertex_stride(size_of::<[f32; 3]>() as vk::DeviceSize)
            .max_vertex(spec.mesh.vertex_count.saturating_sub(1));

        let primitive_count = if let Some(index_buffer) = &spec.mesh.index_buffer {
            triangles = triangles
                .index_type(vk::IndexType::UINT32)
                .index_data(vk::DeviceOrHostAddressConstKHR {
                    device_address: memory.buffer_device_address(index_buffer),
                });
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

        // BLAS v1 supports triangle geometry only.
        let mut geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .geometry(vk::AccelerationStructureGeometryDataKHR { triangles });
        if spec.mesh.opaque {
            geometry = geometry.flags(vk::GeometryFlagsKHR::OPAQUE);
        }

        let range = vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(primitive_count);
        let handle = self.build_acceleration_structure(
            memory,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            spec.flags,
            std::slice::from_ref(&geometry),
            std::slice::from_ref(&primitive_count),
            std::slice::from_ref(&range),
        )?;

        let mut blas = self
            .blas
            .lock()
            .map_err(|_| SoundError::poisoned_lock("RT BLAS lock is poisoned"))?;
        let id = BlasId(blas.len() as u32);
        blas.insert(id, handle);
        Ok(id)
    }

    /// Builds a top-level acceleration structure from BLAS instances.
    ///
    /// The instance buffer is uploaded as host-visible device-address data and
    /// the build completes before the returned `TlasId` can be used for tracing.
    pub fn build_tlas(&self, memory: &GpuAllocator, spec: TlasBuildSpec<'_>) -> SoundResult<TlasId> {
        if spec.instances.is_empty() {
            return Err(SoundError::invalid_argument(
                "RT TLAS must contain at least one instance",
            ));
        }

        // Resolve BLAS IDs while holding the lock, then drop it before GPU work.
        let instances = {
            let blas = self
                .blas
                .lock()
                .map_err(|_| SoundError::poisoned_lock("RT BLAS lock is poisoned"))?;

            spec.instances
                .iter()
                .map(|instance| {
                    let blas = blas
                        .get(&instance.blas_id)
                        .ok_or_else(|| SoundError::not_found(format!("Invalid BLAS id {}", instance.blas_id.0)))?;

                    Ok(vk::AccelerationStructureInstanceKHR {
                        transform: vk::TransformMatrixKHR {
                            matrix: instance.transform,
                        },
                        instance_custom_index_and_mask: vk::Packed24_8::new(instance.custom_index, instance.mask),
                        instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                            instance.sbt_record_offset,
                            0,
                        ),
                        acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                            device_handle: blas.device_address,
                        },
                    })
                })
                .collect::<SoundResult<Vec<_>>>()?
        };

        // Vulkan consumes TLAS instances through a device address.
        let instance_buffer = memory.create_host_device_address_buffer(
            std::mem::size_of_val(instances.as_slice()) as vk::DeviceSize,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        )?;
        memory.write_mapped_bytes(&instance_buffer, bytes_of_slice(instances.as_slice()))?;
        let instance_address = memory.buffer_device_address(&instance_buffer);

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
        let primitive_count = instances.len() as u32;
        let range = vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(primitive_count);

        let handle = self.build_acceleration_structure(
            memory,
            vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            spec.flags,
            std::slice::from_ref(&geometry),
            std::slice::from_ref(&primitive_count),
            std::slice::from_ref(&range),
        )?;

        let mut tlas = self
            .tlas
            .lock()
            .map_err(|_| SoundError::poisoned_lock("RT TLAS lock is poisoned"))?;
        let id = TlasId(tlas.len() as u32);
        tlas.insert(id, handle);
        Ok(id)
    }

    fn build_acceleration_structure(
        &self,
        memory: &GpuAllocator,
        ty: vk::AccelerationStructureTypeKHR,
        flags: vk::BuildAccelerationStructureFlagsKHR,
        geometries: &[vk::AccelerationStructureGeometryKHR<'_>],
        primitive_counts: &[u32],
        ranges: &[vk::AccelerationStructureBuildRangeInfoKHR],
    ) -> SoundResult<AccelerationStructureHandle> {
        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(ty)
            .flags(flags)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(geometries);

        // Query Vulkan for AS storage and temporary scratch requirements.
        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            self.device_context
                .acceleration_structure
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &build_info,
                    primitive_counts,
                    &mut size_info,
                );
        }

        // The AS object is backed by a buffer owned by the returned handle.
        let storage = memory.create_device_address_buffer(
            size_info.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR,
        )?;
        let create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(storage.buffer)
            .size(size_info.acceleration_structure_size)
            .ty(ty);
        let handle = unsafe {
            self.device_context
                .acceleration_structure
                .create_acceleration_structure(&create_info, None)?
        };

        // Scratch is temporary; the synchronous build below finishes before it drops.
        let scratch =
            memory.create_device_address_buffer(size_info.build_scratch_size, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        let build_info = build_info
            .dst_acceleration_structure(handle)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: memory.buffer_device_address(&scratch),
            });

        self.compute.submit_compute_and_wait(|command_buffer| unsafe {
            self.device_context
                .acceleration_structure
                .cmd_build_acceleration_structures(command_buffer, std::slice::from_ref(&build_info), &[ranges]);

            // Make AS writes visible to later AS reads and RT shader traversal.
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
                .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR);
            self.device_context.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
                    | vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );
            Ok(())
        })?;

        let address_info = vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(handle);
        let device_address = unsafe {
            self.device_context
                .acceleration_structure
                .get_acceleration_structure_device_address(&address_info)
        };

        Ok(AccelerationStructureHandle {
            handle,
            device_address,
            storage,
            device_context: self.device_context.clone(),
        })
    }
}

fn bytes_of_slice<T>(slice: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), std::mem::size_of_val(slice)) }
}
