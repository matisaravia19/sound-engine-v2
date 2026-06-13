use glam::{Mat4, Vec3};

use super::*;
use crate::error::{SoundError, SoundResult};
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
    pub fn build_blas(&self, spec: BlasBuildSpec<'_>) -> SoundResult<BlasId> {
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
            triangles = triangles
                .index_type(vk::IndexType::UINT32)
                .index_data(vk::DeviceOrHostAddressConstKHR {
                    device_address: self.memory.buffer_device_address(index_buffer),
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
    pub fn build_tlas(&self, spec: TlasBuildSpec<'_>) -> SoundResult<TlasId> {
        if spec.instances.is_empty() {
            return Err(SoundError::invalid_argument(
                "RT TLAS must contain at least one instance",
            ));
        }

        // Resolve BLAS IDs while holding the lock, then drop it before GPU work.
        let instances = self.resolve_instances(spec.instances)?;

        // Vulkan consumes TLAS instances through a device address.
        let instance_buffer = self.memory.create_host_device_address_buffer(
            std::mem::size_of_val(instances.as_slice()) as vk::DeviceSize,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        )?;
        self.memory
            .write_mapped_bytes(&instance_buffer, acceleration_instances_as_bytes(instances.as_slice()))?;
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
        let primitive_count = instances.len() as u32;
        let range = vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(primitive_count);

        let handle = self.build_acceleration_structure(
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
        let storage = self.memory.create_device_address_buffer(
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
        let scratch = self
            .memory
            .create_device_address_buffer(size_info.build_scratch_size, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        let build_info = build_info
            .dst_acceleration_structure(handle)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: self.memory.buffer_device_address(&scratch),
            });

        self.compute.submit_compute_and_wait(|command_buffer| unsafe {
            self.device_context
                .acceleration_structure
                .cmd_build_acceleration_structures(command_buffer, std::slice::from_ref(&build_info), &[ranges]);

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

    fn resolve_instances(&self, specs: &[RtInstanceSpec]) -> SoundResult<Vec<vk::AccelerationStructureInstanceKHR>> {
        let blas = self
            .blas
            .lock()
            .map_err(|_| SoundError::poisoned_lock("RT BLAS lock is poisoned"))?;

        specs
            .iter()
            .map(|instance| {
                let blas_handle = blas
                    .get(&instance.blas_id)
                    .ok_or_else(|| SoundError::not_found(format!("Invalid BLAS id {}", instance.blas_id.0)))?;

                Ok(vk::AccelerationStructureInstanceKHR {
                    transform: vk::TransformMatrixKHR {
                        matrix: matrix_to_tlas_transform(instance.transform),
                    },
                    instance_custom_index_and_mask: vk::Packed24_8::new(instance.custom_index, instance.mask),
                    instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                        instance.sbt_record_offset,
                        0,
                    ),
                    acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                        device_handle: blas_handle.device_address,
                    },
                })
            })
            .collect::<SoundResult<Vec<_>>>()
    }
}

fn acceleration_instances_as_bytes(instances: &[vk::AccelerationStructureInstanceKHR]) -> &[u8] {
    // SAFETY:
    // - Vulkan defines VkAccelerationStructureInstanceKHR as the exact byte layout
    //   consumed for TLAS instance builds.
    // - ash represents it as a repr(C)-compatible Vulkan FFI type.
    // - The returned byte slice is only used immediately for copying into mapped memory.
    unsafe { std::slice::from_raw_parts(instances.as_ptr().cast::<u8>(), std::mem::size_of_val(instances)) }
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
