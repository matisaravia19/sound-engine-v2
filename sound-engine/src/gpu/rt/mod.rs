use crate::gpu::backend::VkDeviceContext;
use crate::gpu::compute::ComputeContext;
use crate::gpu::memory::BufferHandle;
use crate::gpu::shader::ShaderLibrary;
use ash::vk;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

mod acceleration;
mod descriptor;
mod dispatch;
mod pipeline;
mod scene;

pub use acceleration::{BlasBuildSpec, BlasId, TlasBuildSpec, TlasId};
pub use dispatch::{RtDescriptorWrite, RtTraceSpec};
pub use pipeline::{
    RtDescriptorBindingSpec, RtPipelineId, RtPipelineSpec, RtPushConstantSpec, RtShaderGroupSpec, RtShaderStageSpec,
};
pub use scene::{RtInstanceSpec, RtMeshBuffers, RtMeshSpec};

const DESCRIPTOR_SETS_PER_POOL: u32 = 64;

/// Owns Vulkan ray tracing resources for the backend.
///
/// Acceleration structures and RT pipelines are stored by small IDs so callers
/// can build scenes and record traces without managing raw Vulkan lifetimes.
pub struct RtContext {
    device_context: Arc<VkDeviceContext>,
    compute: Arc<ComputeContext>,
    shaders: Arc<ShaderLibrary>,
    blas: Mutex<HashMap<BlasId, AccelerationStructureHandle>>,
    tlas: Mutex<HashMap<TlasId, AccelerationStructureHandle>>,
    pipelines: Mutex<HashMap<RtPipelineId, RayTracingPipeline>>,
}

struct AccelerationStructureHandle {
    handle: vk::AccelerationStructureKHR,
    device_address: vk::DeviceAddress,
    storage: BufferHandle,
    device_context: Arc<VkDeviceContext>,
}

struct RayTracingPipeline {
    handle: vk::Pipeline,
    layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    push_constant_ranges: Vec<vk::PushConstantRange>,
    pool_sizes_template: Vec<vk::DescriptorPoolSize>,
    descriptor_pools: Vec<vk::DescriptorPool>,
    sbt: BufferHandle,
    raygen_region: vk::StridedDeviceAddressRegionKHR,
    miss_region: vk::StridedDeviceAddressRegionKHR,
    hit_region: vk::StridedDeviceAddressRegionKHR,
    callable_region: vk::StridedDeviceAddressRegionKHR,
}

impl RtContext {
    pub(super) fn new(
        device_context: Arc<VkDeviceContext>,
        shaders: Arc<ShaderLibrary>,
        compute: Arc<ComputeContext>,
    ) -> Self {
        Self {
            device_context,
            compute,
            shaders,
            blas: Mutex::new(HashMap::new()),
            tlas: Mutex::new(HashMap::new()),
            pipelines: Mutex::new(HashMap::new()),
        }
    }
}

impl Drop for RtContext {
    fn drop(&mut self) {
        self.compute
            .wait_for_all()
            .expect("Failed to wait for RT compute work during drop");

        unsafe {
            if let Ok(mut pipelines) = self.pipelines.lock() {
                for (_, pipeline) in pipelines.drain() {
                    for pool in &pipeline.descriptor_pools {
                        self.device_context.device.destroy_descriptor_pool(*pool, None);
                    }
                    self.device_context.device.destroy_pipeline(pipeline.handle, None);
                    self.device_context
                        .device
                        .destroy_pipeline_layout(pipeline.layout, None);
                    self.device_context
                        .device
                        .destroy_descriptor_set_layout(pipeline.descriptor_set_layout, None);
                    let _ = pipeline.sbt.size;
                }
            }

            if let Ok(mut tlas) = self.tlas.lock() {
                tlas.clear();
            }

            if let Ok(mut blas) = self.blas.lock() {
                blas.clear();
            }
        }
    }
}

impl Drop for AccelerationStructureHandle {
    fn drop(&mut self) {
        let _ = self.storage.size;
        unsafe {
            self.device_context
                .acceleration_structure
                .destroy_acceleration_structure(self.handle, None);
        }
    }
}

fn align_up(value: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two());
    (value + alignment - 1) & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::align_up;

    #[test]
    fn align_up_keeps_aligned_values() {
        assert_eq!(align_up(64, 32), 64);
    }

    #[test]
    fn align_up_rounds_to_next_boundary() {
        assert_eq!(align_up(65, 32), 96);
    }
}
