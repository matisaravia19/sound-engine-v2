use crate::gpu::backend::VkDeviceContext;
use crate::gpu::compute::ComputeContext;
use crate::gpu::descriptor::DescriptorPools;
use crate::gpu::memory::{BufferHandle, GpuAllocator};
use crate::gpu::resource::PipelineResources;
use crate::gpu::shader::ShaderLibrary;
use ash::vk;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

mod acceleration;
mod dispatch;
mod pipeline;
mod sbt;
mod scene;

use sbt::ShaderBindingTable;

pub use acceleration::{AabbBlasBuildSpec, BlasBuildSpec, BlasId, TlasBuildSpec, TlasId};
pub use dispatch::{RtDescriptorWrite, RtTraceSpec};
pub use pipeline::{
    RtDescriptorBindingSpec, RtPipelineId, RtPipelineSpec, RtPushConstantSpec, RtShaderGroupSpec,
};
pub use scene::{RtAabbBuffers, RtAabbSpec, RtInstanceSpec, RtMeshBuffers, RtMeshSpec};

/// Owns Vulkan ray tracing resources for the backend.
///
/// Acceleration structures and RT pipelines are stored by small IDs so callers
/// can build scenes and record traces without managing raw Vulkan lifetimes.
pub struct RtContext {
    next_blas_id: AtomicU32,
    next_tlas_id: AtomicU32,
    next_pipeline_id: AtomicU32,
    device_context: Arc<VkDeviceContext>,
    memory: Arc<GpuAllocator>,
    compute: Arc<ComputeContext>,
    shaders: Arc<ShaderLibrary>,
    blas: Mutex<HashMap<BlasId, Arc<AccelerationStructureHandle>>>,
    tlas: Mutex<HashMap<TlasId, Arc<AccelerationStructureHandle>>>,
    pipelines: Mutex<HashMap<RtPipelineId, RayTracingPipeline>>,
}

struct AccelerationStructureHandle {
    handle: vk::AccelerationStructureKHR,
    device_address: vk::DeviceAddress,
    _storage: BufferHandle,
    // TLAS owners retain each distinct referenced BLAS; BLAS records have no children.
    _referenced_blas: Vec<Arc<AccelerationStructureHandle>>,
    device_context: Arc<VkDeviceContext>,
}

struct RayTracingPipeline {
    resources: PipelineResources,
    descriptor_bindings: Vec<vk::DescriptorSetLayoutBinding<'static>>,
    push_constant_ranges: Vec<vk::PushConstantRange>,
    descriptor_pools: DescriptorPools,
    sbt: ShaderBindingTable,
}

impl RtContext {
    /// Creates an RT context sharing backend-owned device, memory, shader, and queue state.
    pub(super) fn new(
        device_context: Arc<VkDeviceContext>,
        memory: Arc<GpuAllocator>,
        shaders: Arc<ShaderLibrary>,
        compute: Arc<ComputeContext>,
    ) -> Self {
        Self {
            next_blas_id: AtomicU32::new(0),
            next_tlas_id: AtomicU32::new(0),
            next_pipeline_id: AtomicU32::new(0),
            device_context,
            memory,
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
        let _ = self.compute.wait_for_all();

        self.pipelines
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clear();
        self.tlas
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clear();
        self.blas
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clear();
    }
}

/// Releases an acoustic pipeline and its SBT after its synchronous owner finishes.
pub(crate) struct OwnedRtPipeline {
    /// Pipeline ID referenced by traces from this owner.
    pub(crate) id: RtPipelineId,
    context: Arc<RtContext>,
}

impl Drop for OwnedRtPipeline {
    fn drop(&mut self) {
        self.context
            .pipelines
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remove(&self.id);
    }
}

/// Removes a scene BLAS from the registry when its owner is replaced.
/// Existing TLASes retain its storage until their own references are released.
pub(crate) struct OwnedBlas {
    /// Registry identifier used by TLAS instances.
    pub(crate) id: BlasId,
    context: Arc<RtContext>,
}

/// Releases a scene or query TLAS on success, error, and scene replacement.
/// All referencing submissions must finish before this lease is dropped.
pub(crate) struct OwnedTlas {
    /// Registry identifier used by trace descriptors.
    pub(crate) id: TlasId,
    context: Arc<RtContext>,
}

impl RtContext {
    /// Transfers a private, synchronously used pipeline to a scoped owner.
    pub(crate) fn own_pipeline(self: &Arc<Self>, id: RtPipelineId) -> OwnedRtPipeline {
        OwnedRtPipeline {
            id,
            context: self.clone(),
        }
    }

    /// Transfers responsibility for releasing a newly built BLAS to a lease.
    pub(crate) fn own_blas(self: &Arc<Self>, id: BlasId) -> OwnedBlas {
        OwnedBlas {
            id,
            context: self.clone(),
        }
    }

    /// Transfers responsibility for releasing a newly built TLAS to a lease.
    pub(crate) fn own_tlas(self: &Arc<Self>, id: TlasId) -> OwnedTlas {
        OwnedTlas {
            id,
            context: self.clone(),
        }
    }
}

impl Drop for OwnedBlas {
    fn drop(&mut self) {
        self.context
            .blas
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remove(&self.id);
    }
}

impl Drop for OwnedTlas {
    fn drop(&mut self) {
        self.context
            .tlas
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remove(&self.id);
    }
}

impl Drop for AccelerationStructureHandle {
    fn drop(&mut self) {
        unsafe {
            self.device_context
                .acceleration_structure
                .destroy_acceleration_structure(self.handle, None);
        }
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/gpu/rt/mod.rs"]
mod tests;
