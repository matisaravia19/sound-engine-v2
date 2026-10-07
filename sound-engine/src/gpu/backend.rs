//! Assembles GPU subsystems around one shared Vulkan device and compute queue.

mod device;

use crate::core::error::SoundResult;
use crate::gpu::compute::ComputeContext;
use crate::gpu::memory::GpuAllocator;
use crate::gpu::rt::RtContext;
use crate::gpu::shader::ShaderLibrary;
use std::sync::Arc;

pub use device::{RayTracingProperties, VkDeviceContext, VkDeviceHandles};

/// Owns the Vulkan device and the GPU subsystems built on top of it.
///
/// The backend selects a ray-tracing-capable device at construction time and
/// tears down child contexts before the device is destroyed.
pub struct VkBackend {
    memory: Arc<GpuAllocator>,
    shaders: Arc<ShaderLibrary>,
    compute: Arc<ComputeContext>,
    rt: Arc<RtContext>,
    device: Arc<VkDeviceContext>,
}

impl VkBackend {
    /// Creates a Vulkan backend with compute, shader, and RT support.
    ///
    /// Device selection requires KHR acceleration structures, ray tracing
    /// pipelines, deferred host operations, and buffer device addresses.
    pub fn new() -> SoundResult<Self> {
        let (device, compute_family) = VkDeviceContext::new()?;
        let device = Arc::new(device);
        let shaders = Arc::new(ShaderLibrary::new(device.clone()));

        let compute = Arc::new(ComputeContext::new(
            device.clone(),
            shaders.clone(),
            compute_family,
        )?);

        let memory = Arc::new(GpuAllocator::new(device.clone(), compute.clone())?);

        Ok(Self {
            memory: memory.clone(),
            shaders: shaders.clone(),
            compute: compute.clone(),
            rt: Arc::new(RtContext::new(device.clone(), memory, shaders, compute)),
            device,
        })
    }

    /// Returns the shared Vulkan device context.
    pub fn device(&self) -> &VkDeviceContext {
        self.device.as_ref()
    }

    /// Clones the shared device context for long-lived GPU resources.
    pub fn device_context(&self) -> Arc<VkDeviceContext> {
        self.device.clone()
    }

    /// Returns the GPU allocator and synchronous transfer helper.
    pub fn memory(&self) -> &GpuAllocator {
        self.memory.as_ref()
    }

    /// Returns the shader library used to compile and own shader modules.
    pub fn shaders(&self) -> &ShaderLibrary {
        &self.shaders
    }

    /// Returns the compute context used for pipeline dispatch and submission.
    pub fn compute(&self) -> &ComputeContext {
        self.compute.as_ref()
    }

    /// Returns the ray tracing context used for AS builds and RT pipelines.
    pub fn rt(&self) -> &RtContext {
        &self.rt
    }

    /// Keeps the RT registry alive for scene-owned acceleration structures.
    pub(crate) fn rt_context(&self) -> Arc<RtContext> {
        self.rt.clone()
    }
}
