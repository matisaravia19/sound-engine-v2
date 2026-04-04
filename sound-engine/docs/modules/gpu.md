# `gpu` Module Spec (Vulkan Wrapper)

## Purpose

Provide a small, explicit Vulkan abstraction for simulation and DSP workloads.

## Submodules

```text
gpu/
  backend/   // instance/device/features
  memory/    // allocator, buffers, staging
  shader/    // shader modules and pipeline cache
  rt/        // BLAS/TLAS and ray tracing pipelines
  sync/      // fences/semaphores and frame completion
```

## `backend` Types and Methods

```rust
pub(crate) struct VkBackend {
    device: Arc<VkDeviceContext>,
    memory: GpuAllocator,
    shader_lib: ShaderLibrary,
    sync: SyncContext,
}

pub(crate) struct VkDeviceContext {
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queues: QueueSet,
    pub caps: DeviceCaps,
}

pub(crate) struct QueueSet {
    pub compute_queue: vk::Queue,
    pub compute_family: u32,
    pub transfer_queue: Option<vk::Queue>,
    pub transfer_family: Option<u32>,
}

pub(crate) struct DeviceCaps {
    pub ray_tracing: bool,
    pub accel_struct: bool,
    pub descriptor_indexing: bool,
    pub sync2: bool,
}
```

```rust
impl VkBackend {
    pub fn new(config: &EngineConfig) -> Result<Self, GpuError>;
    pub fn caps(&self) -> &DeviceCaps;
    pub fn device(&self) -> &VkDeviceContext;
    pub fn memory(&self) -> &GpuAllocator;
    pub fn memory_mut(&mut self) -> &mut GpuAllocator;
    pub fn sync(&self) -> &SyncContext;
}
```

## `memory` Types and Methods

```rust
pub(crate) struct GpuAllocator;

pub(crate) struct BufferHandle {
    pub buffer: vk::Buffer,
    pub size: vk::DeviceSize,
}

pub(crate) struct StagingRing;

pub(crate) struct UploadSlice {
    pub staging_offset: vk::DeviceSize,
    pub size: vk::DeviceSize,
}
```

```rust
impl GpuAllocator {
    pub fn create_storage_buffer(&mut self, size: vk::DeviceSize) -> Result<BufferHandle, GpuError>;
    pub fn create_readback_buffer(&mut self, size: vk::DeviceSize) -> Result<BufferHandle, GpuError>;
    pub fn destroy_buffer(&mut self, handle: BufferHandle);

    pub fn upload_bytes(&mut self, dst: &BufferHandle, data: &[u8]) -> Result<(), GpuError>;
    pub fn download_bytes(&mut self, src: &BufferHandle, size: usize) -> Result<Vec<u8>, GpuError>;
}
```

## `shader` Types and Methods

```rust
pub(crate) struct ShaderLibrary;
pub(crate) struct ShaderId(pub u32);
pub(crate) struct PipelineId(pub u32);
```

```rust
impl ShaderLibrary {
    pub fn load_spirv_file(
        &mut self,
        stage: ShaderStage,
        path: &std::path::Path,
        entry: &str,
    ) -> Result<ShaderId, GpuError>;

    pub fn create_compute_pipeline(&mut self, shader: ShaderId) -> Result<PipelineId, GpuError>;
    pub fn create_rt_pipeline(&mut self, desc: RtPipelineDesc) -> Result<PipelineId, GpuError>;
}
```

## `rt` Types and Methods

```rust
pub(crate) struct RtScene {
    pub blas: Vec<BlasHandle>,
    pub tlas: TlasHandle,
}

pub(crate) struct BlasHandle;
pub(crate) struct TlasHandle;
pub(crate) struct RtPipeline;
pub(crate) struct SbtTable;
```

```rust
impl RtScene {
    pub fn build_full(
        backend: &VkBackend,
        meshes: &[Mesh],
    ) -> Result<Self, GpuError>;

    pub fn update_partial(
        &mut self,
        backend: &VkBackend,
        changed_meshes: &[MeshId],
        all_meshes: &[Mesh],
    ) -> Result<(), GpuError>;
}

impl RtPipeline {
    pub fn trace_rays(
        &self,
        backend: &VkBackend,
        scene: &RtScene,
        queries: &BufferHandle,
        arrivals_out: &BufferHandle,
        ray_count: u32,
    ) -> Result<(), GpuError>;
}
```

## `sync` Types and Methods

```rust
pub(crate) struct SyncContext;
pub(crate) struct FrameToken(pub u64);
```

```rust
impl SyncContext {
    pub fn submit_compute(
        &self,
        device: &VkDeviceContext,
        cmd: vk::CommandBuffer,
    ) -> Result<FrameToken, GpuError>;

    pub fn wait_for(&self, device: &VkDeviceContext, token: FrameToken) -> Result<(), GpuError>;
}
```

## Cross-Module Integration Methods

```rust
impl VkBackend {
    pub fn ensure_scene_resources(
        &mut self,
        scene: &SceneStore,
        plan: &SceneUploadPlan,
    ) -> Result<GpuSceneResources, GpuError>;
}

pub(crate) struct GpuSceneResources {
    pub scene_version: SceneVersion,
    pub rt_scene: RtScene,
    pub material_buffer: BufferHandle,
    pub edge_buffer: BufferHandle,
}
```

## Invariants

- All Vulkan object destruction is centralized in wrapper types.
- No raw Vulkan handles are exposed outside `gpu` except via internal wrapper structs.
- Long-lived buffers/pipelines are reused across simulation ticks.
