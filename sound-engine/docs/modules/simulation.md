# `simulation` Module Spec

## Purpose

Run GPU acoustic propagation for query pairs (sound position, listener position) and emit arrival events.

## Public Config Type

```rust
pub struct SimulationConfig {
    pub rays_per_query: u32,
    pub max_bounces: u32,
    pub update_rate_hz: u32,
    pub max_path_seconds: f32,
}
```

## Internal Types (`pub(crate)`)

```rust
pub(crate) struct Simulator {
    cfg: SimulationConfig,
    gpu_scene_version: SceneVersion,
    gpu_scene: Option<GpuSceneResources>,
    tracer: RayTracer,
}

pub(crate) struct SimQuery {
    pub voice_id: VoiceId,
    pub source_position: [f32; 3],
    pub listener_position: [f32; 3],
    pub listener_orientation_quat: [f32; 4],
    pub scene_version: SceneVersion,
}

pub(crate) struct SimResult {
    pub voice_id: VoiceId,
    pub scene_version: SceneVersion,
    pub arrivals: Vec<ArrivalEvent>,
}

pub(crate) struct ArrivalEvent {
    pub arrival_time_seconds: f32,
    pub linear_gain: f32,
    pub incoming_direction: [f32; 3],
    pub path_order: u32,
    pub effect: EffectType,
}

pub(crate) enum EffectType {
    Reflection,
    Diffraction,
    Transmission,
}
```

## Submodule Types

```rust
pub(crate) struct RayTracer;
pub(crate) struct DiffractionPass;
pub(crate) struct TransmissionPass;
pub(crate) struct AccumulationPass;
```

## Exposed Methods (`pub(crate)`)

```rust
impl Simulator {
    pub fn new(cfg: SimulationConfig, gpu: Arc<VkBackend>) -> Result<Self, SimulationError>;

    pub fn sync_scene(
        &mut self,
        scene: &SceneStore,
        upload_plan: &SceneUploadPlan,
    ) -> Result<(), SimulationError>;

    pub fn run_queries(&mut self, queries: &[SimQuery]) -> Result<Vec<SimResult>, SimulationError>;
}
```

Submodule methods:

```rust
impl RayTracer {
    pub fn new(gpu: Arc<VkBackend>, cfg: &SimulationConfig) -> Result<Self, SimulationError>;
    pub fn trace_reflections(&mut self, queries: &[SimQuery], scene: &GpuSceneResources)
        -> Result<GpuArrivalBuffer, SimulationError>;
}

impl DiffractionPass {
    pub fn run(&mut self, queries: &[SimQuery], scene: &GpuSceneResources)
        -> Result<GpuArrivalBuffer, SimulationError>;
}

impl TransmissionPass {
    pub fn run(&mut self, queries: &[SimQuery], scene: &GpuSceneResources)
        -> Result<GpuArrivalBuffer, SimulationError>;
}

impl AccumulationPass {
    pub fn merge_and_download(
        &mut self,
        buffers: &[GpuArrivalBuffer],
    ) -> Result<Vec<SimResult>, SimulationError>;
}
```

## Invariants

- `sync_scene` must be called when scene version changes before `run_queries`.
- Each `SimResult` contains arrivals for one `voice_id`.
- Effects are merged into one arrival stream per voice for downstream IR builder.
