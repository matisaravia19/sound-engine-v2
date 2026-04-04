# `core` Module Spec

## Purpose

Central orchestration between API commands, scene state, simulation, IR cache, and audio renderer.

## Main Types

```rust
pub(crate) struct EngineCore {
    mode: EngineMode,
    scene: SceneStore,
    simulator: Simulator,
    ir_cache: IrCache,
    ir_builder: IrBuilder,
    listener: ListenerState,
    active_sounds: ActiveSoundSet,
    published_ir: ArcSwap<IrSnapshotSet>,
}

pub(crate) enum EngineMode {
    FullGpu,
    ReducedGpu,
    Bypass,
}

pub(crate) struct ActiveSound {
    voice_id: VoiceId,
    sound: SoundId,
    position: [f32; 3],
    gain_db: f32,
    spatial_radius: f32,
}

pub(crate) type ActiveSoundSet = Vec<ActiveSound>;
```

## Exposed Methods (`pub(crate)`)

```rust
impl EngineCore {
    pub fn new(config: EngineConfig) -> Result<Self, EngineError>;

    pub fn load_scene(&mut self, scene: SceneDescription) -> Result<SceneHandle, EngineError>;
    pub fn update_scene(&mut self, updates: SceneUpdates) -> Result<(), EngineError>;
    pub fn set_listener(&mut self, listener: ListenerState);
    pub fn play_sound(&mut self, event: PlaySoundEvent) -> Result<(), EngineError>;

    pub fn tick_simulation(&mut self, dt_seconds: f32) -> Result<(), EngineError>;
    pub fn published_ir(&self) -> Arc<IrSnapshotSet>;
}
```

## Key Private Methods

```rust
impl EngineCore {
    fn process_scene_changes(&mut self) -> Result<(), EngineError>;
    fn collect_sim_queries(&self) -> Vec<SimQuery>;
    fn resolve_cache_hits(&self, queries: &[SimQuery]) -> CacheResolution;
    fn run_simulation_for_misses(&mut self, misses: &[SimQuery]) -> Result<Vec<SimResult>, EngineError>;
    fn publish_ir_set(&mut self, ir_set: IrSnapshotSet);
}
```

## Data Contract with Audio Side

`published_ir` is an immutable snapshot set:

```rust
pub(crate) struct IrSnapshotSet {
    pub version: u64,
    pub generated_at_seconds: f64,
    pub per_voice: Vec<(VoiceId, Arc<IrSnapshot>)>,
}
```

## Invariants

- Scene updates are applied before simulation queries are built.
- Cache lookup always happens before dispatching new simulation.
- Published IR sets are immutable and replaced atomically.
