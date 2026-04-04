# Public API Design

## 1. API Goals

- Keep integration minimal for thesis experiments and simple demos.
- Hide Vulkan and GPU details from the user-facing API.
- Favor a small set of stable calls over a feature-rich interface.

## 2. Top-Level Types

```rust
pub struct SoundEngine;
pub struct EngineHandle;
pub struct AudioRenderer;

pub struct EngineConfig;
pub struct SceneHandle;
pub struct SoundId(pub u32);
```

## 3. Lifecycle API

```rust
impl SoundEngine {
    pub fn new(config: EngineConfig) -> Result<Self, EngineError>;
    pub fn split(self) -> (EngineHandle, AudioRenderer);
}

impl EngineHandle {
    pub fn load_scene(&self, scene: SceneDescription) -> Result<SceneHandle, EngineError>;
    pub fn update_scene(&self, updates: SceneUpdates) -> Result<(), EngineError>;
    pub fn set_listener(&self, listener: ListenerState) -> Result<(), EngineError>;
    pub fn play_sound(&self, event: PlaySoundEvent) -> Result<(), EngineError>;
    pub fn tick_simulation(&self, dt_seconds: f32) -> Result<(), EngineError>;
}

impl AudioRenderer {
    pub fn process_block(
        &mut self, output: &mut [f32],
    ) -> Result<(), AudioError>;
}
```

## 4. Scene API

Scene data is intentionally simple:

```rust
pub struct SceneDescription {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
}

pub struct SceneUpdates {
    pub material_updates: Vec<MaterialUpdate>,
    pub topology_changes: Vec<TopologyChange>,
}
```

Rules:

- No mesh instance layer in the thesis version.
- Each mesh directly references a material id.
- Scene updates are coarse-grained and can trigger partial or full GPU rebuild.

## 5. Dynamic Source Model

```rust
pub struct PlaySoundEvent {
    pub sound: SoundId,
    pub position: [f32; 3],
    pub gain_db: f32,
    pub spatial_radius: f32,
}

pub struct ListenerState {
    pub position: [f32; 3],
    pub orientation_quat: [f32; 4],
}
```

Behavior:

- The host does not manage persistent acoustic source objects.
- The host issues `play_sound` events with world position.
- The engine manages active voices internally.

## 6. IR Cache Policy

The engine keeps an internal IR cache keyed by quantized source/listener positions.

Suggested key:

- `cell(source_pos, cache_cell_size)` and `cell(listener_pos, cache_cell_size)`.
- Optional direction bucket for listener orientation.

Cache behavior:

- Reuse IR when both source and listener remain within reuse tolerance.
- Recompute when movement exceeds tolerance or scene version changes.
- Crossfade old/new IR to avoid audio artifacts.

## 7. Configuration API (Minimal)

`EngineConfig` includes:

- Audio config: sample rate, block size, output format.
- Simulation config: rays per source, max bounces, update rate.
- Feature toggles: diffraction, transmission.
- IR cache config: cell size, reuse distance, max entries.

## 8. Error Contract

- Keep typed errors (`EngineError`, `AudioError`, `GpuError`, `SceneError`).
- No large diagnostics surface in initial thesis scope.
- On recoverable simulation failures, output should degrade gracefully instead of panicking.
