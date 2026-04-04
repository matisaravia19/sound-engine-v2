# `api` Module Spec

## Purpose

Public entry points for host applications. Keep this minimal and stable.

## Public Types

```rust
pub struct SoundEngine;
pub struct EngineHandle;
pub struct AudioRenderer;

pub struct EngineConfig {
    pub sample_rate: u32,
    pub block_size: usize,
    pub output_channels: OutputChannels, // Stereo | Binaural
    pub simulation: SimulationConfig,
    pub ir_cache: IrCacheConfig,
    pub features: FeatureFlags,
}

pub enum OutputChannels { Stereo, Binaural }

pub struct FeatureFlags {
    pub diffraction: bool,
    pub transmission: bool,
}

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

pub struct SoundId(pub u32);
pub struct SceneHandle(pub u64);
```

Scene-facing types are re-exported from `scene`:

```rust
pub use crate::scene::{SceneDescription, SceneUpdates, Mesh, Material};
```

## Public Methods

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
    pub fn process_block(&mut self, output: &mut [f32]) -> Result<(), AudioError>;
}
```

## Internal `api` Types (`pub(crate)`)

```rust
pub(crate) enum ApiCommand {
    LoadScene(SceneDescription),
    UpdateScene(SceneUpdates),
    SetListener(ListenerState),
    PlaySound(PlaySoundEvent),
}
```

## Invariants

- `process_block` must be allocation-free on hot path.
- `tick_simulation` can fail; audio path must remain usable.
- `EngineHandle` is cheap clone and thread-safe (`Send + Sync`).
