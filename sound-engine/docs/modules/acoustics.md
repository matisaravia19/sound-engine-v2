# `acoustics` Module Spec

## Purpose

Convert simulation arrivals into impulse responses and manage IR cache reuse.

## Types (`pub(crate)`)

```rust
pub(crate) struct IrBuilder {
    sample_rate: u32,
    ir_len_samples: usize,
}

pub(crate) struct IrSnapshot {
    pub sample_rate: u32,
    pub samples: Vec<f32>,
    pub energy: f32,
    pub scene_version: SceneVersion,
}

pub(crate) struct IrCache {
    cfg: IrCacheConfig,
    scene_version: SceneVersion,
    entries: LruCache<IrCacheKey, Arc<IrSnapshot>>,
}

pub struct IrCacheConfig {
    pub cell_size_meters: f32,
    pub reuse_distance_meters: f32,
    pub max_entries: usize,
}

pub(crate) struct IrCacheKey {
    pub scene_version: SceneVersion,
    pub source_cell: [i32; 3],
    pub listener_cell: [i32; 3],
}
```

## Exposed Methods (`pub(crate)`)

```rust
impl IrBuilder {
    pub fn new(sample_rate: u32, ir_len_samples: usize) -> Self;
    pub fn build_from_arrivals(
        &self,
        scene_version: SceneVersion,
        arrivals: &[ArrivalEvent],
    ) -> IrSnapshot;
}

impl IrCache {
    pub fn new(cfg: IrCacheConfig) -> Self;
    pub fn invalidate_for_scene(&mut self, scene_version: SceneVersion);

    pub fn key_for(
        &self,
        scene_version: SceneVersion,
        source_position: [f32; 3],
        listener_position: [f32; 3],
    ) -> IrCacheKey;

    pub fn get_reusable(
        &mut self,
        key: &IrCacheKey,
        source_position: [f32; 3],
        listener_position: [f32; 3],
    ) -> Option<Arc<IrSnapshot>>;

    pub fn insert(&mut self, key: IrCacheKey, ir: Arc<IrSnapshot>);
}
```

## Private Helper Methods

```rust
impl IrBuilder {
    fn accumulate_events(&self, arrivals: &[ArrivalEvent], bins: &mut [f32]);
    fn apply_envelope_smoothing(&self, bins: &mut [f32]);
}

impl IrCache {
    fn quantize(&self, p: [f32; 3]) -> [i32; 3];
    fn within_reuse_distance(&self, a: [f32; 3], b: [f32; 3]) -> bool;
}
```

## Invariants

- `IrSnapshot.samples.len() == ir_len_samples`.
- Cache entries are always scene-version scoped.
- Reuse must honor `reuse_distance_meters` to avoid wrong IR artifacts.
