# `scene` Module Spec

## Purpose

Maintain canonical CPU-side scene data and provide diff/preprocess output for GPU upload.

## Public Types

```rust
pub struct SceneDescription {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
}

pub struct SceneUpdates {
    pub material_updates: Vec<MaterialUpdate>,
    pub topology_changes: Vec<TopologyChange>,
}

pub struct Mesh {
    pub id: MeshId,
    pub vertices: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
    pub material: MaterialId,
}

pub struct Material {
    pub id: MaterialId,
    pub absorption_bands: Vec<f32>, // e.g. octave bands
    pub scattering: f32,
    pub transmission: Option<Vec<f32>>,
}

pub struct MaterialUpdate {
    pub id: MaterialId,
    pub absorption_bands: Option<Vec<f32>>,
    pub scattering: Option<f32>,
    pub transmission: Option<Option<Vec<f32>>>,
}

pub enum TopologyChange {
    AddMesh(Mesh),
    RemoveMesh(MeshId),
    ReplaceMesh(Mesh),
}

pub struct MeshId(pub u32);
pub struct MaterialId(pub u32);
pub struct SceneHandle(pub u64);
```

## Internal Types (`pub(crate)`)

```rust
pub(crate) struct SceneStore {
    scene_handle: SceneHandle,
    version: SceneVersion,
    meshes: Vec<Mesh>,
    materials: Vec<Material>,
}

pub(crate) struct SceneVersion(pub u64);

pub(crate) struct SceneUploadPlan {
    pub scene_version: SceneVersion,
    pub geometry_updates: GeometryUpdateKind,
    pub material_updates: Vec<MaterialUpdate>,
}

pub(crate) enum GeometryUpdateKind {
    None,
    PartialMeshSet(Vec<MeshId>),
    FullRebuild,
}

pub(crate) struct EdgeDataSet {
    pub edges: Vec<EdgeRecord>,
}

pub(crate) struct EdgeRecord {
    pub p0: [f32; 3],
    pub p1: [f32; 3],
    pub face_normal_a: [f32; 3],
    pub face_normal_b: [f32; 3],
}
```

## Exposed Methods (`pub(crate)`)

```rust
impl SceneStore {
    pub fn new() -> Self;
    pub fn load(&mut self, scene: SceneDescription) -> SceneHandle;
    pub fn apply_updates(&mut self, updates: SceneUpdates) -> Result<(), SceneError>;

    pub fn version(&self) -> SceneVersion;
    pub fn meshes(&self) -> &[Mesh];
    pub fn materials(&self) -> &[Material];

    pub fn build_upload_plan(&self, previous_version: SceneVersion) -> SceneUploadPlan;
    pub fn build_edge_dataset(&self) -> EdgeDataSet;
}
```

## Invariants

- Mesh references to `MaterialId` must be valid.
- `version` increments on any applied scene update.
- Edge dataset must be regenerated when topology changes.
