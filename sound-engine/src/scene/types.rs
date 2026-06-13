/// Stable identifier for a triangle mesh asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshId(pub u32);

/// Stable identifier for acoustic material parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialId(pub u32);

/// Stable identifier for an object instance in the scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObjectId(pub u32);

/// Monotonic scene revision used to decide when GPU resources are stale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SceneVersion(pub u64);

/// CPU-side triangle mesh shared by one or more scene objects.
#[derive(Debug, Clone)]
pub struct MeshAsset {
    /// Application-provided mesh identifier.
    pub id: MeshId,
    /// Vertex positions as tightly packed `vec3<f32>` values.
    pub vertices: Vec<[f32; 3]>,
    /// Optional triangle indices; empty means vertices are consumed as triples.
    pub indices: Vec<u32>,
    /// Whether RT traversal may treat the mesh as opaque.
    pub opaque: bool,
}

/// Acoustic material parameters referenced by scene objects.
#[derive(Debug, Clone)]
pub struct Material {
    /// Application-provided material identifier.
    pub id: MaterialId,
    /// Absorption coefficients, initially interpreted as broad frequency bands.
    pub absorption_bands: Vec<f32>,
    /// Scattering coefficient in `[0, 1]`.
    pub scattering: f32,
    /// Optional transmission coefficients for future through-surface paths.
    pub transmission: Option<Vec<f32>>,
}

/// One placeable object instance in world space.
#[derive(Debug, Clone)]
pub struct SceneObject {
    /// Application-provided object identifier.
    pub id: ObjectId,
    /// Mesh asset instanced by this object.
    pub mesh_id: MeshId,
    /// Acoustic material used by this object.
    pub material_id: MaterialId,
    /// Row-major 3x4 object-to-world transform expected by Vulkan TLAS instances.
    pub transform: [f32; 12],
    /// Inactive objects stay in CPU state but are excluded from the TLAS.
    pub active: bool,
}

/// Complete scene replacement payload.
#[derive(Debug, Clone)]
pub struct SceneDescription {
    /// Mesh assets available to scene objects.
    pub meshes: Vec<MeshAsset>,
    /// Material table available to scene objects.
    pub materials: Vec<Material>,
    /// Object instances placed in the scene.
    pub objects: Vec<SceneObject>,
}

/// Incremental scene changes applied to the canonical CPU store.
#[derive(Debug, Default, Clone)]
pub struct SceneUpdates {
    /// Object transform, active-state, and material changes.
    pub object_updates: Vec<ObjectUpdate>,
    /// In-place material parameter changes.
    pub material_updates: Vec<MaterialUpdate>,
    /// Mesh/object add, replace, and remove operations.
    pub topology_changes: Vec<TopologyChange>,
}

/// Object-only update that does not replace mesh topology.
#[derive(Debug, Clone)]
pub enum ObjectUpdate {
    /// Replace an object's world transform.
    SetTransform {
        /// Object to mutate.
        id: ObjectId,
        /// New row-major 3x4 object-to-world transform.
        transform: [f32; 12],
    },
    /// Include or exclude an object from future GPU TLAS builds.
    SetActive {
        /// Object to mutate.
        id: ObjectId,
        /// New active state.
        active: bool,
    },
    /// Reassign an object to another existing material.
    SetMaterial {
        /// Object to mutate.
        id: ObjectId,
        /// Existing material to use.
        material_id: MaterialId,
    },
}

/// Partial material update; `None` fields keep their previous value.
#[derive(Debug, Clone)]
pub struct MaterialUpdate {
    /// Material to mutate.
    pub id: MaterialId,
    /// Replacement absorption bands.
    pub absorption_bands: Option<Vec<f32>>,
    /// Replacement scattering coefficient.
    pub scattering: Option<f32>,
    /// Replacement transmission setting; `Some(None)` clears transmission.
    pub transmission: Option<Option<Vec<f32>>>,
}

/// Coarse topology changes that may require BLAS/TLAS rebuilds.
#[derive(Debug, Clone)]
pub enum TopologyChange {
    /// Insert a new mesh asset.
    AddMesh(MeshAsset),
    /// Replace an existing mesh asset and rebuild its GPU geometry.
    ReplaceMesh(MeshAsset),
    /// Remove an unreferenced mesh asset.
    RemoveMesh(MeshId),
    /// Insert a new object instance.
    AddObject(SceneObject),
    /// Replace an existing object instance.
    ReplaceObject(SceneObject),
    /// Remove an object instance.
    RemoveObject(ObjectId),
}

impl Default for SceneVersion {
    fn default() -> Self {
        Self(0)
    }
}
