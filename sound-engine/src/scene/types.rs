use glam::{Mat4, Vec3};

/// Stable identifier for a triangle mesh asset.
pub type MeshId = u32;

/// Stable identifier for acoustic material parameters.
pub type MaterialId = u32;

/// Stable identifier for an object instance in the scene.
pub type ObjectId = u32;

/// Stable identifier for a manually authored diffraction edge.
pub type DiffractionEdgeId = u32;

/// Monotonic scene revision used to decide when GPU resources are stale.
pub type SceneVersion = u64;

/// CPU-side triangle mesh shared by one or more scene objects.
#[derive(Debug, Clone)]
pub struct MeshAsset {
    /// Application-provided mesh identifier.
    pub id: MeshId,
    /// Vertex positions in mesh-local space.
    pub vertices: Vec<Vec3>,
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
    /// Broadband absorbed energy fraction in `[0, 1]`; the remainder is reflected.
    pub absorption: f32,
    /// Authored rough-reflection energy fraction in `[0, 1]`. Scalar transport
    /// preserves the relative lobe amplitudes but normalizes their pressure sum;
    /// it does not model independent diffuse phases or exact directional energy.
    pub scattering: f32,
    /// Width control for finite glossy reflection lobes in `[0, 1]`.
    pub roughness: f32,
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
    /// Object-to-world transform used for scene placement and TLAS instances.
    pub transform: Mat4,
    /// Inactive objects stay in CPU state but are excluded from the TLAS.
    pub active: bool,
}

/// Manually authored geometry that can diffract acoustic rays.
/// Remove an edge from the scene to disable its diffraction contribution.
#[derive(Debug, Clone)]
pub struct DiffractionEdge {
    /// Application-provided diffraction edge identifier.
    pub id: DiffractionEdgeId,
    /// First endpoint of the finite edge segment in world meters.
    pub start: Vec3,
    /// Second endpoint of the finite edge segment in world meters.
    pub end: Vec3,
    /// Direction perpendicular to the edge bisecting the exterior air wedge.
    /// For a thin screen, points away from the screen into its open continuation.
    pub bisector_dir: Vec3,
    /// Exterior air angle in radians: 3*PI/2 for a solid right-angle corner,
    /// TAU for a zero-thickness screen. Must match the physical mesh faces.
    pub edge_angle_radians: f32,
    /// Width of the exterior ray-traced virtual edge strip in meters.
    ///
    /// The negative half of the bounding box lies in the solid wedge and is
    /// rejected. The triangular kernel integrates to one over [0, width].
    /// Accepted rays advance to the physical edge before continuing; this width
    /// controls discovery and sampling variance, not the diffraction vertex.
    pub diffraction_radius: f32,
}

/// Complete scene replacement payload.
#[derive(Debug, Default, Clone)]
pub struct SceneDescription {
    /// Mesh assets available to scene objects.
    pub meshes: Vec<MeshAsset>,
    /// Material table available to scene objects.
    pub materials: Vec<Material>,
    /// Object instances placed in the scene.
    pub objects: Vec<SceneObject>,
    /// Manually authored diffraction edges placed in world space.
    pub diffraction_edges: Vec<DiffractionEdge>,
}

/// Incremental scene changes applied to the canonical CPU store.
#[derive(Debug, Default, Clone)]
pub struct SceneUpdates {
    /// Ordered updates applied atomically to the scene store.
    pub updates: Vec<SceneUpdate>,
}

/// One ordered scene mutation.
#[derive(Debug, Clone)]
pub enum SceneUpdate {
    /// Insert a new mesh asset.
    AddMesh(MeshAsset),
    /// Replace an existing mesh asset and rebuild its GPU geometry.
    ReplaceMesh(MeshAsset),
    /// Remove an unreferenced mesh asset.
    RemoveMesh(MeshId),
    /// Insert a new material.
    AddMaterial(Material),
    /// Replace an existing material.
    ReplaceMaterial(Material),
    /// Remove a material that is not referenced by any object.
    RemoveMaterial(MaterialId),
    /// Insert a new object instance.
    AddObject(SceneObject),
    /// Replace an existing object instance.
    ReplaceObject(SceneObject),
    /// Remove an object instance.
    RemoveObject(ObjectId),
    /// Insert a new manually authored diffraction edge.
    AddDiffractionEdge(DiffractionEdge),
    /// Replace an existing manually authored diffraction edge.
    ReplaceDiffractionEdge(DiffractionEdge),
    /// Remove a manually authored diffraction edge.
    RemoveDiffractionEdge(DiffractionEdgeId),
    /// Replace an object's world transform.
    SetTransform {
        /// Object to mutate.
        id: ObjectId,
        /// New object-to-world transform.
        transform: Mat4,
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
