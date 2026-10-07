use super::types::{
    DiffractionEdge, DiffractionEdgeId, Material, MaterialId, MeshAsset, MeshId, ObjectId,
    SceneDescription, SceneObject, SceneUpdate, SceneUpdates, SceneVersion,
};
use crate::core::error::{SoundError, SoundResult};
use std::collections::HashMap;
use std::sync::Arc;

/// Canonical CPU-side scene state.
///
/// `SceneStore` owns validated mesh, material, and object records. Any accepted
/// mutation increments `SceneVersion`, which drives lazy GPU synchronization.
#[derive(Clone, Default)]
pub struct SceneStore {
    /// Revision used to invalidate scene-derived GPU data and acoustic caches.
    pub(super) version: SceneVersion,
    revisions: SceneRevisions,
    meshes: HashMap<MeshId, StoredMesh>,
    /// Validated acoustic materials addressable by scene objects.
    pub(super) materials: HashMap<MaterialId, Material>,
    /// Placed objects, including inactive instances.
    pub(super) objects: HashMap<ObjectId, SceneObject>,
    /// Authored edge geometry used by diffraction proxies.
    pub(super) diffraction_edges: HashMap<DiffractionEdgeId, DiffractionEdge>,
}

// Transactions share immutable geometry; moving one object must not copy every
// vertex/index array. Keep the content revision beside the asset it describes.
#[derive(Clone)]
struct StoredMesh {
    asset: Arc<MeshAsset>,
    revision: SceneVersion,
}

/// Last committed edits affecting independently reusable GPU resource groups.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct SceneRevisions {
    /// Mesh storage, active instance membership, ordering and transforms.
    pub(super) geometry: SceneVersion,
    /// Material values/indices and object records, including geometry addresses.
    pub(super) shading: SceneVersion,
    /// Authored edge records and their procedural acceleration structures.
    pub(super) diffraction: SceneVersion,
}

impl SceneRevisions {
    fn invalidate(&mut self, update: &SceneUpdate, revision: SceneVersion) {
        match update {
            SceneUpdate::AddMesh(_)
            | SceneUpdate::ReplaceMesh(_)
            | SceneUpdate::RemoveMesh(_)
            | SceneUpdate::AddObject(_)
            | SceneUpdate::ReplaceObject(_)
            | SceneUpdate::RemoveObject(_)
            | SceneUpdate::SetTransform {
                ..
            }
            | SceneUpdate::SetActive {
                ..
            } => {
                self.geometry = revision;
                // Object records contain geometry addresses, instance ordering
                // and transforms, so a geometry edit also invalidates shading.
                self.shading = revision;
            }
            SceneUpdate::AddMaterial(_)
            | SceneUpdate::ReplaceMaterial(_)
            | SceneUpdate::RemoveMaterial(_)
            | SceneUpdate::SetMaterial {
                ..
            } => self.shading = revision,
            SceneUpdate::AddDiffractionEdge(_)
            | SceneUpdate::ReplaceDiffractionEdge(_)
            | SceneUpdate::RemoveDiffractionEdge(_) => self.diffraction = revision,
        }
    }
}

impl SceneStore {
    /// Creates an empty scene store at version zero.
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the entire CPU scene and increments the version.
    pub fn load(&mut self, scene: SceneDescription) -> SoundResult<()> {
        let revision = self.next_version()?;
        let mut next = Self::default();
        for mesh in scene.meshes {
            validate_mesh(&mesh)?;
            if next
                .meshes
                .insert(
                    mesh.id,
                    StoredMesh {
                        asset: Arc::new(mesh),
                        revision,
                    },
                )
                .is_some()
            {
                return Err(SoundError::invalid_argument("Duplicate mesh id in scene"));
            }
        }
        for material in scene.materials {
            validate_material(&material)?;
            if next.materials.insert(material.id, material).is_some() {
                return Err(SoundError::invalid_argument(
                    "Duplicate material id in scene",
                ));
            }
        }
        for object in scene.objects {
            validate_object_refs(&next, &object)?;
            if next.objects.insert(object.id, object).is_some() {
                return Err(SoundError::invalid_argument("Duplicate object id in scene"));
            }
        }
        for edge in scene.diffraction_edges {
            validate_diffraction_edge(&edge)?;
            if next.diffraction_edges.insert(edge.id, edge).is_some() {
                return Err(SoundError::invalid_argument(
                    "Duplicate diffraction edge id in scene",
                ));
            }
        }

        next.version = revision;
        next.revisions = SceneRevisions {
            geometry: revision,
            shading: revision,
            diffraction: revision,
        };
        *self = next;
        Ok(())
    }

    /// Applies incremental changes and increments the version if anything changed.
    pub fn apply_updates(&mut self, updates: SceneUpdates) -> SoundResult<()> {
        if updates.updates.is_empty() {
            return Ok(());
        }
        let revision = self.next_version()?;
        let mut next = self.clone();

        for update in updates.updates {
            next.apply_update(update, revision)?;
        }

        next.version = revision;
        *self = next;
        Ok(())
    }

    /// Returns the current scene revision.
    pub fn version(&self) -> SceneVersion {
        self.version
    }

    /// Returns committed resource revisions; failed transactions leave them unchanged.
    pub(super) fn revisions(&self) -> SceneRevisions {
        self.revisions
    }

    /// Iterates over all mesh assets in unspecified order.
    pub fn meshes(&self) -> impl Iterator<Item = &MeshAsset> {
        self.meshes.values().map(|mesh| mesh.asset.as_ref())
    }

    /// Returns immutable geometry without exposing the transaction's shared owner.
    pub(super) fn mesh(&self, id: MeshId) -> Option<&MeshAsset> {
        self.meshes.get(&id).map(|mesh| mesh.asset.as_ref())
    }

    /// Returns the geometry revision used to decide whether a BLAS is reusable.
    pub(super) fn mesh_revision(&self, id: MeshId) -> Option<SceneVersion> {
        self.meshes.get(&id).map(|mesh| mesh.revision)
    }

    /// Iterates over all materials in unspecified order.
    pub fn materials(&self) -> impl Iterator<Item = &Material> {
        self.materials.values()
    }

    /// Iterates over all objects in unspecified order.
    pub fn objects(&self) -> impl Iterator<Item = &SceneObject> {
        self.objects.values()
    }

    /// Iterates over all manually authored diffraction edges in unspecified order.
    pub fn diffraction_edges(&self) -> impl Iterator<Item = &DiffractionEdge> {
        self.diffraction_edges.values()
    }

    fn next_version(&self) -> SoundResult<SceneVersion> {
        self.version
            .checked_add(1)
            .ok_or_else(|| SoundError::invalid_state("Scene revision space exhausted"))
    }

    fn apply_update(&mut self, update: SceneUpdate, revision: SceneVersion) -> SoundResult<()> {
        // This method runs only on the transaction's clone. Revisions are
        // published together with its records after every update succeeds.
        self.revisions.invalidate(&update, revision);
        match update {
            SceneUpdate::AddMesh(mesh) => {
                validate_mesh(&mesh)?;
                if self
                    .meshes
                    .insert(
                        mesh.id,
                        StoredMesh {
                            asset: Arc::new(mesh),
                            revision,
                        },
                    )
                    .is_some()
                {
                    return Err(SoundError::invalid_argument(
                        "AddMesh received an existing mesh id",
                    ));
                }
            }
            SceneUpdate::ReplaceMesh(mesh) => {
                validate_mesh(&mesh)?;
                if !self.meshes.contains_key(&mesh.id) {
                    return Err(SoundError::not_found(format!(
                        "Mesh {} does not exist",
                        mesh.id
                    )));
                }
                self.meshes.insert(
                    mesh.id,
                    StoredMesh {
                        asset: Arc::new(mesh),
                        revision,
                    },
                );
            }
            SceneUpdate::RemoveMesh(id) => {
                if self.objects.values().any(|object| object.mesh_id == id) {
                    return Err(SoundError::invalid_state(format!(
                        "Mesh {} is still referenced",
                        id
                    )));
                }
                self.meshes
                    .remove(&id)
                    .ok_or_else(|| SoundError::not_found(format!("Mesh {} does not exist", id)))?;
            }
            SceneUpdate::AddMaterial(material) => {
                validate_material(&material)?;
                if self.materials.insert(material.id, material).is_some() {
                    return Err(SoundError::invalid_argument(
                        "AddMaterial received an existing material id",
                    ));
                }
            }
            SceneUpdate::ReplaceMaterial(material) => {
                validate_material(&material)?;
                if !self.materials.contains_key(&material.id) {
                    return Err(SoundError::not_found(format!(
                        "Material {} does not exist",
                        material.id
                    )));
                }
                self.materials.insert(material.id, material);
            }
            SceneUpdate::RemoveMaterial(id) => {
                if self.objects.values().any(|object| object.material_id == id) {
                    return Err(SoundError::invalid_state(format!(
                        "Material {} is still referenced",
                        id
                    )));
                }
                self.materials.remove(&id).ok_or_else(|| {
                    SoundError::not_found(format!("Material {} does not exist", id))
                })?;
            }
            SceneUpdate::AddObject(object) => {
                validate_object_refs(self, &object)?;
                if self.objects.insert(object.id, object).is_some() {
                    return Err(SoundError::invalid_argument(
                        "AddObject received an existing object id",
                    ));
                }
            }
            SceneUpdate::ReplaceObject(object) => {
                validate_object_refs(self, &object)?;
                if !self.objects.contains_key(&object.id) {
                    return Err(SoundError::not_found(format!(
                        "Object {} does not exist",
                        object.id
                    )));
                }
                self.objects.insert(object.id, object);
            }
            SceneUpdate::RemoveObject(id) => {
                self.objects.remove(&id).ok_or_else(|| {
                    SoundError::not_found(format!("Object {} does not exist", id))
                })?;
            }
            SceneUpdate::AddDiffractionEdge(edge) => {
                validate_diffraction_edge(&edge)?;
                if self.diffraction_edges.insert(edge.id, edge).is_some() {
                    return Err(SoundError::invalid_argument(
                        "AddDiffractionEdge received an existing edge id",
                    ));
                }
            }
            SceneUpdate::ReplaceDiffractionEdge(edge) => {
                validate_diffraction_edge(&edge)?;
                if !self.diffraction_edges.contains_key(&edge.id) {
                    return Err(SoundError::not_found(format!(
                        "Diffraction edge {} does not exist",
                        edge.id
                    )));
                }
                self.diffraction_edges.insert(edge.id, edge);
            }
            SceneUpdate::RemoveDiffractionEdge(id) => {
                self.diffraction_edges.remove(&id).ok_or_else(|| {
                    SoundError::not_found(format!("Diffraction edge {} does not exist", id))
                })?;
            }
            SceneUpdate::SetTransform {
                id,
                transform,
            } => {
                crate::gpu::rt::validate_instance_transform(transform)?;
                self.object_mut(id)?.transform = transform;
            }
            SceneUpdate::SetActive {
                id,
                active,
            } => {
                self.object_mut(id)?.active = active;
            }
            SceneUpdate::SetMaterial {
                id,
                material_id,
            } => {
                if !self.materials.contains_key(&material_id) {
                    return Err(SoundError::not_found(format!(
                        "Material {} does not exist",
                        material_id
                    )));
                }
                self.object_mut(id)?.material_id = material_id;
            }
        }
        Ok(())
    }

    fn object_mut(&mut self, id: ObjectId) -> SoundResult<&mut SceneObject> {
        self.objects
            .get_mut(&id)
            .ok_or_else(|| SoundError::not_found(format!("Object {} does not exist", id)))
    }
}

fn validate_mesh(mesh: &MeshAsset) -> SoundResult<()> {
    crate::gpu::rt::validate_triangle_mesh(
        &mesh.vertices,
        (!mesh.indices.is_empty()).then_some(mesh.indices.as_slice()),
    )?;
    if mesh.vertices.is_empty() {
        return Err(SoundError::invalid_argument(format!(
            "Mesh {} has no vertices",
            mesh.id
        )));
    }
    if mesh.indices.is_empty() && mesh.vertices.len() < 3 {
        return Err(SoundError::invalid_argument(format!(
            "Mesh {} has less than one triangle",
            mesh.id
        )));
    }
    if !mesh.indices.is_empty() && mesh.indices.len() < 3 {
        return Err(SoundError::invalid_argument(format!(
            "Mesh {} has less than one indexed triangle",
            mesh.id
        )));
    }
    Ok(())
}

fn validate_material(material: &Material) -> SoundResult<()> {
    for (name, value) in [
        ("absorption", material.absorption),
        ("scattering", material.scattering),
        ("roughness", material.roughness),
    ] {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(SoundError::invalid_argument(format!(
                "Material {} {name} must be finite and in [0, 1]",
                material.id
            )));
        }
    }
    Ok(())
}

fn validate_object_refs(store: &SceneStore, object: &SceneObject) -> SoundResult<()> {
    crate::gpu::rt::validate_instance_transform(object.transform)?;
    if !store.meshes.contains_key(&object.mesh_id) {
        return Err(SoundError::not_found(format!(
            "Mesh {} does not exist",
            object.mesh_id
        )));
    }
    if !store.materials.contains_key(&object.material_id) {
        return Err(SoundError::not_found(format!(
            "Material {} does not exist",
            object.material_id
        )));
    }
    Ok(())
}

fn validate_diffraction_edge(edge: &DiffractionEdge) -> SoundResult<()> {
    if !edge.start.is_finite() || !edge.end.is_finite() {
        return Err(SoundError::invalid_argument(format!(
            "Diffraction edge {} endpoints must be finite",
            edge.id
        )));
    }
    let direction = edge.end - edge.start;
    if direction.length_squared() <= f32::EPSILON {
        return Err(SoundError::invalid_argument(format!(
            "Diffraction edge {} must have nonzero length",
            edge.id
        )));
    }
    if !edge.bisector_dir.is_finite() || edge.bisector_dir.length_squared() <= f32::EPSILON {
        return Err(SoundError::invalid_argument(format!(
            "Diffraction edge {} bisector must be finite and nonzero",
            edge.id
        )));
    }
    let edge_dir = direction.normalize();
    let bisector = edge.bisector_dir.normalize();
    if edge_dir.cross(bisector).length_squared() <= 1.0e-6 {
        return Err(SoundError::invalid_argument(format!(
            "Diffraction edge {} bisector must not be parallel to the edge",
            edge.id
        )));
    }
    if !edge.edge_angle_radians.is_finite()
        || !(std::f32::consts::PI..=std::f32::consts::TAU).contains(&edge.edge_angle_radians)
    {
        return Err(SoundError::invalid_argument(format!(
            "Diffraction edge {} angle must be finite and in [PI, 2*PI]",
            edge.id
        )));
    }
    if !edge.diffraction_radius.is_finite() || edge.diffraction_radius <= 0.0 {
        return Err(SoundError::invalid_argument(format!(
            "Diffraction edge {} radius must be finite and > 0",
            edge.id
        )));
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/unit/scene/store.rs"]
mod tests;
