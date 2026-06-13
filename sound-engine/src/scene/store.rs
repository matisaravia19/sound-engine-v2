use super::types::{
    Material, MaterialId, MaterialUpdate, MeshAsset, MeshId, ObjectId, ObjectUpdate, SceneDescription, SceneObject,
    SceneUpdates, SceneVersion, TopologyChange,
};
use crate::error::{SoundError, SoundResult};
use std::collections::HashMap;

/// Canonical CPU-side scene state.
///
/// `SceneStore` owns validated mesh, material, and object records. Any accepted
/// mutation increments `SceneVersion`, which drives lazy GPU synchronization.
#[derive(Default)]
pub struct SceneStore {
    pub(super) version: SceneVersion,
    pub(super) meshes: HashMap<MeshId, MeshAsset>,
    pub(super) materials: HashMap<MaterialId, Material>,
    pub(super) objects: HashMap<ObjectId, SceneObject>,
}

impl SceneStore {
    /// Creates an empty scene store at version zero.
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the entire CPU scene and increments the version.
    pub fn load(&mut self, scene: SceneDescription) -> SoundResult<()> {
        let mut next = Self::default();
        for mesh in scene.meshes {
            validate_mesh(&mesh)?;
            if next.meshes.insert(mesh.id, mesh).is_some() {
                return Err(SoundError::invalid_argument("Duplicate mesh id in scene"));
            }
        }
        for material in scene.materials {
            validate_material(&material)?;
            if next.materials.insert(material.id, material).is_some() {
                return Err(SoundError::invalid_argument("Duplicate material id in scene"));
            }
        }
        for object in scene.objects {
            validate_object_refs(&next, &object)?;
            if next.objects.insert(object.id, object).is_some() {
                return Err(SoundError::invalid_argument("Duplicate object id in scene"));
            }
        }

        next.version = SceneVersion(self.version.0.saturating_add(1));
        *self = next;
        Ok(())
    }

    /// Applies incremental changes and increments the version if anything changed.
    pub fn apply_updates(&mut self, updates: SceneUpdates) -> SoundResult<()> {
        let mut changed = false;

        for change in updates.topology_changes {
            changed = true;
            self.apply_topology_change(change)?;
        }

        for update in updates.material_updates {
            changed = true;
            self.apply_material_update(update)?;
        }

        for update in updates.object_updates {
            changed = true;
            self.apply_object_update(update)?;
        }

        if changed {
            self.version.0 = self.version.0.saturating_add(1);
        }
        Ok(())
    }

    /// Returns the current scene revision.
    pub fn version(&self) -> SceneVersion {
        self.version
    }

    /// Iterates over all mesh assets in unspecified order.
    pub fn meshes(&self) -> impl Iterator<Item = &MeshAsset> {
        self.meshes.values()
    }

    /// Iterates over all materials in unspecified order.
    pub fn materials(&self) -> impl Iterator<Item = &Material> {
        self.materials.values()
    }

    /// Iterates over all objects in unspecified order.
    pub fn objects(&self) -> impl Iterator<Item = &SceneObject> {
        self.objects.values()
    }

    fn apply_topology_change(&mut self, change: TopologyChange) -> SoundResult<()> {
        match change {
            TopologyChange::AddMesh(mesh) => {
                validate_mesh(&mesh)?;
                if self.meshes.insert(mesh.id, mesh).is_some() {
                    return Err(SoundError::invalid_argument("AddMesh received an existing mesh id"));
                }
            }
            TopologyChange::ReplaceMesh(mesh) => {
                validate_mesh(&mesh)?;
                if !self.meshes.contains_key(&mesh.id) {
                    return Err(SoundError::not_found(format!("Mesh {} does not exist", mesh.id.0)));
                }
                self.meshes.insert(mesh.id, mesh);
            }
            TopologyChange::RemoveMesh(id) => {
                if self.objects.values().any(|object| object.mesh_id == id) {
                    return Err(SoundError::invalid_state(format!("Mesh {} is still referenced", id.0)));
                }
                self.meshes
                    .remove(&id)
                    .ok_or_else(|| SoundError::not_found(format!("Mesh {} does not exist", id.0)))?;
            }
            TopologyChange::AddObject(object) => {
                validate_object_refs(self, &object)?;
                if self.objects.insert(object.id, object).is_some() {
                    return Err(SoundError::invalid_argument("AddObject received an existing object id"));
                }
            }
            TopologyChange::ReplaceObject(object) => {
                validate_object_refs(self, &object)?;
                if !self.objects.contains_key(&object.id) {
                    return Err(SoundError::not_found(format!("Object {} does not exist", object.id.0)));
                }
                self.objects.insert(object.id, object);
            }
            TopologyChange::RemoveObject(id) => {
                self.objects
                    .remove(&id)
                    .ok_or_else(|| SoundError::not_found(format!("Object {} does not exist", id.0)))?;
            }
        }
        Ok(())
    }

    fn apply_material_update(&mut self, update: MaterialUpdate) -> SoundResult<()> {
        let material = self
            .materials
            .get_mut(&update.id)
            .ok_or_else(|| SoundError::not_found(format!("Material {} does not exist", update.id.0)))?;
        if let Some(absorption_bands) = update.absorption_bands {
            material.absorption_bands = absorption_bands;
        }
        if let Some(scattering) = update.scattering {
            material.scattering = scattering;
        }
        if let Some(transmission) = update.transmission {
            material.transmission = transmission;
        }
        validate_material(material)
    }

    fn apply_object_update(&mut self, update: ObjectUpdate) -> SoundResult<()> {
        match update {
            ObjectUpdate::SetTransform { id, transform } => {
                self.object_mut(id)?.transform = transform;
            }
            ObjectUpdate::SetActive { id, active } => {
                self.object_mut(id)?.active = active;
            }
            ObjectUpdate::SetMaterial { id, material_id } => {
                if !self.materials.contains_key(&material_id) {
                    return Err(SoundError::not_found(format!(
                        "Material {} does not exist",
                        material_id.0
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
            .ok_or_else(|| SoundError::not_found(format!("Object {} does not exist", id.0)))
    }
}

fn validate_mesh(mesh: &MeshAsset) -> SoundResult<()> {
    if mesh.vertices.is_empty() {
        return Err(SoundError::invalid_argument(format!(
            "Mesh {} has no vertices",
            mesh.id.0
        )));
    }
    if mesh.indices.is_empty() && mesh.vertices.len() < 3 {
        return Err(SoundError::invalid_argument(format!(
            "Mesh {} has less than one triangle",
            mesh.id.0
        )));
    }
    if !mesh.indices.is_empty() && mesh.indices.len() < 3 {
        return Err(SoundError::invalid_argument(format!(
            "Mesh {} has less than one indexed triangle",
            mesh.id.0
        )));
    }
    Ok(())
}

fn validate_material(material: &Material) -> SoundResult<()> {
    if material.absorption_bands.is_empty() {
        return Err(SoundError::invalid_argument(format!(
            "Material {} must contain at least one absorption band",
            material.id.0
        )));
    }
    if !(0.0..=1.0).contains(&material.scattering) {
        return Err(SoundError::invalid_argument(format!(
            "Material {} scattering must be in [0, 1]",
            material.id.0
        )));
    }
    Ok(())
}

fn validate_object_refs(store: &SceneStore, object: &SceneObject) -> SoundResult<()> {
    if !store.meshes.contains_key(&object.mesh_id) {
        return Err(SoundError::not_found(format!(
            "Mesh {} does not exist",
            object.mesh_id.0
        )));
    }
    if !store.materials.contains_key(&object.material_id) {
        return Err(SoundError::not_found(format!(
            "Material {} does not exist",
            object.material_id.0
        )));
    }
    Ok(())
}
