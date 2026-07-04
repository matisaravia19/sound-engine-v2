use super::types::{
    DiffractionEdge, DiffractionEdgeId, Material, MaterialId, MeshAsset, MeshId, ObjectId, SceneDescription,
    SceneObject, SceneUpdate, SceneUpdates, SceneVersion,
};
use crate::core::error::{SoundError, SoundResult};
use std::collections::HashMap;

/// Canonical CPU-side scene state.
///
/// `SceneStore` owns validated mesh, material, and object records. Any accepted
/// mutation increments `SceneVersion`, which drives lazy GPU synchronization.
#[derive(Clone, Default)]
pub struct SceneStore {
    pub(super) version: SceneVersion,
    pub(super) meshes: HashMap<MeshId, MeshAsset>,
    pub(super) materials: HashMap<MaterialId, Material>,
    pub(super) objects: HashMap<ObjectId, SceneObject>,
    pub(super) diffraction_edges: HashMap<DiffractionEdgeId, DiffractionEdge>,
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
        for edge in scene.diffraction_edges {
            validate_diffraction_edge(&edge)?;
            if next.diffraction_edges.insert(edge.id, edge).is_some() {
                return Err(SoundError::invalid_argument("Duplicate diffraction edge id in scene"));
            }
        }

        next.version = self.version.saturating_add(1);
        *self = next;
        Ok(())
    }

    /// Applies incremental changes and increments the version if anything changed.
    pub fn apply_updates(&mut self, updates: SceneUpdates) -> SoundResult<()> {
        let changed = !updates.updates.is_empty();
        let mut next = self.clone();

        for update in updates.updates {
            next.apply_update(update)?;
        }

        if changed {
            next.version = self.version.saturating_add(1);
            *self = next;
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

    /// Iterates over all manually authored diffraction edges in unspecified order.
    pub fn diffraction_edges(&self) -> impl Iterator<Item = &DiffractionEdge> {
        self.diffraction_edges.values()
    }

    fn apply_update(&mut self, update: SceneUpdate) -> SoundResult<()> {
        match update {
            SceneUpdate::AddMesh(mesh) => {
                validate_mesh(&mesh)?;
                if self.meshes.insert(mesh.id, mesh).is_some() {
                    return Err(SoundError::invalid_argument("AddMesh received an existing mesh id"));
                }
            }
            SceneUpdate::ReplaceMesh(mesh) => {
                validate_mesh(&mesh)?;
                if !self.meshes.contains_key(&mesh.id) {
                    return Err(SoundError::not_found(format!("Mesh {} does not exist", mesh.id)));
                }
                self.meshes.insert(mesh.id, mesh);
            }
            SceneUpdate::RemoveMesh(id) => {
                if self.objects.values().any(|object| object.mesh_id == id) {
                    return Err(SoundError::invalid_state(format!("Mesh {} is still referenced", id)));
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
                self.materials
                    .remove(&id)
                    .ok_or_else(|| SoundError::not_found(format!("Material {} does not exist", id)))?;
            }
            SceneUpdate::AddObject(object) => {
                validate_object_refs(self, &object)?;
                if self.objects.insert(object.id, object).is_some() {
                    return Err(SoundError::invalid_argument("AddObject received an existing object id"));
                }
            }
            SceneUpdate::ReplaceObject(object) => {
                validate_object_refs(self, &object)?;
                if !self.objects.contains_key(&object.id) {
                    return Err(SoundError::not_found(format!("Object {} does not exist", object.id)));
                }
                self.objects.insert(object.id, object);
            }
            SceneUpdate::RemoveObject(id) => {
                self.objects
                    .remove(&id)
                    .ok_or_else(|| SoundError::not_found(format!("Object {} does not exist", id)))?;
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
                self.diffraction_edges
                    .remove(&id)
                    .ok_or_else(|| SoundError::not_found(format!("Diffraction edge {} does not exist", id)))?;
            }
            SceneUpdate::SetTransform { id, transform } => {
                self.object_mut(id)?.transform = transform;
            }
            SceneUpdate::SetActive { id, active } => {
                self.object_mut(id)?.active = active;
            }
            SceneUpdate::SetMaterial { id, material_id } => {
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
    if material.absorption_bands.is_empty() {
        return Err(SoundError::invalid_argument(format!(
            "Material {} must contain at least one absorption band",
            material.id
        )));
    }
    if !(0.0..=1.0).contains(&material.scattering) {
        return Err(SoundError::invalid_argument(format!(
            "Material {} scattering must be in [0, 1]",
            material.id
        )));
    }
    Ok(())
}

fn validate_object_refs(store: &SceneStore, object: &SceneObject) -> SoundResult<()> {
    if !store.meshes.contains_key(&object.mesh_id) {
        return Err(SoundError::not_found(format!("Mesh {} does not exist", object.mesh_id)));
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
    if !(0.0..=1.0).contains(&edge.base_strength) {
        return Err(SoundError::invalid_argument(format!(
            "Diffraction edge {} base strength must be in [0, 1]",
            edge.id
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::error::ErrorCode;
    use glam::{Mat4, vec3};

    #[test]
    fn apply_updates_can_remove_object_and_its_mesh_in_one_batch() {
        let mut store = populated_store();

        store
            .apply_updates(SceneUpdates {
                updates: vec![SceneUpdate::RemoveObject(10), SceneUpdate::RemoveMesh(1)],
            })
            .unwrap();

        assert!(!store.meshes.contains_key(&1));
        assert!(!store.objects.contains_key(&10));
        assert!(store.meshes.contains_key(&2));
        assert!(store.objects.contains_key(&20));
        assert_eq!(store.version(), 2);
    }

    #[test]
    fn apply_updates_rejects_mesh_removal_when_current_scene_still_references_it() {
        let mut store = populated_store();
        let original_version = store.version();

        let error = store
            .apply_updates(SceneUpdates {
                updates: vec![SceneUpdate::RemoveMesh(1)],
            })
            .unwrap_err();

        assert_eq!(error.code(), ErrorCode::InvalidState);
        assert!(store.meshes.contains_key(&1));
        assert!(store.objects.contains_key(&10));
        assert_eq!(store.version(), original_version);
    }

    #[test]
    fn apply_updates_can_add_reassign_and_remove_materials_in_order() {
        let mut store = populated_store();

        store
            .apply_updates(SceneUpdates {
                updates: vec![
                    SceneUpdate::AddMaterial(Material {
                        id: 2,
                        absorption_bands: vec![0.4],
                        scattering: 0.2,
                        transmission: None,
                    }),
                    SceneUpdate::SetMaterial { id: 10, material_id: 2 },
                    SceneUpdate::SetMaterial { id: 20, material_id: 2 },
                    SceneUpdate::RemoveMaterial(1),
                ],
            })
            .unwrap();

        assert!(!store.materials.contains_key(&1));
        assert!(store.materials.contains_key(&2));
        assert_eq!(store.objects.get(&10).unwrap().material_id, 2);
        assert_eq!(store.objects.get(&20).unwrap().material_id, 2);
    }

    #[test]
    fn apply_updates_rejects_material_removal_when_current_scene_still_references_it() {
        let mut store = populated_store();
        let original_version = store.version();

        let error = store
            .apply_updates(SceneUpdates {
                updates: vec![SceneUpdate::RemoveMaterial(1)],
            })
            .unwrap_err();

        assert_eq!(error.code(), ErrorCode::InvalidState);
        assert!(store.materials.contains_key(&1));
        assert_eq!(store.version(), original_version);
    }

    fn populated_store() -> SceneStore {
        let mut store = SceneStore::new();
        store
            .load(SceneDescription {
                meshes: vec![mesh(1), mesh(2)],
                materials: vec![Material {
                    id: 1,
                    absorption_bands: vec![0.2],
                    scattering: 0.1,
                    transmission: None,
                }],
                objects: vec![object(10, 1), object(20, 2)],
                diffraction_edges: Vec::new(),
            })
            .unwrap();
        store
    }

    fn mesh(id: MeshId) -> MeshAsset {
        MeshAsset {
            id,
            vertices: vec![vec3(0.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0)],
            indices: Vec::new(),
            opaque: true,
        }
    }

    fn object(id: ObjectId, mesh_id: MeshId) -> SceneObject {
        SceneObject {
            id,
            mesh_id,
            material_id: 1,
            transform: Mat4::IDENTITY,
            active: true,
        }
    }

    fn edge(id: DiffractionEdgeId) -> DiffractionEdge {
        DiffractionEdge {
            id,
            start: vec3(0.0, 0.0, 0.0),
            end: vec3(0.0, 1.0, 0.0),
            bisector_dir: vec3(1.0, 0.0, 0.0),
            edge_angle_radians: 1.5 * std::f32::consts::PI,
            diffraction_radius: 0.5,
            base_strength: 0.25,
        }
    }

    #[test]
    fn load_accepts_valid_diffraction_edges() {
        let mut store = SceneStore::new();

        store
            .load(SceneDescription {
                meshes: vec![mesh(1)],
                materials: vec![Material {
                    id: 1,
                    absorption_bands: vec![0.2],
                    scattering: 0.1,
                    transmission: None,
                }],
                objects: vec![object(10, 1)],
                diffraction_edges: vec![edge(1)],
            })
            .unwrap();

        assert_eq!(store.diffraction_edges().count(), 1);
    }

    #[test]
    fn load_rejects_invalid_diffraction_edges() {
        let mut invalid = edge(1);
        invalid.end = invalid.start;

        let mut store = SceneStore::new();
        let error = store
            .load(SceneDescription {
                meshes: vec![mesh(1)],
                materials: vec![Material {
                    id: 1,
                    absorption_bands: vec![0.2],
                    scattering: 0.1,
                    transmission: None,
                }],
                objects: vec![object(10, 1)],
                diffraction_edges: vec![invalid],
            })
            .unwrap_err();

        assert_eq!(error.code(), ErrorCode::InvalidArgument);
    }

    #[test]
    fn load_rejects_diffraction_angles_outside_exterior_range() {
        for angle in [std::f32::consts::PI - 0.01, std::f32::consts::TAU + 0.01] {
            let mut invalid = edge(1);
            invalid.edge_angle_radians = angle;

            let mut store = SceneStore::new();
            let error = store
                .load(SceneDescription {
                    meshes: vec![mesh(1)],
                    materials: vec![Material {
                        id: 1,
                        absorption_bands: vec![0.2],
                        scattering: 0.1,
                        transmission: None,
                    }],
                    objects: vec![object(10, 1)],
                    diffraction_edges: vec![invalid],
                })
                .unwrap_err();

            assert_eq!(error.code(), ErrorCode::InvalidArgument);
        }
    }

    #[test]
    fn apply_updates_can_add_replace_and_remove_diffraction_edges() {
        let mut store = populated_store();

        store
            .apply_updates(SceneUpdates {
                updates: vec![SceneUpdate::AddDiffractionEdge(edge(1))],
            })
            .unwrap();
        assert_eq!(store.diffraction_edges.get(&1).unwrap().base_strength, 0.25);

        let mut replacement = edge(1);
        replacement.base_strength = 0.5;
        store
            .apply_updates(SceneUpdates {
                updates: vec![
                    SceneUpdate::ReplaceDiffractionEdge(replacement),
                    SceneUpdate::RemoveDiffractionEdge(1),
                ],
            })
            .unwrap();

        assert!(!store.diffraction_edges.contains_key(&1));
    }
}
