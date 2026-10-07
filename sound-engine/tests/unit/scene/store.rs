use super::*;
use crate::core::error::ErrorCode;
use glam::{Mat4, vec3};

#[test]
fn object_edits_share_geometry_and_keep_cloned_scenes_unchanged() {
    let mut store = populated_store();
    let before = store.clone();
    let vertex_pointer = store.meshes[&1].asset.vertices.as_ptr();
    let transform = Mat4::from_translation(vec3(2.0, 0.0, 0.0));
    store
        .apply_updates(SceneUpdates {
            updates: vec![SceneUpdate::SetTransform {
                id: 10,
                transform,
            }],
        })
        .unwrap();
    assert_eq!(store.meshes[&1].asset.vertices.as_ptr(), vertex_pointer);
    assert_eq!(store.mesh_revision(1), Some(1));
    assert_eq!(store.objects[&10].transform, transform);
    assert_eq!(before.objects[&10].transform, Mat4::IDENTITY);
    assert_eq!(before.version(), 1);

    store
        .apply_updates(SceneUpdates {
            updates: vec![SceneUpdate::ReplaceMesh(mesh(1))],
        })
        .unwrap();
    assert_ne!(store.meshes[&1].asset.vertices.as_ptr(), vertex_pointer);
    assert_eq!(before.meshes[&1].asset.vertices.as_ptr(), vertex_pointer);
    assert_eq!(store.mesh_revision(1), Some(3));
}

#[test]
fn failed_batch_rolls_back_geometry_revisions_and_object_changes() {
    let mut store = populated_store();
    let revisions = store.revisions();
    let vertex_pointer = store.meshes[&1].asset.vertices.as_ptr();
    assert!(
        store
            .apply_updates(SceneUpdates {
                updates: vec![
                    SceneUpdate::AddDiffractionEdge(edge(1)),
                    SceneUpdate::ReplaceMesh(mesh(1)),
                    SceneUpdate::SetActive {
                        id: 10,
                        active: false
                    },
                    SceneUpdate::RemoveMaterial(1),
                ],
            })
            .is_err()
    );
    assert_eq!(store.version(), 1);
    assert_eq!(store.revisions(), revisions);
    assert!(store.diffraction_edges().next().is_none());
    assert_eq!(store.mesh_revision(1), Some(1));
    assert_eq!(store.meshes[&1].asset.vertices.as_ptr(), vertex_pointer);
    assert!(store.objects[&10].active);
}

#[test]
fn exhausted_revision_rejects_changes_without_invalidating_the_scene() {
    let mut store = populated_store();
    store.version = SceneVersion::MAX;
    assert!(
        store
            .apply_updates(SceneUpdates {
                updates: vec![SceneUpdate::SetActive {
                    id: 10,
                    active: false
                }],
            })
            .is_err()
    );
    assert!(store.load(SceneDescription::default()).is_err());
    store
        .apply_updates(SceneUpdates {
            updates: Vec::new(),
        })
        .unwrap();
    assert!(store.objects[&10].active);
    assert_eq!(store.version(), SceneVersion::MAX);
}

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
                    absorption: 0.4,
                    scattering: 0.2,
                    roughness: 0.2,
                }),
                SceneUpdate::SetMaterial {
                    id: 10,
                    material_id: 2,
                },
                SceneUpdate::SetMaterial {
                    id: 20,
                    material_id: 2,
                },
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
                absorption: 0.2,
                scattering: 0.1,
                roughness: 0.1,
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
        vertices: vec![
            vec3(0.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            vec3(0.0, 1.0, 0.0),
        ],
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
                absorption: 0.2,
                scattering: 0.1,
                roughness: 0.1,
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
                absorption: 0.2,
                scattering: 0.1,
                roughness: 0.1,
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
                    absorption: 0.2,
                    scattering: 0.1,
                    roughness: 0.1,
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
    assert_eq!(
        store.diffraction_edges.get(&1).unwrap().diffraction_radius,
        0.5
    );

    let mut replacement = edge(1);
    replacement.diffraction_radius = 1.0;
    store
        .apply_updates(SceneUpdates {
            updates: vec![SceneUpdate::ReplaceDiffractionEdge(replacement)],
        })
        .unwrap();
    assert_eq!(
        store.diffraction_edges.get(&1).unwrap().diffraction_radius,
        1.0
    );

    store
        .apply_updates(SceneUpdates {
            updates: vec![SceneUpdate::RemoveDiffractionEdge(1)],
        })
        .unwrap();

    assert!(!store.diffraction_edges.contains_key(&1));
}

#[test]
fn invalid_material_coefficients_cannot_enter_or_replace_scene_state() {
    for value in [-0.01, 1.01, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for field in 0..3 {
            let mut material = Material {
                id: 1,
                absorption: 0.2,
                scattering: 0.1,
                roughness: 0.1,
            };
            match field {
                0 => material.absorption = value,
                1 => material.scattering = value,
                _ => material.roughness = value,
            }
            let mut store = populated_store();
            let version = store.version();
            let original = store.materials[&1].clone();
            assert_eq!(
                store
                    .apply_updates(SceneUpdates {
                        updates: vec![SceneUpdate::ReplaceMaterial(material.clone())],
                    })
                    .unwrap_err()
                    .code(),
                ErrorCode::InvalidArgument
            );
            assert_eq!(store.version(), version);
            assert_eq!(store.materials[&1].absorption, original.absorption);
            assert_eq!(store.materials[&1].scattering, original.scattering);
            assert_eq!(store.materials[&1].roughness, original.roughness);
            assert_eq!(
                store
                    .load(SceneDescription {
                        materials: vec![material],
                        ..Default::default()
                    })
                    .unwrap_err()
                    .code(),
                ErrorCode::InvalidArgument
            );
            assert_eq!(store.version(), version);
        }
    }
}

#[test]
fn overflowing_edge_directions_cannot_enter_or_replace_scene_state() {
    let mut store = populated_store();
    store
        .apply_updates(SceneUpdates {
            updates: vec![SceneUpdate::AddDiffractionEdge(edge(1))],
        })
        .unwrap();
    let version = store.version();
    let revisions = store.revisions();
    let mut invalid = edge(1);
    // Finite endpoints can still overflow when subtracted or squared for normalization.
    for (start, end) in [
        (glam::Vec3::splat(-f32::MAX), glam::Vec3::splat(f32::MAX)),
        (glam::Vec3::ZERO, vec3(1.0e20, 0.0, 0.0)),
    ] {
        invalid.start = start;
        invalid.end = end;
        assert_eq!(
            store
                .load(SceneDescription {
                    diffraction_edges: vec![invalid.clone()],
                    ..Default::default()
                })
                .unwrap_err()
                .code(),
            ErrorCode::InvalidArgument
        );
        for update in [
            SceneUpdate::ReplaceDiffractionEdge(invalid.clone()),
            SceneUpdate::AddDiffractionEdge(DiffractionEdge {
                id: 2,
                ..invalid.clone()
            }),
        ] {
            assert_eq!(
                store
                    .apply_updates(SceneUpdates {
                        updates: vec![
                            SceneUpdate::SetActive {
                                id: 10,
                                active: false
                            },
                            update
                        ],
                    })
                    .unwrap_err()
                    .code(),
                ErrorCode::InvalidArgument
            );
        }
        assert_eq!(store.version(), version);
        assert_eq!(store.revisions(), revisions);
        assert_eq!(store.meshes().count(), 2);
        assert!(store.objects[&10].active);
        assert_eq!(store.diffraction_edges().count(), 1);
        assert_eq!(store.diffraction_edges[&1].start, edge(1).start);
        assert_eq!(store.diffraction_edges[&1].end, edge(1).end);
    }
}
