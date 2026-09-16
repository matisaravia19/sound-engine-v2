//! Shared planar geometry for acoustic validation scenes.
#![allow(dead_code)]
use glam::{Mat4, Vec3, vec3};
use sound_engine::scene::{DiffractionEdge, Material, MeshAsset, SceneDescription, SceneObject};

/// Provides a geometry-free direct-path control.
pub fn empty_scene() -> SceneDescription {
    SceneDescription {
        meshes: Vec::new(),
        materials: Vec::new(),
        objects: Vec::new(),
        diffraction_edges: Vec::new(),
    }
}

/// Builds an X=0 reflector extending 80 m in each tangential direction.
pub fn single_plane_scene(absorption: f32) -> SceneDescription {
    let extent = 80.0;
    scene_from_walls(
        vec![quad_x(0.0, -extent, extent, -extent, extent)],
        absorption,
    )
}

/// Builds two parallel X planes for image-source reflection checks.
pub fn parallel_plane_scene(length: f32, absorption: f32) -> SceneDescription {
    let extent = 80.0;
    scene_from_walls(
        vec![
            quad_x(0.0, -extent, extent, -extent, extent),
            quad_x(length, -extent, extent, -extent, extent),
        ],
        absorption,
    )
}

/// Builds six enclosing planes with uniform energy absorption.
pub fn shoebox_scene(size: Vec3, absorption: f32) -> SceneDescription {
    scene_from_walls(
        vec![
            quad_x(0.0, 0.0, size.y, 0.0, size.z),
            quad_x(size.x, 0.0, size.y, 0.0, size.z),
            quad_y(0.0, 0.0, size.x, 0.0, size.z),
            quad_y(size.y, 0.0, size.x, 0.0, size.z),
            quad_z(0.0, 0.0, size.x, 0.0, size.y),
            quad_z(size.z, 0.0, size.x, 0.0, size.y),
        ],
        absorption,
    )
}

/// Builds an air wedge around +Y, bisected by +X, with absorbing faces to isolate diffraction.
pub fn diffraction_wedge_scene(
    exterior_wedge_angle: f32,
    edge_half_length: f32,
    proxy_radius: f32,
) -> SceneDescription {
    let face_length = 80.0;
    let half_angle = 0.5 * exterior_wedge_angle;
    let upper = vec3(half_angle.cos(), 0.0, half_angle.sin()) * face_length;
    let lower = vec3(half_angle.cos(), 0.0, -half_angle.sin()) * face_length;
    let mut scene = scene_from_walls(
        vec![
            quad(
                vec3(0.0, -edge_half_length, 0.0),
                vec3(0.0, edge_half_length, 0.0),
                vec3(upper.x, edge_half_length, upper.z),
                vec3(upper.x, -edge_half_length, upper.z),
            ),
            quad(
                vec3(0.0, -edge_half_length, 0.0),
                vec3(lower.x, -edge_half_length, lower.z),
                vec3(lower.x, edge_half_length, lower.z),
                vec3(0.0, edge_half_length, 0.0),
            ),
        ],
        1.0,
    );
    scene.diffraction_edges.push(DiffractionEdge {
        id: 1,
        start: vec3(0.0, -edge_half_length, 0.0),
        end: vec3(0.0, edge_half_length, 0.0),
        // The solid corner is centered on -X, so the exterior
        // wedge and its two authored shadow boundaries are bisected by +X.
        bisector_dir: Vec3::X,
        edge_angle_radians: exterior_wedge_angle,
        diffraction_radius: proxy_radius,
    });
    scene
}

/// Packs triangle lists into one owned mesh and one broadband material.
pub fn scene_from_walls(walls: Vec<Vec<Vec3>>, absorption: f32) -> SceneDescription {
    let mut vertices = Vec::new();
    for wall in walls {
        vertices.extend(wall);
    }
    SceneDescription {
        meshes: vec![MeshAsset {
            id: 1,
            vertices,
            indices: Vec::new(),
            opaque: true,
        }],
        materials: vec![Material {
            id: 1,
            absorption_bands: vec![absorption],
            scattering: 0.0,
            roughness: 0.0,
            transmission: None,
        }],
        objects: vec![SceneObject {
            id: 1,
            mesh_id: 1,
            material_id: 1,
            transform: Mat4::IDENTITY,
            active: true,
        }],
        diffraction_edges: Vec::new(),
    }
}

/// Triangulates an axis-aligned rectangle at constant X.
pub fn quad_x(x: f32, y0: f32, y1: f32, z0: f32, z1: f32) -> Vec<Vec3> {
    quad(
        vec3(x, y0, z0),
        vec3(x, y1, z0),
        vec3(x, y1, z1),
        vec3(x, y0, z1),
    )
}

/// Triangulates an axis-aligned rectangle at constant Y.
pub fn quad_y(y: f32, x0: f32, x1: f32, z0: f32, z1: f32) -> Vec<Vec3> {
    quad(
        vec3(x0, y, z0),
        vec3(x1, y, z0),
        vec3(x1, y, z1),
        vec3(x0, y, z1),
    )
}

/// Triangulates an axis-aligned rectangle at constant Z.
pub fn quad_z(z: f32, x0: f32, x1: f32, y0: f32, y1: f32) -> Vec<Vec3> {
    quad(
        vec3(x0, y0, z),
        vec3(x1, y0, z),
        vec3(x1, y1, z),
        vec3(x0, y1, z),
    )
}

/// Triangulates an ordered quadrilateral with a shared diagonal.
pub fn quad(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> Vec<Vec3> {
    vec![a, b, c, a, c, d]
}
