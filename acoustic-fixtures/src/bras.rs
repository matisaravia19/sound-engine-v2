//! BRAS RS5/RS6 geometry, matching the measured-reference benchmark.
use super::geometry;
use glam::{Mat4, Vec3, vec3};
use sound_engine::scene::{DiffractionEdge, Material, MeshAsset, SceneDescription, SceneObject};
/// Measured geometry used by the RS5 and RS6 comparisons.
#[derive(Clone, Copy)]
pub enum Setup {
    /// RS5: a thin vertical partition above a rigid floor.
    Partition,
    /// RS6: the rigid outer envelope of a hollow-block assembly.
    Block,
}

/// Builds a measured rigid screen or block with optional authored diffraction edges.
pub fn scene(setup: Setup, radius: f32, diffraction: bool) -> SceneDescription {
    // BRAS uses Z up. Positions.csv explicitly maps (X,Y,Z) to engine (X,Z,Y).
    let mut scene =
        geometry::scene_from_walls(vec![geometry::quad_y(0.0, -20.0, 20.0, -20.0, 20.0)], 0.0);
    let (walls, edges) = if matches!(setup, Setup::Partition) {
        // Thin-screen approximation of the 25 mm partition, at its mid-plane.
        // Height and span come from RS5_RIR.png; no RIR was used to adjust them.
        let (x, height, z0, z1) = (5.4995, 2.066, 2.985 - 4.75 / 2.0, 2.985 + 4.75 / 2.0);
        (
            vec![geometry::quad_x(x, 0.0, height, z0, z1)],
            vec![
                edge(
                    1,
                    vec3(x, height, z0),
                    vec3(x, height, z1),
                    Vec3::Y,
                    std::f32::consts::TAU,
                    radius,
                ),
                edge(
                    2,
                    vec3(x, 0.0, z0),
                    vec3(x, height, z0),
                    -Vec3::Z,
                    std::f32::consts::TAU,
                    radius,
                ),
                edge(
                    3,
                    vec3(x, 0.0, z1),
                    vec3(x, height, z1),
                    Vec3::Z,
                    std::f32::consts::TAU,
                    radius,
                ),
            ],
        )
    } else {
        let (x0, x1, height, z0, z1) = (5.5 - 0.36, 5.5 + 0.36, 0.72, 2.985 - 2.07, 2.985 + 2.07);
        let n = std::f32::consts::FRAC_1_SQRT_2;
        let angle = 1.5 * std::f32::consts::PI;
        // The hollow-block assembly is represented by its measured outer envelope.
        (
            vec![
                geometry::quad_x(x0, 0.0, height, z0, z1),
                geometry::quad_x(x1, 0.0, height, z0, z1),
                geometry::quad_y(height, x0, x1, z0, z1),
                geometry::quad_z(z0, x0, x1, 0.0, height),
                geometry::quad_z(z1, x0, x1, 0.0, height),
            ],
            vec![
                edge(
                    1,
                    vec3(x0, height, z0),
                    vec3(x0, height, z1),
                    vec3(-n, n, 0.0),
                    angle,
                    radius,
                ),
                edge(
                    2,
                    vec3(x1, height, z0),
                    vec3(x1, height, z1),
                    vec3(n, n, 0.0),
                    angle,
                    radius,
                ),
                edge(
                    3,
                    vec3(x0, 0.0, z0),
                    vec3(x0, height, z0),
                    vec3(-n, 0.0, -n),
                    angle,
                    radius,
                ),
                edge(
                    4,
                    vec3(x1, 0.0, z0),
                    vec3(x1, height, z0),
                    vec3(n, 0.0, -n),
                    angle,
                    radius,
                ),
                edge(
                    5,
                    vec3(x0, 0.0, z1),
                    vec3(x0, height, z1),
                    vec3(-n, 0.0, n),
                    angle,
                    radius,
                ),
                edge(
                    6,
                    vec3(x1, 0.0, z1),
                    vec3(x1, height, z1),
                    vec3(n, 0.0, n),
                    angle,
                    radius,
                ),
                edge(
                    7,
                    vec3(x0, height, z0),
                    vec3(x1, height, z0),
                    vec3(0.0, n, -n),
                    angle,
                    radius,
                ),
                edge(
                    8,
                    vec3(x0, height, z1),
                    vec3(x1, height, z1),
                    vec3(0.0, n, n),
                    angle,
                    radius,
                ),
            ],
        )
    };
    scene.meshes.push(MeshAsset {
        id: 2,
        vertices: walls.into_iter().flatten().collect(),
        indices: Vec::new(),
        opaque: true,
    });
    // Rigid boundary is explicit pending use of the supplied MDF material data.
    scene.materials.push(Material {
        id: 2,
        absorption: 0.0,
        scattering: 0.0,
        roughness: 0.0,
    });
    scene.objects.push(SceneObject {
        id: 2,
        mesh_id: 2,
        material_id: 2,
        transform: Mat4::IDENTITY,
        active: true,
    });
    if diffraction {
        scene.diffraction_edges = edges;
    }
    scene
}

fn edge(
    id: u32,
    start: Vec3,
    end: Vec3,
    bisector_dir: Vec3,
    angle: f32,
    radius: f32,
) -> DiffractionEdge {
    DiffractionEdge {
        id,
        start,
        end,
        bisector_dir,
        edge_angle_radians: angle,
        diffraction_radius: radius,
    }
}
