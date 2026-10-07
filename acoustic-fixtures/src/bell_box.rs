//! Figure 8 of Tsingos et al., “Validation of Acoustical Simulations in the Bell Labs Box”.
//! Coordinates: X left to right in the plan, Y height, Z bottom to top in the plan.

use super::geometry::{quad_z, scene_from_walls, shoebox_scene};
use glam::{Vec3, vec3};
use sound_engine::scene::{DiffractionEdge, SceneDescription};

/// Interior dimensions in meters (Section 4).
pub const SIZE: Vec3 = vec3(2.19, 2.42, 3.03);

/// Mouth position in the empty configuration (Figure 8a).
pub const EMPTY_SOURCE: Vec3 = vec3(1.07, 1.35, 3.03 - 0.77);

/// Microphone position in the empty configuration (Figure 8a).
pub const EMPTY_LISTENER: Vec3 = vec3(1.35, 1.32, 0.93);

/// Mouth position with the baffle (Figure 8b).
pub const BAFFLE_SOURCE: Vec3 = vec3(1.45, 1.35, 3.03 - 0.59);

/// Microphone position with the baffle (Figure 8b).
pub const BAFFLE_LISTENER: Vec3 = vec3(1.21, 1.32, 1.02);

/// Lower endpoint of the exposed edge; panel extends 1.26 m from the right wall.
pub const EDGE_START: Vec3 = vec3(2.19 - 1.26, 0.0, 1.91);

/// Builds a broadband approximation; original complex wall calibrations are unavailable.
pub fn scene(baffle: bool, diffraction: bool, absorption: f32, radius: f32) -> SceneDescription {
    let mut scene = shoebox_scene(SIZE, absorption);

    if baffle {
        // Zero thickness is an explicit approximation: Figure 8 dimensions
        // locate the edge but do not specify both acoustic faces precisely.
        let panel = scene_from_walls(
            vec![quad_z(EDGE_START.z, EDGE_START.x, SIZE.x, 0.0, SIZE.y)],
            absorption,
        );
        scene.meshes[0]
            .vertices
            .extend_from_slice(&panel.meshes[0].vertices);

        if diffraction {
            scene.diffraction_edges.push(DiffractionEdge {
                id: 1,
                start: EDGE_START,
                end: EDGE_START + Vec3::Y * SIZE.y,
                bisector_dir: -Vec3::X,
                edge_angle_radians: std::f32::consts::TAU,
                diffraction_radius: radius,
            });
        }
    }

    scene
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn figure_8_matches_table_1_shortest_diffraction_and_reflection() {
        let radial = |v: Vec3| vec3(v.x - EDGE_START.x, 0.0, v.z - EDGE_START.z).length();
        let diffraction = (radial(BAFFLE_SOURCE) + radial(BAFFLE_LISTENER))
            .hypot(BAFFLE_SOURCE.y - BAFFLE_LISTENER.y);
        let image_source = vec3(-BAFFLE_SOURCE.x, BAFFLE_SOURCE.y, BAFFLE_SOURCE.z);

        assert!((diffraction - 1.6758).abs() < 0.001);
        assert!((image_source.distance(BAFFLE_LISTENER) - 3.0154).abs() < 0.001);
    }
}
