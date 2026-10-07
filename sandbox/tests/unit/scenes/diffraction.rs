use super::*;
use glam::vec3;

fn quad([a, b, c, d]: [Vec3; 4]) -> [[Vec3; 3]; 2] {
    [[a, b, c], [a, c, d]]
}

#[test]
fn screen_has_rims_but_no_diagonal_or_t_junction_seams() {
    let mut triangles = Vec::new();
    for (x0, x1, y0, y1) in [
        (0.0, 1.0, 0.0, 2.0),
        (1.0, 2.0, 0.0, 1.0),
        (1.0, 2.0, 1.0, 2.0),
    ] {
        triangles.extend(quad([
            vec3(x0, y0, 0.0),
            vec3(x1, y0, 0.0),
            vec3(x1, y1, 0.0),
            vec3(x0, y1, 0.0),
        ]));
    }
    triangles.push([triangles[0][2], triangles[0][1], triangles[0][0]]);
    let edges = infer(&triangles, &[vec3(1.0, 1.0, 3.0)]);
    assert!(!edges.is_empty());
    let perimeter: f32 = edges.iter().map(|e| e.start.distance(e.end)).sum();
    assert!((perimeter - 8.0).abs() < 0.001);
    for edge in edges {
        let midpoint = (edge.start + edge.end) * 0.5;
        assert!(midpoint.x == 0.0 || midpoint.x == 2.0 || midpoint.y == 0.0 || midpoint.y == 2.0);
        assert!((edge.edge_angle_radians - std::f32::consts::TAU).abs() < 1e-5);
        assert!(edge.bisector_dir.dot(midpoint - vec3(1.0, 1.0, 0.0)) > 0.0);
    }
}

#[test]
fn convex_box_diffracts_but_inward_room_corners_do_not() {
    let points = [
        vec3(-1.0, -1.0, -1.0),
        vec3(1.0, -1.0, -1.0),
        vec3(1.0, 1.0, -1.0),
        vec3(-1.0, 1.0, -1.0),
        vec3(-1.0, -1.0, 1.0),
        vec3(1.0, -1.0, 1.0),
        vec3(1.0, 1.0, 1.0),
        vec3(-1.0, 1.0, 1.0),
    ];
    let mut triangles: Vec<_> = [
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 4, 7, 3],
        [1, 2, 6, 5],
        [3, 7, 6, 2],
        [0, 1, 5, 4],
    ]
    .into_iter()
    .flat_map(|ids| quad(ids.map(|id| points[id])))
    .collect();
    // Imported faces can have inconsistent winding, even within one object.
    for triangle in triangles.iter_mut().step_by(3) {
        triangle.swap(0, 2);
    }
    let edges = infer(&triangles, &[vec3(3.0, 4.0, 5.0)]);
    assert_eq!(edges.len(), 12);
    for edge in edges {
        assert!((edge.edge_angle_radians - 1.5 * std::f32::consts::PI).abs() < 1e-5);
        assert!(edge.bisector_dir.dot((edge.start + edge.end) * 0.5) > 0.0);
    }
    assert!(infer(&triangles, &[Vec3::ZERO]).is_empty());
}
