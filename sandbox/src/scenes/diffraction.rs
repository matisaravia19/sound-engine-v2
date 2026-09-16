//! Load-time edge authoring for surveyed rooms without diffraction metadata.

use glam::Vec3;
use sound_engine::scene::DiffractionEdge;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

const WELD_METERS: f32 = 0.001;
const MIN_EDGE_METERS: f32 = 0.20;
const MIN_BEND_RADIANS: f32 = 20.0 * std::f32::consts::PI / 180.0;

#[derive(Clone, Copy)]
struct Side {
    face: usize,
    forward: bool,
}

/// Authors convex creases and open screen rims in world meters, once at scene load.
/// Known source/receiver positions anchor the air side of connected surfaces.
/// Coplanar seams, short details and non-manifold junctions are intentionally skipped.
pub(super) fn infer(triangles: &[[Vec3; 3]], air_points: &[Vec3]) -> Vec<DiffractionEdge> {
    let mut vertices = Vec::new();
    let mut ids = BTreeMap::new();
    let mut unique_faces = BTreeSet::new();
    let mut faces = Vec::new();

    for triangle in triangles {
        let indices = triangle.map(|point| {
            let key = point.to_array().map(|x| (x / WELD_METERS).round() as i64);
            *ids.entry(key).or_insert_with(|| {
                vertices.push(point);
                vertices.len() - 1
            })
        });
        let [a, b, c] = indices.map(|id| vertices[id]);
        let mut key = indices;
        key.sort();

        if (b - a).cross(c - a).length_squared() > 1e-12 && unique_faces.insert(key) {
            faces.push(indices);
        }
    }

    let mut edges: BTreeMap<(usize, usize), Vec<Side>> = BTreeMap::new();
    for (face, indices) in faces.iter().enumerate() {
        for i in 0..3 {
            insert_side(&mut edges, indices[i], indices[(i + 1) % 3], face);
        }
    }

    // Imported polygons can meet at a T-junction. Split unmatched long sides at
    // welded vertices before deciding that they are exposed screen boundaries.
    let original_edges = std::mem::take(&mut edges);
    for ((a, b), sides) in original_edges {
        if sides.len() != 1 {
            edges.entry((a, b)).or_default().extend(sides);
            continue;
        }
        let side = sides[0];
        let delta = vertices[b] - vertices[a];
        let length_squared = delta.length_squared();
        let mut splits = vec![(0.0, a), (1.0, b)];

        for (id, &point) in vertices.iter().enumerate() {
            let t = (point - vertices[a]).dot(delta) / length_squared;
            if t > 0.0001
                && t < 0.9999
                && point.distance_squared(vertices[a] + t * delta) < WELD_METERS.powi(2)
            {
                splits.push((t, id));
            }
        }

        splits.sort_by(|a, b| a.0.total_cmp(&b.0));
        for pair in splits.windows(2) {
            let (start, end) = if side.forward {
                (pair[0].1, pair[1].1)
            } else {
                (pair[1].1, pair[0].1)
            };
            insert_side(&mut edges, start, end, side.face);
        }
    }

    let normals: Vec<_> = faces
        .iter()
        .map(|ids| {
            let [a, b, c] = ids.map(|id| vertices[id]);
            (b - a).cross(c - a).normalize()
        })
        .collect();
    let mut neighbors = vec![Vec::new(); faces.len()];
    for sides in edges.values() {
        if let [a, b] = sides.as_slice() {
            let sign = if a.forward == b.forward {
                -1.0
            } else {
                1.0
            };
            neighbors[a.face].push((b.face, sign));
            neighbors[b.face].push((a.face, sign));
        }
    }

    // Mesh winding varies by importer/material. Orient each connected component
    // consistently, then use the nearest known air position to select its sign.
    let mut orientation = vec![0.0; faces.len()];
    for root in 0..faces.len() {
        if orientation[root] != 0.0 {
            continue;
        }
        orientation[root] = 1.0;
        let mut queue = VecDeque::from([root]);
        let mut component = Vec::new();
        while let Some(face) = queue.pop_front() {
            component.push(face);
            for &(next, sign) in &neighbors[face] {
                if orientation[next] == 0.0 {
                    orientation[next] = orientation[face] * sign;
                    queue.push_back(next);
                }
            }
        }

        let mut nearest = f32::INFINITY;
        let mut air_sign = 1.0;
        for &face in &component {
            let triangle = faces[face].map(|id| vertices[id]);
            for &point in air_points {
                let offset = point - closest_point(point, triangle);
                let signed = offset.dot(normals[face]) * orientation[face];
                if signed.abs() > WELD_METERS && offset.length_squared() < nearest {
                    nearest = offset.length_squared();
                    air_sign = signed.signum();
                }
            }
        }
        for face in component {
            orientation[face] *= air_sign;
        }
    }

    let mut result = Vec::new();
    for ((a, b), sides) in edges {
        let start = vertices[a];
        let end = vertices[b];
        let length = start.distance(end);
        if length < MIN_EDGE_METERS {
            continue;
        }
        let axis = (end - start) / length;
        let radial = |face: usize| {
            let center = faces[face].map(|id| vertices[id]).into_iter().sum::<Vec3>() / 3.0;
            let offset = center - (start + end) * 0.5;
            (offset - axis * offset.dot(axis)).normalize_or_zero()
        };

        let (bisector, angle) = match sides.as_slice() {
            [side] => (-radial(side.face), std::f32::consts::TAU),
            [a, b] => {
                let relative_sign = if a.forward == b.forward {
                    -1.0
                } else {
                    1.0
                };
                if orientation[a.face] * relative_sign != orientation[b.face] {
                    continue;
                }

                let u = radial(a.face);
                let v = radial(b.face);
                let solid_angle = u.dot(v).clamp(-1.0, 1.0).acos();
                if solid_angle > std::f32::consts::PI - MIN_BEND_RADIANS
                    || solid_angle < MIN_BEND_RADIANS
                {
                    continue;
                }
                let bisector = -(u + v).normalize_or_zero();
                let normal_a = normals[a.face] * orientation[a.face];
                let normal_b = normals[b.face] * orientation[b.face];
                if bisector.dot(normal_a) < 0.01 || bisector.dot(normal_b) < 0.01 {
                    continue;
                }
                (bisector, std::f32::consts::TAU - solid_angle)
            }
            _ => continue,
        };

        if bisector.length_squared() < 0.5 {
            continue;
        }
        result.push(DiffractionEdge {
            id: result.len() as u32 + 1,
            start,
            end,
            bisector_dir: bisector,
            edge_angle_radians: angle,
            diffraction_radius: 0.20_f32.min(length * 0.25),
        });
    }

    result
}

fn insert_side(edges: &mut BTreeMap<(usize, usize), Vec<Side>>, a: usize, b: usize, face: usize) {
    if a == b {
        return;
    }
    edges.entry((a.min(b), a.max(b))).or_default().push(Side {
        face,
        forward: a < b,
    });
}

fn closest_point(point: Vec3, [a, b, c]: [Vec3; 3]) -> Vec3 {
    let normal = (b - a).cross(c - a).normalize();
    let projected = point - normal * (point - a).dot(normal);
    if [(a, b), (b, c), (c, a)]
        .iter()
        .all(|&(start, end)| (end - start).cross(projected - start).dot(normal) >= 0.0)
    {
        return projected;
    }

    [(a, b), (b, c), (c, a)]
        .map(|(start, end)| {
            let edge = end - start;
            start + edge * ((point - start).dot(edge) / edge.length_squared()).clamp(0.0, 1.0)
        })
        .into_iter()
        .min_by(|a, b| {
            point
                .distance_squared(*a)
                .total_cmp(&point.distance_squared(*b))
        })
        .unwrap()
}

#[cfg(test)]
mod tests {
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
            assert!(
                midpoint.x == 0.0 || midpoint.x == 2.0 || midpoint.y == 0.0 || midpoint.y == 2.0
            );
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
}
