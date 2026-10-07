//! Editable room geometry and coordinate conversion.
use glam::{Mat4, Vec3, vec3};
use raylib::prelude::{Color, Vector3};
use sound_engine::scene::{DiffractionEdge, Material, MeshAsset, SceneDescription, SceneObject};

/// Half-width shared by room construction and floor rendering, in meters.
pub(super) const ROOM_HALF_X: f32 = 8.0;
/// Half-depth shared by room construction and floor rendering, in meters.
pub(super) const ROOM_HALF_Z: f32 = 5.0;
const WALL_HEIGHT: f32 = 3.0;
const WALL_THICKNESS: f32 = 0.25;
const DOOR_HALF_WIDTH: f32 = 1.0;
const ROOF_THICKNESS: f32 = 0.2;

/// Display geometry and acoustic edge definitions for the editable room.
pub(super) struct PlaygroundScene {
    /// Owned triangle data submitted to acoustic scene construction.
    pub(super) acoustic_mesh: AcousticMesh,
    /// Visible walls and roof, also used to construct acoustic triangles.
    pub(super) blocks: Vec<Block>,
    /// Authored doorway edges in world coordinates.
    pub(super) diffraction_edges: Vec<DiffractionEdge>,
}

/// Owned indexed display mesh used to construct acoustic triangles.
pub(super) struct AcousticMesh {
    /// Vertex positions in world meters.
    pub(super) vertices: Vec<Vector3>,
    /// Triangle vertex indices.
    pub(super) triangles: Vec<[u32; 3]>,
}

/// Axis-aligned room surface shared by rendering and acoustic geometry.
pub(super) struct Block {
    /// Center in world meters.
    pub(super) center: Vector3,
    /// Full extent along each world axis.
    pub(super) size: Vector3,
    /// Base display color before directional lighting.
    pub(super) color: Color,
}

/// Editable room geometry settings.
#[derive(Clone, Copy)]
pub(super) struct SceneConfig {
    /// Includes the ceiling in both display and acoustic geometry.
    pub(super) roof_enabled: bool,
    /// Leaves the doorway open and enables its diffraction edges.
    pub(super) door_open: bool,
    /// Includes the enclosing walls.
    pub(super) outer_walls_enabled: bool,
    /// Edge-strip width retained when geometry is rebuilt.
    pub(super) diffraction_radius: f32,
}

/// Builds matching display geometry and acoustic triangles for the editable room.
pub(super) fn build_scene(config: SceneConfig) -> PlaygroundScene {
    let wall_color = Color::new(108, 116, 122, 255);
    let mut blocks = Vec::new();

    if config.outer_walls_enabled {
        blocks.extend([
            Block {
                center: Vector3::new(0.0, WALL_HEIGHT * 0.5, -ROOM_HALF_Z),
                size: Vector3::new(ROOM_HALF_X * 2.0, WALL_HEIGHT, WALL_THICKNESS),
                color: wall_color,
            },
            Block {
                center: Vector3::new(0.0, WALL_HEIGHT * 0.5, ROOM_HALF_Z),
                size: Vector3::new(ROOM_HALF_X * 2.0, WALL_HEIGHT, WALL_THICKNESS),
                color: wall_color,
            },
            Block {
                center: Vector3::new(-ROOM_HALF_X, WALL_HEIGHT * 0.5, 0.0),
                size: Vector3::new(WALL_THICKNESS, WALL_HEIGHT, ROOM_HALF_Z * 2.0),
                color: wall_color,
            },
            Block {
                center: Vector3::new(ROOM_HALF_X, WALL_HEIGHT * 0.5, 0.0),
                size: Vector3::new(WALL_THICKNESS, WALL_HEIGHT, ROOM_HALF_Z * 2.0),
                color: wall_color,
            },
        ]);
    }

    blocks.extend([
        Block {
            center: Vector3::new(
                0.0,
                WALL_HEIGHT * 0.5,
                -(ROOM_HALF_Z + DOOR_HALF_WIDTH) * 0.5,
            ),
            size: Vector3::new(WALL_THICKNESS, WALL_HEIGHT, ROOM_HALF_Z - DOOR_HALF_WIDTH),
            color: Color::new(125, 132, 138, 255),
        },
        Block {
            center: Vector3::new(
                0.0,
                WALL_HEIGHT * 0.5,
                (ROOM_HALF_Z + DOOR_HALF_WIDTH) * 0.5,
            ),
            size: Vector3::new(WALL_THICKNESS, WALL_HEIGHT, ROOM_HALF_Z - DOOR_HALF_WIDTH),
            color: Color::new(125, 132, 138, 255),
        },
    ]);

    if !config.door_open {
        blocks.push(Block {
            center: Vector3::new(0.0, WALL_HEIGHT * 0.5, 0.0),
            size: Vector3::new(WALL_THICKNESS, WALL_HEIGHT, DOOR_HALF_WIDTH * 2.0),
            color: Color::new(116, 124, 130, 255),
        });
    }

    if config.roof_enabled {
        blocks.push(Block {
            center: Vector3::new(0.0, WALL_HEIGHT + ROOF_THICKNESS * 0.5, 0.0),
            size: Vector3::new(ROOM_HALF_X * 2.0, ROOF_THICKNESS, ROOM_HALF_Z * 2.0),
            color: Color::new(92, 99, 106, 255),
        });
    }

    let mut acoustic_mesh = AcousticMesh::new();
    acoustic_mesh.add_floor(ROOM_HALF_X * 2.0, ROOM_HALF_Z * 2.0);
    for block in &blocks {
        acoustic_mesh.add_box(block.center, block.size);
    }

    PlaygroundScene {
        acoustic_mesh,
        blocks,
        diffraction_edges: if config.door_open {
            doorway_diffraction_edges(config.diffraction_radius)
        } else {
            Vec::new()
        },
    }
}

fn doorway_diffraction_edges(radius: f32) -> Vec<DiffractionEdge> {
    let half_thickness = WALL_THICKNESS * 0.5;
    let diagonal = std::f32::consts::FRAC_1_SQRT_2;
    vec![
        DiffractionEdge {
            id: 1,
            start: vec3(-half_thickness, 0.0, -DOOR_HALF_WIDTH),
            end: vec3(-half_thickness, WALL_HEIGHT, -DOOR_HALF_WIDTH),
            bisector_dir: vec3(-diagonal, 0.0, diagonal),
            edge_angle_radians: 1.5 * std::f32::consts::PI,
            diffraction_radius: radius,
        },
        DiffractionEdge {
            id: 2,
            start: vec3(half_thickness, 0.0, -DOOR_HALF_WIDTH),
            end: vec3(half_thickness, WALL_HEIGHT, -DOOR_HALF_WIDTH),
            bisector_dir: vec3(diagonal, 0.0, diagonal),
            edge_angle_radians: 1.5 * std::f32::consts::PI,
            diffraction_radius: radius,
        },
        DiffractionEdge {
            id: 3,
            start: vec3(-half_thickness, 0.0, DOOR_HALF_WIDTH),
            end: vec3(-half_thickness, WALL_HEIGHT, DOOR_HALF_WIDTH),
            bisector_dir: vec3(-diagonal, 0.0, -diagonal),
            edge_angle_radians: 1.5 * std::f32::consts::PI,
            diffraction_radius: radius,
        },
        DiffractionEdge {
            id: 4,
            start: vec3(half_thickness, 0.0, DOOR_HALF_WIDTH),
            end: vec3(half_thickness, WALL_HEIGHT, DOOR_HALF_WIDTH),
            bisector_dir: vec3(diagonal, 0.0, -diagonal),
            edge_angle_radians: 1.5 * std::f32::consts::PI,
            diffraction_radius: radius,
        },
    ]
}

/// Copies the displayed room into an owned acoustic scene.
pub(super) fn scene_description(scene: &PlaygroundScene) -> SceneDescription {
    let indices = scene
        .acoustic_mesh
        .triangles
        .iter()
        .flat_map(|triangle| triangle.iter().copied())
        .collect::<Vec<_>>();

    SceneDescription {
        meshes: vec![MeshAsset {
            id: 1,
            vertices: scene
                .acoustic_mesh
                .vertices
                .iter()
                .copied()
                .map(to_glam)
                .collect(),
            indices,
            opaque: true,
        }],
        materials: vec![Material {
            id: 1,
            absorption: 0.18,
            scattering: 0.12,
            roughness: 0.15,
        }],
        objects: vec![SceneObject {
            id: 1,
            mesh_id: 1,
            material_id: 1,
            transform: Mat4::IDENTITY,
            active: true,
        }],
        diffraction_edges: scene.diffraction_edges.clone(),
    }
}

/// Converts display coordinates to engine coordinates without changing axes.
pub(super) fn to_glam(value: Vector3) -> Vec3 {
    vec3(value.x, value.y, value.z)
}

/// Converts engine coordinates to display coordinates without changing axes.
pub(super) fn from_glam(value: Vec3) -> Vector3 {
    Vector3::new(value.x, value.y, value.z)
}

impl AcousticMesh {
    fn new() -> Self {
        Self {
            vertices: Vec::new(),
            triangles: Vec::new(),
        }
    }

    fn add_floor(&mut self, width: f32, depth: f32) {
        let half_width = width * 0.5;
        let half_depth = depth * 0.5;
        self.add_quad(
            Vector3::new(-half_width, 0.0, -half_depth),
            Vector3::new(half_width, 0.0, -half_depth),
            Vector3::new(half_width, 0.0, half_depth),
            Vector3::new(-half_width, 0.0, half_depth),
        );
    }

    fn add_box(&mut self, center: Vector3, size: Vector3) {
        let half = Vector3::new(size.x * 0.5, size.y * 0.5, size.z * 0.5);
        let min = Vector3::new(center.x - half.x, center.y - half.y, center.z - half.z);
        let max = Vector3::new(center.x + half.x, center.y + half.y, center.z + half.z);

        let p000 = Vector3::new(min.x, min.y, min.z);
        let p001 = Vector3::new(min.x, min.y, max.z);
        let p010 = Vector3::new(min.x, max.y, min.z);
        let p011 = Vector3::new(min.x, max.y, max.z);
        let p100 = Vector3::new(max.x, min.y, min.z);
        let p101 = Vector3::new(max.x, min.y, max.z);
        let p110 = Vector3::new(max.x, max.y, min.z);
        let p111 = Vector3::new(max.x, max.y, max.z);

        self.add_quad(p001, p101, p111, p011);
        self.add_quad(p100, p000, p010, p110);
        self.add_quad(p000, p001, p011, p010);
        self.add_quad(p101, p100, p110, p111);
        self.add_quad(p010, p011, p111, p110);
        self.add_quad(p000, p100, p101, p001);
    }

    fn add_quad(&mut self, a: Vector3, b: Vector3, c: Vector3, d: Vector3) {
        let base = self.vertices.len() as u32;
        self.vertices.extend([a, b, c, d]);
        self.triangles.push([base, base + 1, base + 2]);
        self.triangles.push([base, base + 2, base + 3]);
    }
}
