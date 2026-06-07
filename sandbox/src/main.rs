use raylib::prelude::*;

struct Scene {
    acoustic_mesh: AcousticMesh,
    blocks: Vec<Block>,
    markers: Vec<Marker>,
}

struct AcousticMesh {
    vertices: Vec<Vector3>,
    triangles: Vec<[u32; 3]>,
}

struct Block {
    center: Vector3,
    size: Vector3,
    color: Color,
}

struct Marker {
    position: Vector3,
    color: Color,
}

fn main() {
    let (mut rl, thread) = raylib::init()
        .size(1280, 720)
        .title("sound-engine sandbox")
        .resizable()
        .msaa_4x()
        .build();

    rl.set_target_fps(60);
    rl.disable_cursor();

    let mut camera = Camera3D::perspective(
        Vector3::new(8.0, 5.0, 8.0),
        Vector3::new(0.0, 1.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        60.0,
    );
    let scene = build_scene();

    while !rl.window_should_close() {
        rl.update_camera(&mut camera, CameraMode::CAMERA_FREE);
        update_sound_engine_placeholder(&camera, &scene.acoustic_mesh);

        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::new(18, 22, 26, 255));

        {
            let mut d3 = d.begin_mode3D(camera);
            draw_scene(&mut d3, &scene);
        }

        draw_overlay(&mut d, &camera);
    }
}

fn update_sound_engine_placeholder(_camera: &Camera3D, _mesh: &AcousticMesh) {
    // Later: pass listener pose plus mesh.vertices/mesh.triangles into sound-engine here.
}

fn build_scene() -> Scene {
    let blocks = vec![
        Block {
            center: Vector3::new(0.0, 1.5, -8.0),
            size: Vector3::new(12.0, 3.0, 0.4),
            color: Color::DARKGRAY,
        },
        Block {
            center: Vector3::new(-6.0, 1.5, 0.0),
            size: Vector3::new(0.4, 3.0, 12.0),
            color: Color::DARKGRAY,
        },
        Block {
            center: Vector3::new(4.0, 0.75, 3.0),
            size: Vector3::new(3.0, 1.5, 2.0),
            color: Color::GRAY,
        },
        Block {
            center: Vector3::new(-2.0, 1.0, -2.5),
            size: Vector3::new(2.0, 2.0, 2.0),
            color: Color::BROWN,
        },
        Block {
            center: Vector3::new(-4.0, 1.5, -5.0),
            size: Vector3::new(0.7, 3.0, 0.7),
            color: Color::new(82, 98, 112, 255),
        },
        Block {
            center: Vector3::new(2.5, 1.5, -5.0),
            size: Vector3::new(0.7, 3.0, 0.7),
            color: Color::new(82, 98, 112, 255),
        },
        Block {
            center: Vector3::new(-4.0, 1.5, 2.0),
            size: Vector3::new(0.7, 3.0, 0.7),
            color: Color::new(82, 98, 112, 255),
        },
        Block {
            center: Vector3::new(2.5, 1.5, 2.0),
            size: Vector3::new(0.7, 3.0, 0.7),
            color: Color::new(82, 98, 112, 255),
        },
    ];
    let markers = vec![
        Marker {
            position: Vector3::new(-3.5, 0.45, 4.0),
            color: Color::GOLD,
        },
        Marker {
            position: Vector3::new(1.0, 0.45, -3.5),
            color: Color::SKYBLUE,
        },
        Marker {
            position: Vector3::new(5.0, 0.45, -1.0),
            color: Color::LIME,
        },
    ];

    let mut acoustic_mesh = AcousticMesh::new();
    acoustic_mesh.add_floor(32.0);
    for block in &blocks {
        acoustic_mesh.add_box(block.center, block.size);
    }

    Scene {
        acoustic_mesh,
        blocks,
        markers,
    }
}

impl AcousticMesh {
    fn new() -> Self {
        Self {
            vertices: Vec::new(),
            triangles: Vec::new(),
        }
    }

    fn add_floor(&mut self, size: f32) {
        let half = size * 0.5;
        self.add_quad(
            Vector3::new(-half, 0.0, -half),
            Vector3::new(half, 0.0, -half),
            Vector3::new(half, 0.0, half),
            Vector3::new(-half, 0.0, half),
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

fn draw_scene(d: &mut RaylibMode3D<'_, RaylibDrawHandle<'_>>, scene: &Scene) {
    d.draw_plane(
        Vector3::new(0.0, 0.0, 0.0),
        Vector2::new(32.0, 32.0),
        Color::new(36, 43, 48, 255),
    );
    d.draw_grid(32, 1.0);

    for block in &scene.blocks {
        draw_block(d, block);
    }

    for marker in &scene.markers {
        draw_marker(d, marker);
    }

    d.draw_line_3D(
        Vector3::new(-16.0, 0.02, 0.0),
        Vector3::new(16.0, 0.02, 0.0),
        Color::RED,
    );
    d.draw_line_3D(
        Vector3::new(0.0, 0.02, -16.0),
        Vector3::new(0.0, 0.02, 16.0),
        Color::BLUE,
    );
}

fn draw_block(d: &mut RaylibMode3D<'_, RaylibDrawHandle<'_>>, block: &Block) {
    d.draw_cube_v(block.center, block.size, block.color);
    d.draw_cube_wires_v(block.center, block.size, Color::new(190, 198, 204, 255));
}

fn draw_marker(d: &mut RaylibMode3D<'_, RaylibDrawHandle<'_>>, marker: &Marker) {
    d.draw_sphere(marker.position, 0.45, marker.color);
    d.draw_sphere_wires(marker.position, 0.45, 12, 12, Color::WHITE);
    d.draw_line_3D(
        Vector3::new(marker.position.x, 0.0, marker.position.z),
        Vector3::new(marker.position.x, marker.position.y, marker.position.z),
        marker.color,
    );
}

fn draw_overlay(d: &mut RaylibDrawHandle, camera: &Camera3D) {
    let fps = d.get_fps();
    d.draw_rectangle(12, 12, 440, 84, Color::new(12, 16, 20, 210));
    d.draw_text(
        "WASD + mouse: move camera | Space/Ctrl: up/down",
        24,
        24,
        18,
        Color::RAYWHITE,
    );
    d.draw_text("Esc: close sandbox", 24, 48, 18, Color::LIGHTGRAY);
    d.draw_text(
        &format!(
            "FPS: {fps} | camera: {:.1}, {:.1}, {:.1}",
            camera.position.x, camera.position.y, camera.position.z
        ),
        24,
        72,
        18,
        Color::LIGHTGRAY,
    );
}
