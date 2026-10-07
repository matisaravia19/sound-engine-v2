//! Room rendering and playback overlays.
use super::geometry::{Block, ROOM_HALF_X, ROOM_HALF_Z, from_glam, to_glam};
use super::poses::{listener_position_name, source_position_name};
use super::{PlaygroundState, playback_label};
use glam::{Vec3, vec3};
use raylib::prelude::*;
use sound_engine::scene::DiffractionEdge;

/// Draws the catalog page containing the current selection.
pub(super) fn draw_scene_menu(d: &mut RaylibDrawHandle<'_>, state: &PlaygroundState) {
    let width = (d.get_screen_width() - 80).clamp(240, 760);
    let rows = scene_menu_rows(d.get_screen_height());
    let start = state.scene_choice / rows * rows;
    d.draw_rectangle(
        32,
        48,
        width + 16,
        rows as i32 * 26 + 104,
        Color::new(20, 28, 36, 250),
    );
    d.draw_text("Elegir escena", 48, 62, 24, Color::RAYWHITE);
    d.draw_text(
        "Arriba/abajo o rueda | Enter o clic | F1: cerrar",
        48,
        92,
        16,
        Color::LIGHTGRAY,
    );
    for (row, entry) in state.catalog.iter().enumerate().skip(start).take(rows) {
        let y = 112 + (row - start) as i32 * 26;
        if row == state.scene_choice {
            d.draw_rectangle(40, y, width, 25, Color::new(40, 83, 105, 255));
        }
        let label = format!(
            "{}{}",
            if row == state.scene_index {
                "* "
            } else {
                "  "
            },
            entry.title
        );
        d.draw_text(&label, 48, y + 4, 17, Color::RAYWHITE);
    }
    d.draw_text(
        &format!(
            "{}/{}  |  {}",
            state.scene_choice + 1,
            state.catalog.len(),
            state.ui.status
        ),
        48,
        120 + rows as i32 * 26,
        14,
        Color::LIGHTGRAY,
    );
}

/// Computes the catalog page size shared by drawing and hit testing.
pub(super) fn scene_menu_rows(height: i32) -> usize {
    ((height - 180) / 26).clamp(3, 18) as usize
}

/// Draws the active room, acoustic edges, source, and listener.
pub(super) fn draw_scene(d: &mut RaylibMode3D<'_, RaylibDrawHandle<'_>>, state: &PlaygroundState) {
    if let Some(scene) = &state.benchmark {
        for triangle in &scene.triangles {
            let [a, b, c] = triangle.map(from_glam);
            if state.ui.benchmark_wireframe {
                d.draw_line_3D(a, b, Color::new(108, 122, 132, 255));
                d.draw_line_3D(b, c, Color::new(108, 122, 132, 255));
                d.draw_line_3D(c, a, Color::new(108, 122, 132, 255));
            } else {
                draw_lit_triangle(d, *triangle, Color::new(164, 177, 186, 255));
            }
        }
        draw_marker(d, state.source_position(), 0.12, Color::GREEN);
        draw_listener(d, state.listener_position(), state.listener_forward());
        if state.ui.show_bisector_planes {
            draw_bisector_planes(d, &scene.description.diffraction_edges);
        }
        return;
    }
    d.draw_plane(
        Vector3::new(0.0, 0.0, 0.0),
        Vector2::new(ROOM_HALF_X * 2.0, ROOM_HALF_Z * 2.0),
        lit_surface_color(Vec3::Y, Color::new(65, 74, 80, 255)),
    );
    d.draw_grid(20, 1.0);

    for block in &state.scene.blocks {
        draw_block(d, block);
    }

    draw_marker(d, state.source_position(), 0.38, Color::GREEN);
    draw_listener(d, state.listener_position(), state.listener_forward());
    if state.ui.show_bisector_planes {
        draw_bisector_planes(d, &state.scene.diffraction_edges);
    }
}

fn draw_block(d: &mut RaylibMode3D<'_, RaylibDrawHandle<'_>>, block: &Block) {
    let center = to_glam(block.center);
    let half_size = to_glam(block.size) * 0.5;
    let corners = [
        vec3(-1.0, -1.0, -1.0),
        vec3(1.0, -1.0, -1.0),
        vec3(1.0, 1.0, -1.0),
        vec3(-1.0, 1.0, -1.0),
        vec3(-1.0, -1.0, 1.0),
        vec3(1.0, -1.0, 1.0),
        vec3(1.0, 1.0, 1.0),
        vec3(-1.0, 1.0, 1.0),
    ]
    .map(|corner| center + corner * half_size);

    for [a, b, c, e] in [
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 4, 7, 3],
        [1, 2, 6, 5],
        [3, 7, 6, 2],
        [0, 1, 5, 4],
    ] {
        draw_lit_triangle(d, [corners[a], corners[b], corners[c]], block.color);
        draw_lit_triangle(d, [corners[a], corners[c], corners[e]], block.color);
    }

    d.draw_cube_wires_v(block.center, block.size, Color::new(78, 91, 102, 255));
}

fn lit_surface_color(normal: Vec3, base: Color) -> Color {
    let toward_light = vec3(-0.45, 0.82, 0.35).normalize();
    let direct = normal.dot(toward_light).max(0.0);
    let sky = (normal.y + 1.0) * 0.5;
    let brightness = 0.38 + 0.52 * direct + 0.10 * sky;

    Color::new(
        (base.r as f32 * brightness) as u8,
        (base.g as f32 * brightness) as u8,
        (base.b as f32 * brightness) as u8,
        base.a,
    )
}

fn draw_lit_triangle(
    d: &mut RaylibMode3D<'_, RaylibDrawHandle<'_>>,
    triangle: [Vec3; 3],
    base: Color,
) {
    let [a, b, c] = triangle;
    let normal = (b - a).cross(c - a).normalize_or_zero();
    let [a, b, c] = triangle.map(from_glam);

    // Benchmark screens have no thickness; shade each visible side with its own normal.
    d.draw_triangle3D(a, b, c, lit_surface_color(normal, base));
    d.draw_triangle3D(c, b, a, lit_surface_color(-normal, base));
}

fn draw_marker(
    d: &mut RaylibMode3D<'_, RaylibDrawHandle<'_>>,
    position: Vector3,
    radius: f32,
    color: Color,
) {
    d.draw_sphere(position, radius, color);
    d.draw_sphere_wires(position, radius, 16, 16, Color::WHITE);
    d.draw_line_3D(
        Vector3::new(position.x, 0.0, position.z),
        Vector3::new(position.x, position.y, position.z),
        color,
    );
}

fn draw_listener(d: &mut RaylibMode3D<'_, RaylibDrawHandle<'_>>, position: Vector3, forward: Vec3) {
    draw_marker(d, position, 0.38, Color::RED);
    let origin = to_glam(position);
    let tip = origin + forward.normalize_or_zero() * 1.1;
    draw_plane_line(d, origin, tip, Color::new(255, 150, 150, 255));
    d.draw_sphere(from_glam(tip), 0.11, Color::new(255, 150, 150, 255));
}

fn draw_bisector_planes(d: &mut RaylibMode3D<'_, RaylibDrawHandle<'_>>, edges: &[DiffractionEdge]) {
    for edge in edges {
        let start = edge.start;
        let end = edge.end;
        let plane_dir = edge.bisector_dir.normalize_or_zero() * edge.diffraction_radius;
        let far_start = start + plane_dir;
        let far_end = end + plane_dir;

        draw_plane_line(d, start, far_start, Color::SKYBLUE);
        draw_plane_line(d, end, far_end, Color::SKYBLUE);
        draw_plane_line(d, far_start, far_end, Color::SKYBLUE);

        for fraction in [0.25, 0.5, 0.75] {
            let near = start.lerp(end, fraction);
            let far = near + plane_dir;
            draw_plane_line(d, near, far, Color::new(88, 180, 220, 255));
        }
    }
}

fn draw_plane_line(
    d: &mut RaylibMode3D<'_, RaylibDrawHandle<'_>>,
    start: Vec3,
    end: Vec3,
    color: Color,
) {
    d.draw_line_3D(from_glam(start), from_glam(end), color);
}

/// Draws playback controls and the current scene status.
pub(super) fn draw_overlay(d: &mut RaylibDrawHandle, camera: &Camera3D, state: &PlaygroundState) {
    let fps = d.get_fps();
    let active_listener = state.active_listener_pose(camera).position;
    d.draw_rectangle(12, 12, 1_040, 284, Color::new(12, 16, 20, 220));

    let button_color = Color::new(42, 128, 74, 255);
    d.draw_rectangle(24, 24, 160, 40, button_color);
    d.draw_rectangle_lines(24, 24, 160, 40, Color::new(190, 230, 205, 255));
    d.draw_text("Start Sound", 42, 35, 18, Color::RAYWHITE);

    let toggle_color = if state.ui.show_bisector_planes {
        Color::new(35, 118, 150, 255)
    } else {
        Color::new(46, 54, 62, 255)
    };
    d.draw_rectangle(24, 70, 230, 34, toggle_color);
    d.draw_rectangle_lines(24, 70, 230, 34, Color::new(160, 205, 225, 255));
    d.draw_text("Bisector Planes", 42, 78, 17, Color::RAYWHITE);

    let audio_color = if state.ui.full_acoustics {
        Color::new(35, 118, 150, 255)
    } else {
        Color::new(115, 82, 35, 255)
    };
    d.draw_rectangle(278, 70, 310, 34, audio_color);
    d.draw_rectangle_lines(278, 70, 310, 34, Color::new(160, 205, 225, 255));
    d.draw_text(
        &format!("T: {}", playback_label(state)),
        292,
        78,
        17,
        Color::RAYWHITE,
    );

    d.draw_text(
        "F1: scenes | Space: sound | B: planes | L: listener",
        278,
        34,
        18,
        Color::LIGHTGRAY,
    );
    d.draw_text(
        &format!(
            "camera: {:.1}, {:.1}, {:.1} | FPS: {fps}",
            camera.position.x, camera.position.y, camera.position.z
        ),
        24,
        114,
        18,
        Color::LIGHTGRAY,
    );
    d.draw_text(
        &format!(
            "source: {:.1}, {:.1}, {:.1} | active listener ({}): {:.1}, {:.1}, {:.1}",
            state.source_position().x,
            state.source_position().y,
            state.source_position().z,
            state.listener_mode_label(),
            active_listener.x,
            active_listener.y,
            active_listener.z
        ),
        24,
        138,
        18,
        Color::LIGHTGRAY,
    );
    d.draw_text(
        &if let Some(scene) = &state.benchmark {
            format!(
                "3: next pair | 5: previous | {} ({}/{})",
                scene.pairs[state.pair_index].name,
                state.pair_index + 1,
                scene.pairs.len()
            )
        } else {
            format!(
                "1 roof: {} | 2 door: {} | 3 red pos: {} | 4 red rot: {}",
                on_off(state.scene_config.roof_enabled),
                if state.scene_config.door_open {
                    "open"
                } else {
                    "closed"
                },
                listener_position_name(state.listener_index),
                state.listener_rotation_label()
            )
        },
        24,
        162,
        18,
        Color::LIGHTGRAY,
    );
    d.draw_text(
        &if state.benchmark.is_some() {
            format!(
                "{} | H: view pair | V: wire/solid | interactive budget",
                state.catalog[state.scene_index].title
            )
        } else {
            format!(
                "5 source: {} | 6 outer walls: {}",
                source_position_name(state.source_index),
                on_off(state.scene_config.outer_walls_enabled)
            )
        },
        24,
        186,
        18,
        Color::LIGHTGRAY,
    );
    d.draw_text(
        &format!(
            "7 red spin: {} | L listener: {} | bisector planes: {}",
            on_off(state.listener_spin_enabled),
            state.listener_mode_label(),
            on_off(state.ui.show_bisector_planes)
        ),
        24,
        210,
        18,
        Color::LIGHTGRAY,
    );
    d.draw_text("Source: omnidirectional", 24, 234, 18, Color::GRAY);
    d.draw_text(&state.ui.status, 24, 258, 18, Color::RAYWHITE);
}

fn on_off(value: bool) -> &'static str {
    if value {
        "on"
    } else {
        "off"
    }
}
