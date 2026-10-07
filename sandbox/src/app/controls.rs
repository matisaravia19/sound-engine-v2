//! Input handling and synchronization of interactive scene changes.
use super::audio::AudioSession;
use super::geometry::{from_glam, scene_description, to_glam};
use super::layout::{BISECTOR_BUTTON, PLAYBACK_MODE_BUTTON, START_SOUND_BUTTON, SceneMenuLayout};
use super::listener::{ListenerController, ListenerMode};
use super::poses::{listener_position_count, source_position_count};
use super::{PlaygroundState, empty_acoustic_scene, playback_label, playback_scene};
use glam::vec3;
use raylib::prelude::*;
use std::time::Instant;

/// Frames the current source and receiver in the display camera.
pub(super) fn frame_selected_pair(state: &PlaygroundState, camera: &mut Camera3D) {
    let source = to_glam(state.source_position());
    let listener = to_glam(state.listener_position());
    let center = (source + listener) * 0.5;
    let distance = source.distance(listener).max(2.0);
    camera.target = from_glam(center);
    camera.position = from_glam(center + vec3(0.8, 0.65, 1.0) * distance);
    camera.up = Vector3::new(0.0, 1.0, 0.0);
}

/// Handles scene selection and synchronizes the newly selected scene.
pub(super) fn handle_scene_menu(
    rl: &mut RaylibHandle,
    audio: &mut AudioSession,
    state: &mut PlaygroundState,
    camera: &mut Camera3D,
) {
    if rl.is_key_pressed(KeyboardKey::KEY_F1) {
        state.scene_menu = !state.scene_menu;
        state.scene_choice = state.scene_index;
        if state.scene_menu {
            rl.enable_cursor();
        } else {
            rl.disable_cursor();
        }
    }
    if !state.scene_menu {
        if rl.is_key_pressed(KeyboardKey::KEY_H) {
            frame_selected_pair(state, camera);
            if state.listener.mode() == ListenerMode::Camera {
                sync_camera_listener_if_changed(audio, state, camera, true);
            }
        }
        return;
    }
    let count = state.catalog.len();
    let wheel = rl.get_mouse_wheel_move();
    if rl.is_key_pressed(KeyboardKey::KEY_DOWN) || wheel < 0.0 {
        state.scene_choice = (state.scene_choice + 1) % count;
    }
    if rl.is_key_pressed(KeyboardKey::KEY_UP) || wheel > 0.0 {
        state.scene_choice = (state.scene_choice + count - 1) % count;
    }
    let layout = SceneMenuLayout::new(
        rl.get_screen_width(),
        rl.get_screen_height(),
        state.scene_choice,
        count,
    );
    let mut choose = rl.is_key_pressed(KeyboardKey::KEY_ENTER);
    if rl.is_mouse_button_pressed(MouseButton::MOUSE_BUTTON_LEFT)
        && let Some(index) = layout.hit_test(rl.get_mouse_position())
    {
        state.scene_choice = index;
        choose = true;
    }
    if !choose {
        return;
    }
    match select_scene(audio, state, camera, state.scene_choice) {
        Err(e) => state.ui.status = format!("Cannot load scene: {e}"),
        Ok(()) => {
            state.scene_menu = false;
            rl.disable_cursor();
        }
    }
}

fn select_scene(
    audio: &mut AudioSession,
    state: &mut PlaygroundState,
    camera: &mut Camera3D,
    index: usize,
) -> Result<(), String> {
    let entry = state.catalog.get(index).ok_or("Unknown scene index")?;
    let benchmark = if index == 0 {
        None
    } else {
        Some(entry.load()?)
    };
    let description = if state.ui.full_acoustics {
        benchmark
            .as_ref()
            .map(|b| b.description.clone())
            .unwrap_or_else(|| scene_description(&state.scene))
    } else {
        empty_acoustic_scene()
    };
    audio
        .engine
        .load_scene(description)
        .map_err(|e| e.to_string())?;
    state.benchmark = benchmark;
    state.scene_index = index;
    state.pair_index = 0;
    state.listener = ListenerController::default();
    frame_selected_pair(state, camera);
    sync_source(audio, state);
    sync_red_sphere_listener(audio, state, "scene");
    if let Some(scene) = &state.benchmark {
        state.ui.status = format!(
            "{} diffraction edges ({} inferred from room geometry)",
            scene.description.diffraction_edges.len(),
            scene.inferred_edges,
        );
    }
    Ok(())
}

/// Applies room, source, and listener controls to the active audio.
pub(super) fn handle_config_keys(
    rl: &RaylibHandle,
    audio: &mut AudioSession,
    state: &mut PlaygroundState,
    camera: &Camera3D,
) {
    if rl.is_key_pressed(KeyboardKey::KEY_T) || button_clicked(rl, PLAYBACK_MODE_BUTTON) {
        state.ui.full_acoustics = !state.ui.full_acoustics;
        match audio.engine.load_scene(playback_scene(state)) {
            Ok(_) => {
                state.ui.status = format!("Playback: {}", playback_label(state));
            }
            Err(error) => {
                state.ui.full_acoustics = !state.ui.full_acoustics;
                state.ui.status = format!("Playback switch failed: {error}");
            }
        }
    }

    if rl.is_key_pressed(KeyboardKey::KEY_V) {
        state.ui.benchmark_wireframe = !state.ui.benchmark_wireframe;
    }
    if state.benchmark.is_none() && rl.is_key_pressed(KeyboardKey::KEY_ONE) {
        state.scene_config.roof_enabled = !state.scene_config.roof_enabled;
        reload_scene(audio, state, "roof");
    }

    if state.benchmark.is_none() && rl.is_key_pressed(KeyboardKey::KEY_TWO) {
        state.scene_config.door_open = !state.scene_config.door_open;
        reload_scene(audio, state, "door");
    }

    if rl.is_key_pressed(KeyboardKey::KEY_THREE) {
        if let Some(scene) = &state.benchmark {
            state.pair_index = next_index(state.pair_index, scene.pairs.len());
            sync_source(audio, state);
        } else {
            state.listener_index = next_index(state.listener_index, listener_position_count());
        }
        sync_red_sphere_listener(audio, state, "red sphere position");
    }

    if rl.is_key_pressed(KeyboardKey::KEY_FOUR) {
        state.listener.next_orientation();
        sync_red_sphere_listener(audio, state, "red sphere rotation");
    }

    if rl.is_key_pressed(KeyboardKey::KEY_FIVE) {
        if let Some(scene) = &state.benchmark {
            state.pair_index = (state.pair_index + scene.pairs.len() - 1) % scene.pairs.len();
            sync_red_sphere_listener(audio, state, "benchmark pair");
        } else {
            state.source_index = next_index(state.source_index, source_position_count());
        }
        sync_source(audio, state);
    }

    if state.benchmark.is_none() && rl.is_key_pressed(KeyboardKey::KEY_SIX) {
        state.scene_config.outer_walls_enabled = !state.scene_config.outer_walls_enabled;
        reload_scene(audio, state, "outer walls");
    }

    if rl.is_key_pressed(KeyboardKey::KEY_SEVEN) {
        state.listener.toggle_spin();
        sync_red_sphere_listener(audio, state, "red sphere spin");
    }
    if rl.is_key_pressed(KeyboardKey::KEY_L) {
        state.listener.toggle_mode();
        match state.listener.mode() {
            ListenerMode::RedSphere => sync_red_sphere_listener(audio, state, "listener mode"),
            ListenerMode::Camera => sync_camera_listener_if_changed(audio, state, camera, true),
        }
    }

    if bisector_plane_toggle_requested(rl) {
        state.ui.show_bisector_planes = !state.ui.show_bisector_planes;
        state.ui.status = if state.ui.show_bisector_planes {
            "Showing bisector planes".to_owned()
        } else {
            "Hiding bisector planes".to_owned()
        };
    }
}

fn sync_source(audio: &mut AudioSession, state: &mut PlaygroundState) {
    match audio.update_source(state.point_source()) {
        Ok(()) => {
            state.ui.status = format!(
                "Changed source to {}; active voices will refresh IR if needed",
                state.position_label(true)
            );
        }
        Err(error) => {
            state.ui.status = format!("Source update error: {error}");
        }
    }
}

fn reload_scene(audio: &mut AudioSession, state: &mut PlaygroundState, label: &str) {
    state.rebuild_scene();
    match audio.engine.load_scene(playback_scene(state)) {
        Ok(version) => {
            state.ui.status = format!("{label} changed; scene version {version}");
        }
        Err(error) => {
            state.ui.status = format!("Scene reload error: {error}");
        }
    }
}

/// Updates the audio from the receiver marker and reports failures.
pub(super) fn sync_red_sphere_listener(
    audio: &mut AudioSession,
    state: &mut PlaygroundState,
    label: &str,
) {
    if state.listener.mode() == ListenerMode::Camera {
        state.ui.status = format!(
            "Changed {label}: {} / {}; camera remains the active listener",
            state.position_label(false),
            state.listener.rotation_label()
        );
        return;
    }

    match audio
        .engine
        .update_listener_pose(state.red_sphere_listener_pose())
    {
        Ok(()) => {
            state.ui.status = format!(
                "Changed {label}: {} / {}",
                state.position_label(false),
                state.listener.rotation_label()
            );
        }
        Err(error) => {
            state.ui.status = format!("Listener update error: {error}");
        }
    }
}

/// Coalesces small or frequent camera movements before updating the listener.
pub(super) fn sync_camera_listener_if_changed(
    audio: &mut AudioSession,
    state: &mut PlaygroundState,
    camera: &Camera3D,
    announce: bool,
) {
    let Some(pose) = state
        .listener
        .pending_camera_pose(camera, Instant::now(), announce)
    else {
        return;
    };

    match audio.engine.update_listener_pose(pose) {
        Ok(()) => {
            state.listener.camera_synced(pose, Instant::now());
            if announce {
                state.ui.status = "Camera is now the active listener".to_owned();
            }
        }
        Err(error) => {
            state.ui.status = format!("Camera listener update error: {error}");
        }
    }
}

fn next_index(index: usize, len: usize) -> usize {
    (index + 1) % len
}

/// Recognizes the keyboard and mouse actions that start playback.
pub(super) fn start_sound_requested(rl: &RaylibHandle) -> bool {
    rl.is_key_pressed(KeyboardKey::KEY_SPACE) || button_clicked(rl, START_SOUND_BUTTON)
}

fn bisector_plane_toggle_requested(rl: &RaylibHandle) -> bool {
    rl.is_key_pressed(KeyboardKey::KEY_B) || button_clicked(rl, BISECTOR_BUTTON)
}

fn button_clicked(rl: &RaylibHandle, bounds: Rectangle) -> bool {
    rl.is_mouse_button_pressed(MouseButton::MOUSE_BUTTON_LEFT)
        && bounds.check_collision_point_rec(rl.get_mouse_position())
}
