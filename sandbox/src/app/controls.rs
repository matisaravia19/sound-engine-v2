//! Input handling and synchronization of interactive scene changes.
use super::geometry::{from_glam, scene_description, to_glam};
use super::poses::{
    camera_listener_pose, listener_poses_match, listener_position_count, listener_rotation_count,
    source_position_count,
};
use super::render::scene_menu_rows;
use super::{ListenerMode, PlaygroundState, empty_acoustic_scene, playback_label, playback_scene};
use glam::vec3;
use raylib::prelude::*;
use sound_engine::SoundEngine;
use sound_engine::core::error::SoundResult;
use std::time::{Duration, Instant};

const CAMERA_LISTENER_UPDATE_INTERVAL: Duration = Duration::from_millis(50);

/// Synchronizes the listener before starting a tracked spatial voice.
pub(super) fn play_source_sound(
    engine: &mut SoundEngine,
    sound_id: u64,
    state: &PlaygroundState,
    camera: &Camera3D,
    playback_gain: f32,
) -> SoundResult<u64> {
    engine.set_listener_pose(state.active_listener_pose(camera))?;
    let source_id = state.sound_source_id.ok_or_else(|| {
        sound_engine::core::error::SoundError::invalid_state(
            "sandbox sound source is not initialized",
        )
    })?;
    engine.play_sound_from_source(sound_id, source_id, playback_gain)
}

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
    engine: &mut SoundEngine,
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
            if state.listener_mode == ListenerMode::Camera {
                sync_camera_listener_if_changed(engine, state, camera, true);
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
    let rows = scene_menu_rows(rl.get_screen_height());
    let start = state.scene_choice / rows * rows;
    let mouse = rl.get_mouse_position();
    let mut choose = rl.is_key_pressed(KeyboardKey::KEY_ENTER);
    if rl.is_mouse_button_pressed(MouseButton::MOUSE_BUTTON_LEFT)
        && mouse.x >= 40.0
        && mouse.x < (rl.get_screen_width() - 40).min(800) as f32
        && mouse.y >= 112.0
    {
        let row = ((mouse.y - 112.0) / 26.0) as usize;
        if row < rows && start + row < count {
            state.scene_choice = start + row;
            choose = true;
        }
    }
    if !choose {
        return;
    }
    match select_scene(engine, state, camera, state.scene_choice) {
        Err(e) => state.ui.status = format!("Cannot load scene: {e}"),
        Ok(()) => {
            state.scene_menu = false;
            rl.disable_cursor();
        }
    }
}

fn select_scene(
    engine: &mut SoundEngine,
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
    engine.load_scene(description).map_err(|e| e.to_string())?;
    state.benchmark = benchmark;
    state.scene_index = index;
    state.pair_index = 0;
    state.listener_mode = ListenerMode::RedSphere;
    state.listener_spin_enabled = false;
    state.listener_rotation_index = 0;
    state.last_camera_listener_pose = None;
    state.last_camera_listener_sync = None;
    frame_selected_pair(state, camera);
    sync_source(engine, state);
    sync_red_sphere_listener(engine, state, "scene");
    if let Some(scene) = &state.benchmark {
        state.ui.status = format!(
            "{} diffraction edges ({} inferred from room geometry)",
            scene.description.diffraction_edges.len(),
            scene.inferred_edges,
        );
    }
    Ok(())
}

/// Applies room, source, and listener controls to the active engine.
pub(super) fn handle_config_keys(
    rl: &RaylibHandle,
    engine: &mut SoundEngine,
    state: &mut PlaygroundState,
    camera: &Camera3D,
) {
    let mouse = rl.get_mouse_position();
    let audio_button = mouse.x >= 278.0 && mouse.x <= 588.0 && mouse.y >= 70.0 && mouse.y <= 104.0;
    if rl.is_key_pressed(KeyboardKey::KEY_T)
        || (rl.is_mouse_button_pressed(MouseButton::MOUSE_BUTTON_LEFT) && audio_button)
    {
        state.ui.full_acoustics = !state.ui.full_acoustics;
        match engine.load_scene(playback_scene(state)) {
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
        reload_scene(engine, state, "roof");
    }

    if state.benchmark.is_none() && rl.is_key_pressed(KeyboardKey::KEY_TWO) {
        state.scene_config.door_open = !state.scene_config.door_open;
        reload_scene(engine, state, "door");
    }

    if rl.is_key_pressed(KeyboardKey::KEY_THREE) {
        if let Some(scene) = &state.benchmark {
            state.pair_index = next_index(state.pair_index, scene.pairs.len());
            sync_source(engine, state);
        } else {
            state.listener_index = next_index(state.listener_index, listener_position_count());
        }
        sync_red_sphere_listener(engine, state, "red sphere position");
    }

    if rl.is_key_pressed(KeyboardKey::KEY_FOUR) {
        state.listener_spin_enabled = false;
        state.listener_rotation_index =
            next_index(state.listener_rotation_index, listener_rotation_count());
        sync_red_sphere_listener(engine, state, "red sphere rotation");
    }

    if rl.is_key_pressed(KeyboardKey::KEY_FIVE) {
        if let Some(scene) = &state.benchmark {
            state.pair_index = (state.pair_index + scene.pairs.len() - 1) % scene.pairs.len();
            sync_red_sphere_listener(engine, state, "benchmark pair");
        } else {
            state.source_index = next_index(state.source_index, source_position_count());
        }
        sync_source(engine, state);
    }

    if state.benchmark.is_none() && rl.is_key_pressed(KeyboardKey::KEY_SIX) {
        state.scene_config.outer_walls_enabled = !state.scene_config.outer_walls_enabled;
        reload_scene(engine, state, "outer walls");
    }

    if rl.is_key_pressed(KeyboardKey::KEY_SEVEN) {
        state.toggle_listener_spin();
        sync_red_sphere_listener(engine, state, "red sphere spin");
    }
    if rl.is_key_pressed(KeyboardKey::KEY_L) {
        state.toggle_listener_mode();
        match state.listener_mode {
            ListenerMode::RedSphere => sync_red_sphere_listener(engine, state, "listener mode"),
            ListenerMode::Camera => sync_camera_listener_if_changed(engine, state, camera, true),
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

fn sync_source(engine: &mut SoundEngine, state: &mut PlaygroundState) {
    let Some(source_id) = state.sound_source_id else {
        state.ui.status = "Source update error: sandbox sound source is not initialized".to_owned();
        return;
    };

    match engine.update_sound_source(source_id, state.point_source()) {
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

fn reload_scene(engine: &mut SoundEngine, state: &mut PlaygroundState, label: &str) {
    state.rebuild_scene();
    match engine.load_scene(playback_scene(state)) {
        Ok(version) => {
            state.ui.status = format!("{label} changed; scene version {version}");
        }
        Err(error) => {
            state.ui.status = format!("Scene reload error: {error}");
        }
    }
}

/// Updates the engine from the receiver marker and reports failures.
pub(super) fn sync_red_sphere_listener(
    engine: &mut SoundEngine,
    state: &mut PlaygroundState,
    label: &str,
) {
    if state.listener_mode == ListenerMode::Camera {
        state.ui.status = format!(
            "Changed {label}: {} / {}; camera remains the active listener",
            state.position_label(false),
            state.listener_rotation_label()
        );
        return;
    }

    match engine.update_listener_pose(state.red_sphere_listener_pose()) {
        Ok(()) => {
            state.ui.status = format!(
                "Changed {label}: {} / {}",
                state.position_label(false),
                state.listener_rotation_label()
            );
        }
        Err(error) => {
            state.ui.status = format!("Listener update error: {error}");
        }
    }
}

/// Coalesces small or frequent camera movements before updating the listener.
pub(super) fn sync_camera_listener_if_changed(
    engine: &mut SoundEngine,
    state: &mut PlaygroundState,
    camera: &Camera3D,
    announce: bool,
) {
    if !announce
        && state
            .last_camera_listener_sync
            .is_some_and(|last_sync| last_sync.elapsed() < CAMERA_LISTENER_UPDATE_INTERVAL)
    {
        return;
    }

    let pose = camera_listener_pose(camera);
    if state
        .last_camera_listener_pose
        .is_some_and(|previous| listener_poses_match(previous, pose))
    {
        return;
    }

    match engine.update_listener_pose(pose) {
        Ok(()) => {
            state.last_camera_listener_pose = Some(pose);
            state.last_camera_listener_sync = Some(Instant::now());
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
    rl.is_key_pressed(KeyboardKey::KEY_SPACE)
        || (rl.is_mouse_button_pressed(MouseButton::MOUSE_BUTTON_LEFT) && mouse_in_start_button(rl))
}

fn bisector_plane_toggle_requested(rl: &RaylibHandle) -> bool {
    rl.is_key_pressed(KeyboardKey::KEY_B)
        || (rl.is_mouse_button_pressed(MouseButton::MOUSE_BUTTON_LEFT)
            && mouse_in_bisector_toggle(rl))
}

fn mouse_in_start_button(rl: &RaylibHandle) -> bool {
    let mouse = rl.get_mouse_position();
    mouse.x >= 24.0 && mouse.x <= 184.0 && mouse.y >= 24.0 && mouse.y <= 64.0
}

fn mouse_in_bisector_toggle(rl: &RaylibHandle) -> bool {
    let mouse = rl.get_mouse_position();
    mouse.x >= 24.0 && mouse.x <= 254.0 && mouse.y >= 70.0 && mouse.y <= 104.0
}
