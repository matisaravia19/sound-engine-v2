//! Input handling and synchronization of interactive scene changes.
use super::PlaygroundState;
use super::audio::AudioSession;
use super::geometry::{SceneConfig, from_glam, to_glam};
use super::layout::{BISECTOR_BUTTON, PLAYBACK_MODE_BUTTON, START_SOUND_BUTTON, SceneMenuLayout};
use super::listener::{ListenerController, ListenerMode};
use glam::vec3;
use raylib::prelude::*;
use sound_engine::core::error::SoundResult;
use std::time::Instant;

/// Frames the current source and receiver in the display camera.
pub(super) fn frame_selected_pair(state: &PlaygroundState, camera: &mut Camera3D) {
    let source = to_glam(state.scene.source_position());
    let listener = to_glam(state.scene.listener_position());
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
        state.scene_choice = state.scene.index();
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
    let count = state.scene.catalog().len();
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
) -> SoundResult<()> {
    state
        .scene
        .select(index, |description| audio.engine.load_scene(description))?;
    state.listener = ListenerController::default();
    frame_selected_pair(state, camera);
    sync_source(audio, state);
    sync_red_sphere_listener(audio, state, "scene");
    if let Some(scene) = state.scene.benchmark() {
        state.ui.status = format!(
            "{} diffraction edges ({} inferred from room geometry)",
            scene.loaded.description.diffraction_edges.len(),
            scene.loaded.inferred_edges,
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
        match state
            .scene
            .toggle_playback(|description| audio.engine.load_scene(description))
        {
            Ok(_) => {
                state.ui.status = format!("Playback: {}", state.scene.playback_mode().label());
            }
            Err(error) => {
                state.ui.status = format!("Playback switch failed: {error}");
            }
        }
    }

    if rl.is_key_pressed(KeyboardKey::KEY_V) {
        state.ui.benchmark_wireframe = !state.ui.benchmark_wireframe;
    }
    if state.scene.benchmark().is_none() && rl.is_key_pressed(KeyboardKey::KEY_ONE) {
        let mut config = state.scene.room().config;
        config.roof_enabled = !config.roof_enabled;
        reload_scene(audio, state, config, "roof");
    }

    if state.scene.benchmark().is_none() && rl.is_key_pressed(KeyboardKey::KEY_TWO) {
        let mut config = state.scene.room().config;
        config.door_open = !config.door_open;
        reload_scene(audio, state, config, "door");
    }

    if rl.is_key_pressed(KeyboardKey::KEY_THREE) {
        state.scene.next_listener();
        if state.scene.benchmark().is_some() {
            sync_source(audio, state);
        }
        sync_red_sphere_listener(audio, state, "red sphere position");
    }

    if rl.is_key_pressed(KeyboardKey::KEY_FOUR) {
        state.listener.next_orientation();
        sync_red_sphere_listener(audio, state, "red sphere rotation");
    }

    if rl.is_key_pressed(KeyboardKey::KEY_FIVE) {
        state.scene.next_source();
        if state.scene.benchmark().is_some() {
            sync_red_sphere_listener(audio, state, "benchmark pair");
        }
        sync_source(audio, state);
    }

    if state.scene.benchmark().is_none() && rl.is_key_pressed(KeyboardKey::KEY_SIX) {
        let mut config = state.scene.room().config;
        config.outer_walls_enabled = !config.outer_walls_enabled;
        reload_scene(audio, state, config, "outer walls");
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
                state.scene.source_label()
            );
        }
        Err(error) => {
            state.ui.status = format!("Source update error: {error}");
        }
    }
}

fn reload_scene(
    audio: &mut AudioSession,
    state: &mut PlaygroundState,
    config: SceneConfig,
    label: &str,
) {
    match state
        .scene
        .update_room(config, |description| audio.engine.load_scene(description))
    {
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
            state.scene.listener_label(),
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
                state.scene.listener_label(),
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
