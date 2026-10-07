//! Interactive playground state and frame loop.
mod audio;
mod config;
mod controls;
mod geometry;
mod poses;
mod render;

use crate::scenes;
use audio::AudioSession;
use config::Options;
use controls::{
    frame_selected_pair, handle_config_keys, handle_scene_menu, start_sound_requested,
    sync_camera_listener_if_changed, sync_red_sphere_listener,
};
use geometry::{PlaygroundScene, SceneConfig, build_scene, from_glam, scene_description, to_glam};
use glam::Vec3;
use poses::{
    angle_from_forward, camera_listener_pose, listener_forward, listener_position,
    listener_position_name, listener_rotation_name, listener_spin_forward, source_position,
    source_position_name,
};
use raylib::prelude::*;
use render::{draw_overlay, draw_scene, draw_scene_menu};
use sound_engine::auralization::VoiceId;
use sound_engine::core::error::SoundResult;
use sound_engine::playback::SoundPlayer;
use sound_engine::scene::SceneDescription;
use sound_engine::{ListenerPose, PointSource, SoundEngine};
use std::time::Instant;

const LISTENER_SPIN_RADIANS_PER_SECOND: f32 = std::f32::consts::PI * 0.125;

struct UiState {
    status: String,
    last_voice: Option<VoiceId>,
    show_bisector_planes: bool,
    /// Wire view exposes sources and receivers inside closed benchmark rooms.
    benchmark_wireframe: bool,
    /// Full geometry when enabled; an empty acoustic scene supplies the distance-only comparison.
    full_acoustics: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ListenerMode {
    RedSphere,
    Camera,
}

struct PlaygroundState {
    /// Available built-in and locally installed benchmark scenes.
    catalog: Vec<scenes::Entry>,
    /// Active catalog index; zero retains the editable doorway scene.
    scene_index: usize,
    /// Validated benchmark geometry and paired source/receiver positions.
    benchmark: Option<scenes::Loaded>,
    /// Active measured pair in the benchmark scene.
    pair_index: usize,
    /// Visible scene chooser and its highlighted entry.
    scene_menu: bool,
    scene_choice: usize,
    scene_config: SceneConfig,
    scene: PlaygroundScene,
    listener_index: usize,
    listener_rotation_index: usize,
    listener_spin_enabled: bool,
    listener_spin_angle: f32,
    listener_mode: ListenerMode,
    last_camera_listener_pose: Option<ListenerPose>,
    last_camera_listener_sync: Option<Instant>,
    source_index: usize,
    ui: UiState,
}

impl PlaygroundState {
    fn new(diffraction_radius: f32, catalog: Vec<scenes::Entry>) -> Self {
        let scene_config = SceneConfig {
            roof_enabled: false,
            door_open: true,
            outer_walls_enabled: true,
            diffraction_radius,
        };

        Self {
            catalog,
            scene_index: 0,
            benchmark: None,
            pair_index: 0,
            scene_menu: false,
            scene_choice: 0,
            scene_config,
            scene: build_scene(scene_config),
            listener_index: 0,
            listener_rotation_index: 0,
            listener_spin_enabled: false,
            listener_spin_angle: 0.0,
            listener_mode: ListenerMode::RedSphere,
            last_camera_listener_pose: None,
            last_camera_listener_sync: None,
            source_index: 0,
            ui: UiState {
                status: "Ready".to_owned(),
                last_voice: None,
                show_bisector_planes: false,
                benchmark_wireframe: true,
                full_acoustics: true,
            },
        }
    }

    fn listener_position(&self) -> Vector3 {
        if let Some(scene) = &self.benchmark {
            return from_glam(scene.pairs[self.pair_index].listener);
        }
        listener_position(self.listener_index)
    }

    fn position_label(&self, source: bool) -> &str {
        if let Some(scene) = &self.benchmark {
            &scene.pairs[self.pair_index].name
        } else if source {
            source_position_name(self.source_index)
        } else {
            listener_position_name(self.listener_index)
        }
    }

    fn listener_forward(&self) -> Vec3 {
        if self.listener_spin_enabled {
            listener_spin_forward(self.listener_spin_angle)
        } else {
            listener_forward(self.listener_rotation_index)
        }
    }

    fn listener_rotation_label(&self) -> String {
        if self.listener_spin_enabled {
            format!(
                "spinning {:.0} deg",
                self.listener_spin_angle
                    .rem_euclid(std::f32::consts::TAU)
                    .to_degrees()
            )
        } else {
            listener_rotation_name(self.listener_rotation_index).to_owned()
        }
    }

    fn red_sphere_listener_pose(&self) -> ListenerPose {
        let forward = self.listener_forward().normalize_or_zero();
        ListenerPose {
            position: to_glam(self.listener_position()),
            forward,
            up: Vec3::Y,
        }
    }

    fn active_listener_pose(&self, camera: &Camera3D) -> ListenerPose {
        match self.listener_mode {
            ListenerMode::RedSphere => self.red_sphere_listener_pose(),
            ListenerMode::Camera => camera_listener_pose(camera),
        }
    }

    fn listener_mode_label(&self) -> &'static str {
        match self.listener_mode {
            ListenerMode::RedSphere => "red sphere",
            ListenerMode::Camera => "camera",
        }
    }

    fn toggle_listener_mode(&mut self) {
        self.listener_mode = match self.listener_mode {
            ListenerMode::RedSphere => ListenerMode::Camera,
            ListenerMode::Camera => ListenerMode::RedSphere,
        };
        self.last_camera_listener_pose = None;
        self.last_camera_listener_sync = None;
    }

    fn source_position(&self) -> Vector3 {
        if let Some(scene) = &self.benchmark {
            return from_glam(scene.pairs[self.pair_index].source);
        }
        source_position(self.source_index)
    }

    fn point_source(&self) -> PointSource {
        PointSource {
            position: to_glam(self.source_position()),
            amplitude: 1.0,
        }
    }

    fn rebuild_scene(&mut self) {
        self.scene = build_scene(self.scene_config);
    }

    fn toggle_listener_spin(&mut self) {
        self.listener_spin_enabled = !self.listener_spin_enabled;
        if self.listener_spin_enabled {
            self.listener_spin_angle =
                angle_from_forward(listener_forward(self.listener_rotation_index));
        }
    }

    fn update_listener_spin(&mut self, frame_time_seconds: f32) -> bool {
        if !self.listener_spin_enabled {
            return false;
        }

        self.listener_spin_angle = (self.listener_spin_angle
            + LISTENER_SPIN_RADIANS_PER_SECOND * frame_time_seconds)
            .rem_euclid(std::f32::consts::TAU);
        true
    }
}

/// Runs the interactive acoustic playground and owns its audio worker.
pub(crate) fn run() -> SoundResult<()> {
    let options = Options::parse(std::env::args().skip(1))?;
    if options.help {
        Options::print_help();
        return Ok(());
    }
    let catalog = scenes::catalog(&options.benchmark_root);
    if options.list_scenes || options.validate_scenes {
        for entry in &catalog {
            println!("{}: {}", entry.id, entry.title);
            if options.validate_scenes && entry.id != "playground" {
                let loaded = entry
                    .load()
                    .map_err(sound_engine::core::error::SoundError::external)?;
                println!(
                    "  {} triangles, {} pairs, {} diffraction edges ({} inferred)",
                    loaded.triangles.len(),
                    loaded.pairs.len(),
                    loaded.description.diffraction_edges.len(),
                    loaded.inferred_edges,
                );
            }
        }
        return Ok(());
    }
    let sample_rate = SoundPlayer::default_output_sample_rate()?;
    let diffraction_radius = options.diffraction_radius;
    let mut state = PlaygroundState::new(diffraction_radius, catalog);
    if let Some(id) = &options.scene {
        state.scene_index = state
            .catalog
            .iter()
            .position(|e| e.id.eq_ignore_ascii_case(id))
            .ok_or_else(|| {
                sound_engine::core::error::SoundError::invalid_argument(format!(
                    "Unknown scene {id}; use --list-scenes"
                ))
            })?;
        if state.scene_index != 0 {
            state.benchmark = Some(
                state.catalog[state.scene_index]
                    .load()
                    .map_err(sound_engine::core::error::SoundError::external)?,
            );
        }
    }
    let playback_gain = options.volume;
    let config = options.engine_config(sample_rate);
    println!(
        "Playground model=ScalarGeometry, sample_rate={sample_rate}, rays={}, IR samples={}, max_path={}, playback_gain={playback_gain}, diffraction_radius={diffraction_radius}, bounces={}",
        config.acoustics.listener_ray_count,
        config.sound.ir_num_samples,
        config.acoustics.maximum_path_length,
        config.acoustics.max_bounces
    );
    let mut engine = SoundEngine::new(config)?;
    engine.load_scene(playback_scene(&state))?;
    engine.set_listener_pose(state.red_sphere_listener_pose())?;
    let mut audio = AudioSession::new(engine, state.point_source(), &options.sound, playback_gain)?;

    let (mut rl, thread) = raylib::init()
        .size(1280, 720)
        .title("sound-engine playground / scalar diffraction")
        .resizable()
        .msaa_4x()
        .build();

    rl.set_target_fps(60);
    rl.disable_cursor();

    let mut camera = Camera3D::perspective(
        Vector3::new(7.0, 5.5, 8.0),
        Vector3::new(0.0, 1.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        60.0,
    );
    if state.benchmark.is_some() {
        frame_selected_pair(&state, &mut camera);
    }
    if options.camera_stress.is_some() {
        // Start on the motion path before preparing the first voice's IR. A
        // first-frame teleport otherwise rebuilds it immediately after playback.
        camera.position = Vector3::new(7.0, 2.0, 0.0);
        camera.target = Vector3::new(0.0, 1.0, 0.0);
        state.listener_mode = ListenerMode::Camera;
        sync_camera_listener_if_changed(&mut audio, &mut state, &camera, true);
        let voice_id = audio.play(state.active_listener_pose(&camera))?;
        state.ui.last_voice = Some(voice_id);
    }
    let camera_stress = options
        .camera_stress
        .map(|duration| (duration, Instant::now()));

    let mut frame_times = Vec::new();

    while !rl.window_should_close() {
        let frame_started = Instant::now();
        if let Some((duration, started)) = camera_stress {
            let elapsed = started.elapsed().as_secs_f32();
            if started.elapsed() >= duration {
                break;
            }
            let angle = elapsed * 0.75;
            camera.position = Vector3::new(
                angle.cos() * 7.0,
                2.0 + (angle * 0.37).sin(),
                angle.sin() * 4.0,
            );
            camera.target = Vector3::new(0.0, 1.0, 0.0);
        } else if !state.scene_menu {
            rl.update_camera(&mut camera, CameraMode::CAMERA_FREE);
        }
        handle_scene_menu(&mut rl, &mut audio, &mut state, &mut camera);
        if !state.scene_menu {
            handle_config_keys(&rl, &mut audio, &mut state, &camera);
        }
        if state.listener_mode == ListenerMode::Camera {
            sync_camera_listener_if_changed(&mut audio, &mut state, &camera, false);
        }
        if state.update_listener_spin(rl.get_frame_time())
            && state.listener_mode == ListenerMode::RedSphere
        {
            sync_red_sphere_listener(&mut audio, &mut state, "listener spin");
        }

        if !state.scene_menu && start_sound_requested(&rl) {
            state.ui.status = "Building IR and starting sound...".to_owned();
            match audio.play(state.active_listener_pose(&camera)) {
                Ok(voice_id) => {
                    state.ui.last_voice = Some(voice_id);
                    state.ui.status = format!("Started voice {voice_id}");
                }
                Err(error) => {
                    state.ui.status = format!("Audio error: {error}");
                }
            }
        }

        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::new(18, 22, 26, 255));

        {
            let mut d3 = d.begin_mode3D(camera);
            draw_scene(&mut d3, &state);
        }

        draw_overlay(&mut d, &camera, &state);
        if state.scene_menu {
            draw_scene_menu(&mut d, &state);
        }
        drop(d);
        if camera_stress.is_some() {
            frame_times.push(frame_started.elapsed().as_secs_f64() * 1000.0);
        }
    }

    audio.engine.stop_sound_player()?;
    drop(audio);
    if let Some(path) = &options.frame_report {
        use std::io::Write;
        let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
        writeln!(file, "frame,total_ms")?;
        for (index, ms) in frame_times.iter().enumerate() {
            writeln!(file, "{index},{ms}")?;
        }
    }
    Ok(())
}

fn playback_label(state: &PlaygroundState) -> &'static str {
    if state.ui.full_acoustics {
        "Full simulation"
    } else {
        "Distance only"
    }
}

fn empty_acoustic_scene() -> SceneDescription {
    SceneDescription::default()
}

fn playback_scene(state: &PlaygroundState) -> SceneDescription {
    // Keep the visible scene and playback cursor intact. Empty acoustic geometry
    // retains the same 1/r pressure, propagation delay and listener spatialization,
    // while scene invalidation refreshes active voices through their usual crossfade.
    if state.ui.full_acoustics {
        state
            .benchmark
            .as_ref()
            .map(|scene| scene.description.clone())
            .unwrap_or_else(|| scene_description(&state.scene))
    } else {
        empty_acoustic_scene()
    }
}

// A fixed world-space light makes differently oriented walls readable while moving.
// Ambient fill keeps enclosed and back-facing surfaces visible without shadow passes.
