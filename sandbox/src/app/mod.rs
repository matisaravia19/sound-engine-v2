//! Interactive playground state and frame loop.
mod audio;
mod config;
mod controls;
mod geometry;
mod layout;
mod listener;
mod poses;
mod render;
mod scene;

use crate::scenes;
use audio::AudioSession;
use config::Options;
use controls::{
    frame_selected_pair, handle_config_keys, handle_scene_menu, start_sound_requested,
    sync_camera_listener_if_changed, sync_red_sphere_listener,
};
use geometry::to_glam;
use listener::{ListenerController, ListenerMode};
use raylib::prelude::*;
use render::{draw_overlay, draw_scene, draw_scene_menu};
use scene::SceneState;
use sound_engine::core::error::SoundResult;
use sound_engine::playback::SoundPlayer;
use sound_engine::{ListenerPose, PointSource, SoundEngine};
use std::time::Instant;

struct UiState {
    status: String,
    show_bisector_planes: bool,
    /// Wire view exposes sources and receivers inside closed benchmark rooms.
    benchmark_wireframe: bool,
}

struct PlaygroundState {
    scene: SceneState,
    /// Visible scene chooser and its highlighted entry.
    scene_menu: bool,
    scene_choice: usize,
    listener: ListenerController,
    ui: UiState,
}

impl PlaygroundState {
    fn new(scene: SceneState) -> Self {
        Self {
            scene_choice: scene.index(),
            scene,
            scene_menu: false,
            listener: ListenerController::default(),
            ui: UiState {
                status: "Ready".to_owned(),
                show_bisector_planes: false,
                benchmark_wireframe: true,
            },
        }
    }

    fn red_sphere_listener_pose(&self) -> ListenerPose {
        self.listener
            .marker_pose(to_glam(self.scene.listener_position()))
    }

    fn active_listener_pose(&self, camera: &Camera3D) -> ListenerPose {
        self.listener
            .active_pose(to_glam(self.scene.listener_position()), camera)
    }

    fn point_source(&self) -> PointSource {
        PointSource {
            position: to_glam(self.scene.source_position()),
            amplitude: 1.0,
        }
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
    let diffraction_radius = options.diffraction_radius;
    let mut state = PlaygroundState::new(SceneState::new(
        catalog,
        diffraction_radius,
        options.scene.as_deref(),
    )?);
    let sample_rate = SoundPlayer::default_output_sample_rate()?;
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
    engine.load_scene(state.scene.description())?;
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
    if state.scene.benchmark().is_some() {
        frame_selected_pair(&state, &mut camera);
    }
    if options.camera_stress.is_some() {
        // Start on the motion path before preparing the first voice's IR. A
        // first-frame teleport otherwise rebuilds it immediately after playback.
        camera.position = Vector3::new(7.0, 2.0, 0.0);
        camera.target = Vector3::new(0.0, 1.0, 0.0);
        state.listener.set_mode(ListenerMode::Camera);
        sync_camera_listener_if_changed(&mut audio, &mut state, &camera, true);
        audio.play(state.active_listener_pose(&camera))?;
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
        if state.listener.mode() == ListenerMode::Camera {
            sync_camera_listener_if_changed(&mut audio, &mut state, &camera, false);
        }
        if state.listener.advance_spin(rl.get_frame_time())
            && state.listener.mode() == ListenerMode::RedSphere
        {
            sync_red_sphere_listener(&mut audio, &mut state, "listener spin");
        }

        if !state.scene_menu && start_sound_requested(&rl) {
            state.ui.status = "Building IR and starting sound...".to_owned();
            match audio.play(state.active_listener_pose(&camera)) {
                Ok(voice_id) => {
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
