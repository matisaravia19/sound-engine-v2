use glam::{Mat4, Vec3, vec3};
use raylib::prelude::*;
use sound_engine::core::config::{
    AcousticsConfig, AuralizationConfig, EngineConfig, IrCacheConfig, OutputChannels, SoundConfig,
};
use sound_engine::core::error::SoundResult;
use sound_engine::playback::{PlaybackConfig, SoundPlayer};
use sound_engine::scene::{DiffractionEdge, Material, MeshAsset, SceneDescription, SceneObject};
use sound_engine::{ListenerPose, PointSource, SoundEngine, SoundSourceId};
use std::time::{Duration, Instant};
mod scenes;

const GUITAR_SAMPLE_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../circus.wav");
const BLOCK_SIZE: usize = 1024;
const IR_SECONDS: u32 = 2;
const ROOM_HALF_X: f32 = 8.0;
const ROOM_HALF_Z: f32 = 5.0;
const WALL_HEIGHT: f32 = 3.0;
const WALL_THICKNESS: f32 = 0.25;
const DOOR_HALF_WIDTH: f32 = 1.0;
const ROOF_THICKNESS: f32 = 0.2;
const LISTENER_SPIN_RADIANS_PER_SECOND: f32 = std::f32::consts::PI * 0.125;
const CAMERA_LISTENER_UPDATE_INTERVAL: Duration = Duration::from_millis(50);
const CAMERA_LISTENER_POSITION_EPSILON_METERS: f32 = 0.01;
const CAMERA_LISTENER_ORIENTATION_DOT_THRESHOLD: f32 = 0.999_96;

struct PlaygroundScene {
    acoustic_mesh: AcousticMesh,
    blocks: Vec<Block>,
    diffraction_edges: Vec<DiffractionEdge>,
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

struct UiState {
    status: String,
    last_voice: Option<u64>,
    show_bisector_planes: bool,
    /// Wire view exposes sources and receivers inside closed benchmark rooms.
    benchmark_wireframe: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ListenerMode {
    RedSphere,
    Camera,
}

#[derive(Clone, Copy)]
struct SceneConfig {
    roof_enabled: bool,
    door_open: bool,
    outer_walls_enabled: bool,
    /// Existing edge-strip width, retained when interactive geometry is rebuilt.
    diffraction_radius: f32,
    /// Existing per-edge energy scale, retained across door and wall toggles.
    diffraction_strength: f32,
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
    sound_source_id: Option<SoundSourceId>,
    ui: UiState,
}

impl PlaygroundState {
    fn new(diffraction_radius: f32, diffraction_strength: f32) -> Self {
        let scene_config = SceneConfig {
            roof_enabled: false,
            door_open: true,
            outer_walls_enabled: true,
            diffraction_radius,
            diffraction_strength,
        };

        Self {
            catalog: scenes::catalog(
                &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../output"),
            ),
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
            sound_source_id: None,
            ui: UiState {
                status: "Ready".to_owned(),
                last_voice: None,
                show_bisector_planes: false,
                benchmark_wireframe: true,
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

fn main() -> SoundResult<()> {
    let args: Vec<_> = std::env::args().collect();

    for arg in args.iter().skip(1) {
        let known = matches!(arg.as_str(), "--list-scenes" | "--validate-scenes")
            || [
                "--benchmark-root=",
                "--scene=",
                "--diffraction-radius=",
                "--diffraction-strength=",
                "--volume=",
                "--rays=",
                "--bounces=",
                "--camera-stress-seconds=",
                "--frame-report=",
            ]
            .iter()
            .any(|prefix| arg.starts_with(prefix));

        if !known {
            return Err(sound_engine::core::error::SoundError::invalid_argument(
                format!("unknown option: {arg}"),
            ));
        }
    }

    let benchmark_root = args
        .iter()
        .find_map(|a| a.strip_prefix("--benchmark-root="))
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../output"));
    let catalog = scenes::catalog(&benchmark_root);
    if args
        .iter()
        .any(|a| a == "--list-scenes" || a == "--validate-scenes")
    {
        for entry in &catalog {
            println!("{}: {}", entry.id, entry.title);
            if args.iter().any(|a| a == "--validate-scenes") && entry.id != "playground" {
                let loaded = entry
                    .load()
                    .map_err(sound_engine::core::error::SoundError::external)?;
                println!(
                    "  {} triangles, {} pairs",
                    loaded.triangles.len(),
                    loaded.pairs.len()
                );
            }
        }
        return Ok(());
    }
    let sample_rate = SoundPlayer::default_output_sample_rate()?;
    let diffraction_radius = args
        .iter()
        .find_map(|a| a.strip_prefix("--diffraction-radius="))
        .unwrap_or("1.25")
        .parse::<f32>()
        .map_err(|_| {
            sound_engine::core::error::SoundError::invalid_argument("invalid --diffraction-radius")
        })?;
    let diffraction_strength = args
        .iter()
        .find_map(|a| a.strip_prefix("--diffraction-strength="))
        .unwrap_or("0.85")
        .parse::<f32>()
        .map_err(|_| {
            sound_engine::core::error::SoundError::invalid_argument(
                "invalid --diffraction-strength",
            )
        })?;
    if !diffraction_radius.is_finite()
        || diffraction_radius <= 0.0
        || !diffraction_strength.is_finite()
        || !(0.0..=1.0).contains(&diffraction_strength)
    {
        return Err(sound_engine::core::error::SoundError::invalid_argument(
            "diffraction radius must be positive and finite; strength must be in [0, 1]",
        ));
    }
    let mut state = PlaygroundState::new(diffraction_radius, diffraction_strength);
    state.catalog = catalog;
    if let Some(id) = args.iter().find_map(|a| a.strip_prefix("--scene=")) {
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
    // Playback headroom is separate from calibrated ray pressure. The same
    // dry gain applies to every acoustic model and can be overridden for A/B runs.
    let playback_gain = args
        .iter()
        .find_map(|a| a.strip_prefix("--volume="))
        .unwrap_or("0.6")
        .parse::<f32>()
        .map_err(|_| sound_engine::core::error::SoundError::invalid_argument("invalid --volume"))?;
    if !playback_gain.is_finite() || playback_gain < 0.0 {
        return Err(sound_engine::core::error::SoundError::invalid_argument(
            "--volume must be finite and non-negative",
        ));
    }
    let mut config = engine_config(sample_rate);
    if let Some(value) = args.iter().find_map(|a| a.strip_prefix("--rays=")) {
        config.acoustics.listener_ray_count = value.parse().map_err(|_| {
            sound_engine::core::error::SoundError::invalid_argument("invalid --rays")
        })?;
    }
    if let Some(value) = args.iter().find_map(|a| a.strip_prefix("--bounces=")) {
        config.acoustics.max_bounces = value.parse().map_err(|_| {
            sound_engine::core::error::SoundError::invalid_argument("invalid --bounces")
        })?;
    }
    println!(
        "Playground model=ScalarGeometry, sample_rate={sample_rate}, rays={}, IR samples={}, max_path={}, playback_gain={playback_gain}, diffraction_radius={diffraction_radius}, diffraction_strength={diffraction_strength}, bounces={}",
        config.acoustics.listener_ray_count,
        config.sound.ir_num_samples,
        config.acoustics.maximum_path_length,
        config.acoustics.max_bounces
    );
    let mut engine = SoundEngine::new(config)?;
    engine.load_scene(
        state
            .benchmark
            .as_ref()
            .map(|b| b.description.clone())
            .unwrap_or_else(|| scene_description(&state.scene)),
    )?;
    engine.set_listener_pose(state.red_sphere_listener_pose())?;
    let sound_source_id = engine.create_sound_source(state.point_source())?;
    state.sound_source_id = Some(sound_source_id);
    let sound_id = engine.load_wav_sound(GUITAR_SAMPLE_PATH)?;
    engine.start_sound_player(PlaybackConfig {
        prefill_blocks: 3,
        max_queue_seconds: 0.064,
    })?;

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
    let camera_stress_seconds = std::env::args()
        .find_map(|argument| {
            argument
                .strip_prefix("--camera-stress-seconds=")
                .map(str::to_owned)
        })
        .and_then(|seconds| seconds.parse::<f32>().ok())
        .filter(|seconds| seconds.is_finite() && *seconds > 0.0);
    if camera_stress_seconds.is_some() {
        // Start on the motion path before preparing the first voice's IR. A
        // first-frame teleport otherwise rebuilds it immediately after playback.
        camera.position = Vector3::new(7.0, 2.0, 0.0);
        camera.target = Vector3::new(0.0, 1.0, 0.0);
        state.listener_mode = ListenerMode::Camera;
        sync_camera_listener_if_changed(&mut engine, &mut state, &camera, true);
        let voice_id = play_source_sound(&mut engine, sound_id, &state, &camera, playback_gain)?;
        state.ui.last_voice = Some(voice_id);
    }
    let camera_stress =
        camera_stress_seconds.map(|seconds| (Duration::from_secs_f32(seconds), Instant::now()));

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
        handle_scene_menu(&mut rl, &mut engine, &mut state, &mut camera);
        if !state.scene_menu {
            handle_config_keys(&rl, &mut engine, &mut state, &camera);
        }
        if state.listener_mode == ListenerMode::Camera {
            sync_camera_listener_if_changed(&mut engine, &mut state, &camera, false);
        }
        if state.update_listener_spin(rl.get_frame_time())
            && state.listener_mode == ListenerMode::RedSphere
        {
            sync_red_sphere_listener(&mut engine, &mut state, "listener spin");
        }

        if !state.scene_menu && start_sound_requested(&rl) {
            state.ui.status = "Building IR and starting sound...".to_owned();
            match play_source_sound(&mut engine, sound_id, &state, &camera, playback_gain) {
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

    engine.stop_sound_player()?;
    drop(engine);
    if let Some(path) = args.iter().find_map(|a| a.strip_prefix("--frame-report=")) {
        use std::io::Write;
        let mut file = std::io::BufWriter::new(
            std::fs::File::create(path)
                .map_err(|e| sound_engine::core::error::SoundError::external(e.to_string()))?,
        );
        writeln!(file, "frame,total_ms")
            .map_err(|e| sound_engine::core::error::SoundError::external(e.to_string()))?;
        for (index, ms) in frame_times.iter().enumerate() {
            writeln!(file, "{index},{ms}")
                .map_err(|e| sound_engine::core::error::SoundError::external(e.to_string()))?;
        }
    }
    Ok(())
}

fn engine_config(sample_rate: u32) -> EngineConfig {
    EngineConfig {
        sound: SoundConfig {
            output_channels: OutputChannels::Stereo,
            sample_rate,
            ir_num_samples: sample_rate * IR_SECONDS,
        },
        acoustics: AcousticsConfig {
            listener_ray_count: 65_536,
            max_bounces: 3,
            maximum_path_length: 343.0,
            minimum_throughput: 1.0e-6,
            sampling_seed: 0,
        },
        cache: IrCacheConfig::default(),
        binaural: Default::default(),
        auralization: AuralizationConfig {
            block_size: BLOCK_SIZE,
        },
    }
}

fn play_source_sound(
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

fn frame_selected_pair(state: &PlaygroundState, camera: &mut Camera3D) {
    let source = to_glam(state.source_position());
    let listener = to_glam(state.listener_position());
    let center = (source + listener) * 0.5;
    let distance = source.distance(listener).max(2.0);
    camera.target = from_glam(center);
    camera.position = from_glam(center + vec3(0.8, 0.65, 1.0) * distance);
    camera.up = Vector3::new(0.0, 1.0, 0.0);
}

fn scene_menu_rows(height: i32) -> usize {
    ((height - 180) / 26).clamp(3, 18) as usize
}

fn handle_scene_menu(
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
    let description = benchmark
        .as_ref()
        .map(|b| b.description.clone())
        .unwrap_or_else(|| scene_description(&state.scene));
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
    Ok(())
}

fn draw_scene_menu(d: &mut RaylibDrawHandle<'_>, state: &PlaygroundState) {
    let width = (d.get_screen_width() - 80).min(760).max(240);
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

fn handle_config_keys(
    rl: &RaylibHandle,
    engine: &mut SoundEngine,
    state: &mut PlaygroundState,
    camera: &Camera3D,
) {
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
    match engine.load_scene(scene_description(&state.scene)) {
        Ok(version) => {
            state.ui.status = format!("{label} changed; scene version {version}");
        }
        Err(error) => {
            state.ui.status = format!("Scene reload error: {error}");
        }
    }
}

fn sync_red_sphere_listener(engine: &mut SoundEngine, state: &mut PlaygroundState, label: &str) {
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

fn sync_camera_listener_if_changed(
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

fn camera_listener_pose(camera: &Camera3D) -> ListenerPose {
    ListenerPose {
        position: to_glam(camera.position),
        forward: (to_glam(camera.target) - to_glam(camera.position)).normalize_or_zero(),
        up: to_glam(camera.up).normalize_or_zero(),
    }
}

fn listener_poses_match(left: ListenerPose, right: ListenerPose) -> bool {
    left.position.distance_squared(right.position)
        <= CAMERA_LISTENER_POSITION_EPSILON_METERS.powi(2)
        && left.forward.dot(right.forward) >= CAMERA_LISTENER_ORIENTATION_DOT_THRESHOLD
        && left.up.dot(right.up) >= CAMERA_LISTENER_ORIENTATION_DOT_THRESHOLD
}

fn next_index(index: usize, len: usize) -> usize {
    (index + 1) % len
}

fn listener_position_count() -> usize {
    7
}

fn listener_position(index: usize) -> Vector3 {
    match index % listener_position_count() {
        0 => Vector3::new(4.5, 0.45, 0.0),
        1 => Vector3::new(5.4, 0.45, -3.2),
        2 => Vector3::new(5.4, 0.45, 3.2),
        3 => Vector3::new(1.8, 0.45, 0.0),
        4 => Vector3::new(0.55, 0.45, -1.6),
        5 => Vector3::new(0.55, 0.45, 1.6),
        _ => Vector3::new(7.35, 0.45, 0.0),
    }
}

fn listener_position_name(index: usize) -> &'static str {
    match index % listener_position_count() {
        0 => "right room center",
        1 => "right room back",
        2 => "right room front",
        3 => "near doorway",
        4 => "near north jamb",
        5 => "near south jamb",
        _ => "near outer wall",
    }
}

fn listener_rotation_count() -> usize {
    4
}

fn listener_forward(index: usize) -> Vec3 {
    match index % listener_rotation_count() {
        0 => -Vec3::X,
        1 => Vec3::Z,
        2 => Vec3::X,
        _ => -Vec3::Z,
    }
}

fn listener_spin_forward(angle_radians: f32) -> Vec3 {
    Vec3::new(angle_radians.cos(), 0.0, angle_radians.sin())
}

fn angle_from_forward(forward: Vec3) -> f32 {
    forward.z.atan2(forward.x)
}

fn listener_rotation_name(index: usize) -> &'static str {
    match index % listener_rotation_count() {
        0 => "facing doorway",
        1 => "facing +Z",
        2 => "facing away",
        _ => "facing -Z",
    }
}

fn source_position_count() -> usize {
    8
}

fn source_position(index: usize) -> Vector3 {
    match index % source_position_count() {
        0 => Vector3::new(-4.5, 0.45, 0.0),
        1 => Vector3::new(-5.4, 0.45, -3.2),
        2 => Vector3::new(-5.4, 0.45, 3.2),
        3 => Vector3::new(-1.8, 0.45, 0.0),
        4 => Vector3::new(4.5, 0.45, 0.0),
        5 => Vector3::new(5.4, 0.45, -3.2),
        6 => Vector3::new(5.4, 0.45, 3.2),
        _ => Vector3::new(1.8, 0.45, 0.0),
    }
}

fn source_position_name(index: usize) -> &'static str {
    match index % source_position_count() {
        0 => "left room center",
        1 => "left room back",
        2 => "left room front",
        3 => "left near doorway",
        4 => "same room center",
        5 => "same room back",
        6 => "same room front",
        _ => "same room doorway",
    }
}

fn build_scene(config: SceneConfig) -> PlaygroundScene {
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
            doorway_diffraction_edges(config.diffraction_radius, config.diffraction_strength)
        } else {
            Vec::new()
        },
    }
}

fn doorway_diffraction_edges(radius: f32, strength: f32) -> Vec<DiffractionEdge> {
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
            base_strength: strength,
        },
        DiffractionEdge {
            id: 2,
            start: vec3(half_thickness, 0.0, -DOOR_HALF_WIDTH),
            end: vec3(half_thickness, WALL_HEIGHT, -DOOR_HALF_WIDTH),
            bisector_dir: vec3(diagonal, 0.0, diagonal),
            edge_angle_radians: 1.5 * std::f32::consts::PI,
            diffraction_radius: radius,
            base_strength: strength,
        },
        DiffractionEdge {
            id: 3,
            start: vec3(-half_thickness, 0.0, DOOR_HALF_WIDTH),
            end: vec3(-half_thickness, WALL_HEIGHT, DOOR_HALF_WIDTH),
            bisector_dir: vec3(-diagonal, 0.0, -diagonal),
            edge_angle_radians: 1.5 * std::f32::consts::PI,
            diffraction_radius: radius,
            base_strength: strength,
        },
        DiffractionEdge {
            id: 4,
            start: vec3(half_thickness, 0.0, DOOR_HALF_WIDTH),
            end: vec3(half_thickness, WALL_HEIGHT, DOOR_HALF_WIDTH),
            bisector_dir: vec3(diagonal, 0.0, -diagonal),
            edge_angle_radians: 1.5 * std::f32::consts::PI,
            diffraction_radius: radius,
            base_strength: strength,
        },
    ]
}

fn scene_description(scene: &PlaygroundScene) -> SceneDescription {
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
            absorption_bands: vec![0.18],
            scattering: 0.12,
            roughness: 0.15,
            transmission: None,
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

fn start_sound_requested(rl: &RaylibHandle) -> bool {
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

fn draw_scene(d: &mut RaylibMode3D<'_, RaylibDrawHandle<'_>>, state: &PlaygroundState) {
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
        draw_diffraction_edges(d, &scene.description.diffraction_edges);
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
    draw_diffraction_edges(d, &state.scene.diffraction_edges);
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

// A fixed world-space light makes differently oriented walls readable while moving.
// Ambient fill keeps enclosed and back-facing surfaces visible without shadow passes.
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

fn draw_diffraction_edges(
    d: &mut RaylibMode3D<'_, RaylibDrawHandle<'_>>,
    edges: &[DiffractionEdge],
) {
    for edge in edges {
        let start = from_glam(edge.start);
        let end = from_glam(edge.end);
        d.draw_line_3D(start, end, Color::YELLOW);
        d.draw_sphere(start, 0.08, Color::YELLOW);
        d.draw_sphere(end, 0.08, Color::YELLOW);
    }
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
        draw_plane_line(d, start, end, Color::YELLOW);

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

fn draw_overlay(d: &mut RaylibDrawHandle, camera: &Camera3D, state: &PlaygroundState) {
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

fn to_glam(value: Vector3) -> Vec3 {
    vec3(value.x, value.y, value.z)
}

fn from_glam(value: Vec3) -> Vector3 {
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
