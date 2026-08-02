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
}

struct PlaygroundState {
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
    fn new() -> Self {
        let scene_config = SceneConfig {
            roof_enabled: false,
            door_open: true,
            outer_walls_enabled: true,
        };

        Self {
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
            },
        }
    }

    fn listener_position(&self) -> Vector3 {
        listener_position(self.listener_index)
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
                self.listener_spin_angle.rem_euclid(std::f32::consts::TAU).to_degrees()
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
            self.listener_spin_angle = angle_from_forward(listener_forward(self.listener_rotation_index));
        }
    }

    fn update_listener_spin(&mut self, frame_time_seconds: f32) -> bool {
        if !self.listener_spin_enabled {
            return false;
        }

        self.listener_spin_angle = (self.listener_spin_angle + LISTENER_SPIN_RADIANS_PER_SECOND * frame_time_seconds)
            .rem_euclid(std::f32::consts::TAU);
        true
    }
}

fn main() -> SoundResult<()> {
    let mut state = PlaygroundState::new();
    let sample_rate = SoundPlayer::default_output_sample_rate()?;
    let mut engine = SoundEngine::new(engine_config(sample_rate))?;
    engine.load_scene(scene_description(&state.scene))?;
    engine.set_listener_pose(state.red_sphere_listener_pose())?;
    let sound_source_id = engine.create_sound_source(state.point_source())?;
    state.sound_source_id = Some(sound_source_id);
    let sound_id = engine.load_wav_sound(GUITAR_SAMPLE_PATH)?;
    engine.start_sound_player(PlaybackConfig::default())?;

    let (mut rl, thread) = raylib::init()
        .size(1280, 720)
        .title("sound-engine playground")
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

    while !rl.window_should_close() {
        rl.update_camera(&mut camera, CameraMode::CAMERA_FREE);
        handle_config_keys(&rl, &mut engine, &mut state, &camera);
        if state.listener_mode == ListenerMode::Camera {
            sync_camera_listener_if_changed(&mut engine, &mut state, &camera, false);
        }
        if state.update_listener_spin(rl.get_frame_time()) && state.listener_mode == ListenerMode::RedSphere {
            sync_red_sphere_listener(&mut engine, &mut state, "listener spin");
        }

        if start_sound_requested(&rl) {
            state.ui.status = "Building IR and starting sound...".to_owned();
            match play_source_sound(&mut engine, sound_id, &state, &camera) {
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
    }

    engine.stop_sound_player()?;
    drop(engine);
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
        auralization: AuralizationConfig { block_size: BLOCK_SIZE },
    }
}

fn play_source_sound(
    engine: &mut SoundEngine,
    sound_id: u64,
    state: &PlaygroundState,
    camera: &Camera3D,
) -> SoundResult<u64> {
    engine.set_listener_pose(state.active_listener_pose(camera))?;
    let source_id = state.sound_source_id.ok_or_else(|| {
        sound_engine::core::error::SoundError::invalid_state("sandbox sound source is not initialized")
    })?;
    engine.play_sound_from_source(sound_id, source_id, 0.9)
}

fn handle_config_keys(rl: &RaylibHandle, engine: &mut SoundEngine, state: &mut PlaygroundState, camera: &Camera3D) {
    if rl.is_key_pressed(KeyboardKey::KEY_ONE) {
        state.scene_config.roof_enabled = !state.scene_config.roof_enabled;
        reload_scene(engine, state, "roof");
    }

    if rl.is_key_pressed(KeyboardKey::KEY_TWO) {
        state.scene_config.door_open = !state.scene_config.door_open;
        reload_scene(engine, state, "door");
    }

    if rl.is_key_pressed(KeyboardKey::KEY_THREE) {
        state.listener_index = next_index(state.listener_index, listener_position_count());
        sync_red_sphere_listener(engine, state, "red sphere position");
    }

    if rl.is_key_pressed(KeyboardKey::KEY_FOUR) {
        state.listener_spin_enabled = false;
        state.listener_rotation_index = next_index(state.listener_rotation_index, listener_rotation_count());
        sync_red_sphere_listener(engine, state, "red sphere rotation");
    }

    if rl.is_key_pressed(KeyboardKey::KEY_FIVE) {
        state.source_index = next_index(state.source_index, source_position_count());
        sync_source(engine, state);
    }

    if rl.is_key_pressed(KeyboardKey::KEY_SIX) {
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
                source_position_name(state.source_index)
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
            listener_position_name(state.listener_index),
            state.listener_rotation_label()
        );
        return;
    }

    match engine.set_listener_pose(state.red_sphere_listener_pose()) {
        Ok(()) => {
            state.ui.status = format!(
                "Changed {label}: {} / {}",
                listener_position_name(state.listener_index),
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

    match engine.set_listener_pose(pose) {
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
    left.position.distance_squared(right.position) <= CAMERA_LISTENER_POSITION_EPSILON_METERS.powi(2)
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
            center: Vector3::new(0.0, WALL_HEIGHT * 0.5, -(ROOM_HALF_Z + DOOR_HALF_WIDTH) * 0.5),
            size: Vector3::new(WALL_THICKNESS, WALL_HEIGHT, ROOM_HALF_Z - DOOR_HALF_WIDTH),
            color: Color::new(125, 132, 138, 255),
        },
        Block {
            center: Vector3::new(0.0, WALL_HEIGHT * 0.5, (ROOM_HALF_Z + DOOR_HALF_WIDTH) * 0.5),
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
            doorway_diffraction_edges()
        } else {
            Vec::new()
        },
    }
}

fn doorway_diffraction_edges() -> Vec<DiffractionEdge> {
    let half_thickness = WALL_THICKNESS * 0.5;
    let diagonal = std::f32::consts::FRAC_1_SQRT_2;
    vec![
        DiffractionEdge {
            id: 1,
            start: vec3(-half_thickness, 0.0, -DOOR_HALF_WIDTH),
            end: vec3(-half_thickness, WALL_HEIGHT, -DOOR_HALF_WIDTH),
            bisector_dir: vec3(-diagonal, 0.0, diagonal),
            edge_angle_radians: 1.5 * std::f32::consts::PI,
            diffraction_radius: 1.25,
            base_strength: 0.65,
        },
        DiffractionEdge {
            id: 2,
            start: vec3(half_thickness, 0.0, -DOOR_HALF_WIDTH),
            end: vec3(half_thickness, WALL_HEIGHT, -DOOR_HALF_WIDTH),
            bisector_dir: vec3(diagonal, 0.0, diagonal),
            edge_angle_radians: 1.5 * std::f32::consts::PI,
            diffraction_radius: 1.25,
            base_strength: 0.65,
        },
        DiffractionEdge {
            id: 3,
            start: vec3(-half_thickness, 0.0, DOOR_HALF_WIDTH),
            end: vec3(-half_thickness, WALL_HEIGHT, DOOR_HALF_WIDTH),
            bisector_dir: vec3(-diagonal, 0.0, -diagonal),
            edge_angle_radians: 1.5 * std::f32::consts::PI,
            diffraction_radius: 1.25,
            base_strength: 0.65,
        },
        DiffractionEdge {
            id: 4,
            start: vec3(half_thickness, 0.0, DOOR_HALF_WIDTH),
            end: vec3(half_thickness, WALL_HEIGHT, DOOR_HALF_WIDTH),
            bisector_dir: vec3(diagonal, 0.0, -diagonal),
            edge_angle_radians: 1.5 * std::f32::consts::PI,
            diffraction_radius: 1.25,
            base_strength: 0.65,
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
            vertices: scene.acoustic_mesh.vertices.iter().copied().map(to_glam).collect(),
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
        || (rl.is_mouse_button_pressed(MouseButton::MOUSE_BUTTON_LEFT) && mouse_in_bisector_toggle(rl))
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
    d.draw_plane(
        Vector3::new(0.0, 0.0, 0.0),
        Vector2::new(ROOM_HALF_X * 2.0, ROOM_HALF_Z * 2.0),
        Color::new(38, 44, 48, 255),
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
    d.draw_cube_v(block.center, block.size, block.color);
    d.draw_cube_wires_v(block.center, block.size, Color::new(190, 198, 204, 255));
}

fn draw_marker(d: &mut RaylibMode3D<'_, RaylibDrawHandle<'_>>, position: Vector3, radius: f32, color: Color) {
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

fn draw_diffraction_edges(d: &mut RaylibMode3D<'_, RaylibDrawHandle<'_>>, edges: &[DiffractionEdge]) {
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

fn draw_plane_line(d: &mut RaylibMode3D<'_, RaylibDrawHandle<'_>>, start: Vec3, end: Vec3, color: Color) {
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
        "Space: start sound | B: toggle planes | L: toggle camera/red-sphere listener",
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
        &format!(
            "1 roof: {} | 2 door: {} | 3 red pos: {} | 4 red rot: {}",
            on_off(state.scene_config.roof_enabled),
            if state.scene_config.door_open { "open" } else { "closed" },
            listener_position_name(state.listener_index),
            state.listener_rotation_label()
        ),
        24,
        162,
        18,
        Color::LIGHTGRAY,
    );
    d.draw_text(
        &format!(
            "5 source: {} | 6 outer walls: {}",
            source_position_name(state.source_index),
            on_off(state.scene_config.outer_walls_enabled)
        ),
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
    d.draw_text(
        "Suggested next toggles: absorption preset, wall thickness, obstacle in doorway",
        24,
        234,
        18,
        Color::GRAY,
    );
    d.draw_text(&state.ui.status, 24, 258, 18, Color::RAYWHITE);
}

fn on_off(value: bool) -> &'static str {
    if value { "on" } else { "off" }
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
