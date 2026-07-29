//! Shared correctness and timing contract for acoustic transport implementations.
//!
//! This listener-backward adapter is intentionally kept structurally identical
//! to the source-forward adapter on `bench/source-forward`. Scene definitions,
//! analytical references, frequencies, metrics, warm-up policy, and CSV schema
//! must change in both worktrees together.

use glam::{Mat4, Vec3, vec3};
use sound_engine::acoustics::{AcousticPipeline, AcousticQuery, IrSample, IrSnapshot};
use sound_engine::core::config::{
    AcousticsConfig, AuralizationConfig, EngineConfig, IrCacheConfig, OutputChannels, SoundConfig,
};
use sound_engine::core::error::{SoundError, SoundResult};
use sound_engine::gpu::backend::VkBackend;
use sound_engine::scene::{Material, MeshAsset, SceneDescription, SceneManager, SceneObject};
use std::env;
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

const METHOD: &str = "listener_backward";
const COMMIT: &str = "a5c4e408483cd91e71301f6c6b88e8d9c807f2c9";
const SAMPLE_RATE: u32 = 48_000;
const SAMPLE_COUNT: u32 = 16_384;
const SPEED_OF_SOUND: f64 = 343.0;
const SOURCE_AMPLITUDE: f32 = 1.0;
const ROOM_REFLECTION_PRESSURE: f64 = 0.8;
const ROOM_ABSORPTION_ENERGY: f32 = 1.0 - (ROOM_REFLECTION_PRESSURE * ROOM_REFLECTION_PRESSURE) as f32;
const FREQUENCIES_HZ: &[f64] = &[125.0, 250.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0];

#[derive(Debug, Clone)]
struct Options {
    rays: u32,
    seed: u32,
    warm_repeats: usize,
    output: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    const ZERO: Self = Self { re: 0.0, im: 0.0 };

    fn from_polar(magnitude: f64, phase: f64) -> Self {
        Self {
            re: magnitude * phase.cos(),
            im: magnitude * phase.sin(),
        }
    }

    fn magnitude(self) -> f64 {
        self.re.hypot(self.im)
    }

    fn phase(self) -> f64 {
        self.im.atan2(self.re)
    }
}

impl std::ops::AddAssign for Complex {
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ExactPath {
    length_meters: f64,
    pressure_reflection: f64,
    _reflection_order: u32,
}

struct BenchmarkCase {
    name: &'static str,
    source: Vec3,
    listener: Vec3,
    max_bounces: u32,
    scene: SceneDescription,
    exact_paths: Vec<ExactPath>,
}

struct TraceMeasurement {
    ir: IrSnapshot,
    setup: Duration,
    cold_trace: Duration,
    warm_median: Duration,
}

fn main() -> SoundResult<()> {
    let options = parse_options()?;
    let gpu_start = Instant::now();
    let gpu = VkBackend::new()?;
    let gpu_setup = gpu_start.elapsed();
    let cases = benchmark_cases();

    let output: Box<dyn Write> = match &options.output {
        Some(path) => Box::new(BufWriter::new(File::create(path)?)),
        None => Box::new(BufWriter::new(std::io::stdout())),
    };
    run_benchmarks(&gpu, &options, gpu_setup, cases, output)
}

fn run_benchmarks(
    gpu: &VkBackend,
    options: &Options,
    gpu_setup: Duration,
    cases: Vec<BenchmarkCase>,
    mut output: Box<dyn Write>,
) -> SoundResult<()> {
    writeln!(
        output,
        "method,commit,case,rays,seed,max_bounces,gpu_setup_ms,setup_ms,cold_trace_ms,warm_trace_ms,\
frequency_hz,measured_magnitude,exact_continuous_magnitude,exact_sampled_magnitude,\
magnitude_error_db,phase_error_degrees,complex_relative_error,measured_ir_energy,exact_path_count"
    )?;

    for case in cases {
        let measurement = trace_case(gpu, options, &case)?;
        for &frequency in FREQUENCIES_HZ {
            let measured = transfer_from_samples(&measurement.ir.samples, frequency);
            let exact_continuous = exact_transfer(&case.exact_paths, frequency);
            let exact_samples = exact_fractional_ir(&case.exact_paths);
            let exact_sampled = transfer_from_f64_samples(&exact_samples, frequency);
            let magnitude_error_db = ratio_db(measured.magnitude(), exact_continuous.magnitude());
            let phase_error_degrees = wrapped_phase_degrees(measured.phase() - exact_continuous.phase());
            let complex_relative_error =
                (measured - exact_continuous).magnitude() / exact_continuous.magnitude().max(1.0e-12);

            writeln!(
                output,
                "{METHOD},{COMMIT},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.3},{:.9},{:.9},{:.9},\
{:.6},{:.6},{:.9},{:.9},{}",
                case.name,
                options.rays,
                options.seed,
                case.max_bounces,
                milliseconds(gpu_setup),
                milliseconds(measurement.setup),
                milliseconds(measurement.cold_trace),
                milliseconds(measurement.warm_median),
                frequency,
                measured.magnitude(),
                exact_continuous.magnitude(),
                exact_sampled.magnitude(),
                magnitude_error_db,
                phase_error_degrees,
                complex_relative_error,
                measurement.ir.energy,
                case.exact_paths.len(),
            )?;
        }
    }
    output.flush()?;
    Ok(())
}

fn trace_case(gpu: &VkBackend, options: &Options, case: &BenchmarkCase) -> SoundResult<TraceMeasurement> {
    let setup_start = Instant::now();
    let mut scene = SceneManager::new();
    scene.load(case.scene.clone())?;
    let mut pipeline = AcousticPipeline::new(gpu, engine_config(options.rays, options.seed, case.max_bounces))?;
    let setup = setup_start.elapsed();

    let query = AcousticQuery {
        query_id: 1,
        source_position: case.source,
        listener_position: case.listener,
        listener_right: Vec3::X,
        source_amplitude: SOURCE_AMPLITUDE,
    };

    let cold_start = Instant::now();
    let mut ir = pipeline.build_ir(gpu, &mut scene, query)?;
    let cold_trace = cold_start.elapsed();

    let mut warm_times = Vec::with_capacity(options.warm_repeats);
    for _ in 0..options.warm_repeats {
        let start = Instant::now();
        ir = pipeline.build_ir(gpu, &mut scene, query)?;
        warm_times.push(start.elapsed());
    }

    Ok(TraceMeasurement {
        ir,
        setup,
        cold_trace,
        warm_median: median_duration(&mut warm_times),
    })
}

fn engine_config(listener_ray_count: u32, sampling_seed: u32, max_bounces: u32) -> EngineConfig {
    EngineConfig {
        sound: SoundConfig {
            output_channels: OutputChannels::Mono,
            sample_rate: SAMPLE_RATE,
            ir_num_samples: SAMPLE_COUNT,
        },
        acoustics: AcousticsConfig {
            listener_ray_count,
            max_bounces,
            maximum_path_length: SPEED_OF_SOUND as f32 * SAMPLE_COUNT as f32 / SAMPLE_RATE as f32,
            minimum_throughput: 0.0,
            sampling_seed,
        },
        cache: IrCacheConfig::default(),
        auralization: AuralizationConfig { block_size: 1024 },
    }
}

fn benchmark_cases() -> Vec<BenchmarkCase> {
    let direct_source = vec3(0.0, 0.0, 0.0);
    let direct_listener = vec3(1.003, 0.217, -0.091);
    let direct_length = direct_source.distance(direct_listener) as f64;

    let plane_source = vec3(1.1, -0.3, 0.2);
    let plane_listener = vec3(2.7, 0.8, -0.4);
    let plane_direct = plane_source.distance(plane_listener) as f64;
    let plane_image = vec3(-plane_source.x, plane_source.y, plane_source.z);
    let plane_reflected = plane_image.distance(plane_listener) as f64;

    let parallel_length = 10.0;
    let parallel_source = vec3(2.1, -0.4, 0.3);
    let parallel_listener = vec3(7.3, 0.6, -0.2);
    let parallel_bounces = 6;

    let room_size = vec3(10.0, 8.0, 6.0);
    let room_source = vec3(2.1, 2.7, 2.2);
    let room_listener = vec3(7.4, 5.9, 3.8);
    let room_bounces = 6;

    vec![
        BenchmarkCase {
            name: "free_field_direct",
            source: direct_source,
            listener: direct_listener,
            max_bounces: 0,
            scene: empty_scene(),
            exact_paths: vec![ExactPath {
                length_meters: direct_length,
                pressure_reflection: 1.0,
                _reflection_order: 0,
            }],
        },
        BenchmarkCase {
            name: "rigid_single_plane",
            source: plane_source,
            listener: plane_listener,
            max_bounces: 1,
            scene: single_plane_scene(0.0),
            exact_paths: vec![
                ExactPath {
                    length_meters: plane_direct,
                    pressure_reflection: 1.0,
                    _reflection_order: 0,
                },
                ExactPath {
                    length_meters: plane_reflected,
                    pressure_reflection: 1.0,
                    _reflection_order: 1,
                },
            ],
        },
        BenchmarkCase {
            name: "parallel_planes",
            source: parallel_source,
            listener: parallel_listener,
            max_bounces: parallel_bounces,
            scene: parallel_plane_scene(parallel_length, ROOM_ABSORPTION_ENERGY),
            exact_paths: parallel_image_paths(
                parallel_length as f64,
                parallel_source,
                parallel_listener,
                parallel_bounces,
                ROOM_REFLECTION_PRESSURE,
            ),
        },
        BenchmarkCase {
            name: "closed_shoebox",
            source: room_source,
            listener: room_listener,
            max_bounces: room_bounces,
            scene: shoebox_scene(room_size, ROOM_ABSORPTION_ENERGY),
            exact_paths: shoebox_image_paths(
                room_size,
                room_source,
                room_listener,
                room_bounces,
                ROOM_REFLECTION_PRESSURE,
            ),
        },
    ]
}

fn exact_transfer(paths: &[ExactPath], frequency_hz: f64) -> Complex {
    let wave_number = 2.0 * PI * frequency_hz / SPEED_OF_SOUND;
    let mut result = Complex::ZERO;
    for path in paths {
        let amplitude = path.pressure_reflection / path.length_meters.max(1.0e-9);
        result += Complex::from_polar(amplitude, -wave_number * path.length_meters);
    }
    result
}

fn exact_fractional_ir(paths: &[ExactPath]) -> Vec<f64> {
    let mut samples = vec![0.0; SAMPLE_COUNT as usize];
    for path in paths {
        let sample_position = path.length_meters / SPEED_OF_SOUND * SAMPLE_RATE as f64;
        let base = sample_position.floor() as usize;
        let fraction = sample_position - base as f64;
        let amplitude = path.pressure_reflection / path.length_meters.max(1.0e-9);
        if let Some(sample) = samples.get_mut(base) {
            *sample += amplitude * (1.0 - fraction);
        }
        if let Some(sample) = samples.get_mut(base + 1) {
            *sample += amplitude * fraction;
        }
    }
    samples
}

fn transfer_from_samples(samples: &[IrSample], frequency_hz: f64) -> Complex {
    let mut result = Complex::ZERO;
    let angular_step = -2.0 * PI * frequency_hz / SAMPLE_RATE as f64;
    for (index, sample) in samples.iter().enumerate() {
        result += Complex::from_polar(sample.left as f64, angular_step * index as f64);
    }
    result
}

fn transfer_from_f64_samples(samples: &[f64], frequency_hz: f64) -> Complex {
    let mut result = Complex::ZERO;
    let angular_step = -2.0 * PI * frequency_hz / SAMPLE_RATE as f64;
    for (index, &sample) in samples.iter().enumerate() {
        result += Complex::from_polar(sample, angular_step * index as f64);
    }
    result
}

fn parallel_image_paths(
    room_length: f64,
    source: Vec3,
    listener: Vec3,
    max_order: u32,
    reflection_pressure: f64,
) -> Vec<ExactPath> {
    let mut paths = Vec::new();
    let cells = max_order as i32 + 1;
    for cell in -cells..=cells {
        for parity in 0..=1 {
            let order = (2 * cell - parity).unsigned_abs();
            if order > max_order {
                continue;
            }
            let image_x = 2.0 * cell as f64 * room_length
                + if parity == 0 {
                    source.x as f64
                } else {
                    -(source.x as f64)
                };
            let image = Vec3::new(image_x as f32, source.y, source.z);
            paths.push(ExactPath {
                length_meters: image.distance(listener) as f64,
                pressure_reflection: reflection_pressure.powi(order as i32),
                _reflection_order: order,
            });
        }
    }
    paths
}

fn shoebox_image_paths(
    room_size: Vec3,
    source: Vec3,
    listener: Vec3,
    max_order: u32,
    reflection_pressure: f64,
) -> Vec<ExactPath> {
    let cells = max_order as i32 + 1;
    let mut paths = Vec::new();
    for nx in -cells..=cells {
        for px in 0..=1 {
            let ox = (2 * nx - px).unsigned_abs();
            for ny in -cells..=cells {
                for py in 0..=1 {
                    let oy = (2 * ny - py).unsigned_abs();
                    for nz in -cells..=cells {
                        for pz in 0..=1 {
                            let oz = (2 * nz - pz).unsigned_abs();
                            let order = ox + oy + oz;
                            if order > max_order {
                                continue;
                            }
                            let image = Vec3::new(
                                image_coordinate(room_size.x, source.x, nx, px),
                                image_coordinate(room_size.y, source.y, ny, py),
                                image_coordinate(room_size.z, source.z, nz, pz),
                            );
                            paths.push(ExactPath {
                                length_meters: image.distance(listener) as f64,
                                pressure_reflection: reflection_pressure.powi(order as i32),
                                _reflection_order: order,
                            });
                        }
                    }
                }
            }
        }
    }
    paths
}

fn image_coordinate(length: f32, source: f32, cell: i32, parity: i32) -> f32 {
    2.0 * cell as f32 * length + if parity == 0 { source } else { -source }
}

fn empty_scene() -> SceneDescription {
    SceneDescription {
        meshes: Vec::new(),
        materials: Vec::new(),
        objects: Vec::new(),
        diffraction_edges: Vec::new(),
    }
}

fn single_plane_scene(absorption: f32) -> SceneDescription {
    let extent = 80.0;
    scene_from_walls(vec![quad_x(0.0, -extent, extent, -extent, extent)], absorption)
}

fn parallel_plane_scene(length: f32, absorption: f32) -> SceneDescription {
    let extent = 80.0;
    scene_from_walls(
        vec![
            quad_x(0.0, -extent, extent, -extent, extent),
            quad_x(length, -extent, extent, -extent, extent),
        ],
        absorption,
    )
}

fn shoebox_scene(size: Vec3, absorption: f32) -> SceneDescription {
    scene_from_walls(
        vec![
            quad_x(0.0, 0.0, size.y, 0.0, size.z),
            quad_x(size.x, 0.0, size.y, 0.0, size.z),
            quad_y(0.0, 0.0, size.x, 0.0, size.z),
            quad_y(size.y, 0.0, size.x, 0.0, size.z),
            quad_z(0.0, 0.0, size.x, 0.0, size.y),
            quad_z(size.z, 0.0, size.x, 0.0, size.y),
        ],
        absorption,
    )
}

fn scene_from_walls(walls: Vec<Vec<Vec3>>, absorption: f32) -> SceneDescription {
    let mut vertices = Vec::new();
    for wall in walls {
        vertices.extend(wall);
    }
    SceneDescription {
        meshes: vec![MeshAsset {
            id: 1,
            vertices,
            indices: Vec::new(),
            opaque: true,
        }],
        materials: vec![Material {
            id: 1,
            absorption_bands: vec![absorption],
            scattering: 0.0,
            roughness: 0.0,
            transmission: None,
        }],
        objects: vec![SceneObject {
            id: 1,
            mesh_id: 1,
            material_id: 1,
            transform: Mat4::IDENTITY,
            active: true,
        }],
        diffraction_edges: Vec::new(),
    }
}

fn quad_x(x: f32, y0: f32, y1: f32, z0: f32, z1: f32) -> Vec<Vec3> {
    quad(vec3(x, y0, z0), vec3(x, y1, z0), vec3(x, y1, z1), vec3(x, y0, z1))
}

fn quad_y(y: f32, x0: f32, x1: f32, z0: f32, z1: f32) -> Vec<Vec3> {
    quad(vec3(x0, y, z0), vec3(x1, y, z0), vec3(x1, y, z1), vec3(x0, y, z1))
}

fn quad_z(z: f32, x0: f32, x1: f32, y0: f32, y1: f32) -> Vec<Vec3> {
    quad(vec3(x0, y0, z), vec3(x1, y0, z), vec3(x1, y1, z), vec3(x0, y1, z))
}

fn quad(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> Vec<Vec3> {
    vec![a, b, c, a, c, d]
}

fn parse_options() -> SoundResult<Options> {
    let mut options = Options {
        rays: 65_536,
        seed: 7,
        warm_repeats: 5,
        output: None,
    };
    let args = env::args().skip(1).collect::<Vec<_>>();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--rays" => {
                index += 1;
                options.rays = parse_value(&args, index, "--rays")?;
            }
            "--seed" => {
                index += 1;
                options.seed = parse_value(&args, index, "--seed")?;
            }
            "--warm-repeats" => {
                index += 1;
                options.warm_repeats = parse_value(&args, index, "--warm-repeats")?;
            }
            "--output" => {
                index += 1;
                options.output = Some(PathBuf::from(
                    args.get(index)
                        .ok_or_else(|| SoundError::invalid_argument("--output needs a path"))?,
                ));
            }
            unknown => {
                return Err(SoundError::invalid_argument(format!(
                    "unknown benchmark argument: {unknown}"
                )));
            }
        }
        index += 1;
    }
    if options.rays == 0 || options.warm_repeats == 0 {
        return Err(SoundError::invalid_argument(
            "ray count and warm repeat count must be non-zero",
        ));
    }
    Ok(options)
}

fn parse_value<T: std::str::FromStr>(args: &[String], index: usize, name: &str) -> SoundResult<T> {
    args.get(index)
        .ok_or_else(|| SoundError::invalid_argument(format!("{name} needs a value")))?
        .parse()
        .map_err(|_| SoundError::invalid_argument(format!("invalid value for {name}")))
}

fn median_duration(values: &mut [Duration]) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn ratio_db(measured: f64, expected: f64) -> f64 {
    20.0 * (measured.max(1.0e-15) / expected.max(1.0e-15)).log10()
}

fn wrapped_phase_degrees(phase: f64) -> f64 {
    let wrapped = (phase + PI).rem_euclid(2.0 * PI) - PI;
    wrapped.to_degrees()
}
