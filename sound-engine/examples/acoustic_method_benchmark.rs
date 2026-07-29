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
use sound_engine::scene::{
    DiffractionEdge, Material, MeshAsset, SceneDescription, SceneManager, SceneObject,
};
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
const MAXIMUM_PATH_LENGTH: f64 = SPEED_OF_SOUND * SAMPLE_COUNT as f64 / SAMPLE_RATE as f64;
const SOURCE_AMPLITUDE: f32 = 1.0;
const ROOM_REFLECTION_PRESSURE: f64 = 0.8;
const ROOM_ABSORPTION_ENERGY: f32 = 1.0 - (ROOM_REFLECTION_PRESSURE * ROOM_REFLECTION_PRESSURE) as f32;
const DIFFRACTION_EDGE_HALF_LENGTH: f64 = 20.0;
const DIFFRACTION_QUADRATURE_SEGMENTS: usize = 131_072;
const FREQUENCIES_HZ: &[f64] = &[125.0, 250.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0];

#[derive(Debug, Clone)]
struct Options {
    rays: u32,
    seed: u32,
    warm_repeats: usize,
    case_filter: Option<String>,
    drop_backend: bool,
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
    cold_gpu: Duration,
    warm_gpu_median: Duration,
    warm_gpu_p95: Duration,
    warm_gpu_p99: Duration,
    warm_gpu_max: Duration,
    readback_median: Duration,
    readback_p99: Duration,
}

struct TimeDomainMetrics {
    relative_l1: f64,
    relative_l2: f64,
    measured_pressure_sum: f64,
    exact_pressure_sum: f64,
    nonzero_samples: usize,
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
    let result = run_benchmarks(&gpu, &options, gpu_setup, cases, output);
    if !options.drop_backend {
        // Backend destruction currently stalls in Vulkan teardown on the test
        // machine. Per-query sweeps skip that unrelated lifecycle cost so each
        // process exits promptly; `--drop-backend` reproduces and times it.
        std::mem::forget(gpu);
    }
    result
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
        "method,commit,case,rays,seed,max_bounces,gpu_setup_ms,setup_ms,cold_gpu_ms,warm_gpu_ms,warm_gpu_p95_ms,\
warm_gpu_p99_ms,warm_gpu_max_ms,readback_ms,readback_p99_ms,\
frequency_hz,measured_magnitude,exact_continuous_magnitude,exact_sampled_magnitude,\
magnitude_error_db,phase_error_degrees,complex_relative_error,sampled_magnitude_error_db,\
sampled_phase_error_degrees,sampled_complex_relative_error,measured_ir_energy,exact_path_count,\
time_relative_l1,time_relative_l2,measured_pressure_sum,exact_pressure_sum,nonzero_samples"
    )?;

    for case in cases {
        if options.case_filter.as_deref().is_some_and(|filter| filter != case.name) {
            continue;
        }
        let measurement = trace_case(gpu, options, &case)?;
        let exact_samples = exact_fractional_ir(&case.exact_paths);
        let time_metrics = time_domain_metrics(&measurement.ir.samples, &exact_samples);
        for &frequency in FREQUENCIES_HZ {
            let measured = transfer_from_samples(&measurement.ir.samples, frequency);
            let exact_continuous = exact_transfer(&case.exact_paths, frequency);
            let exact_sampled = transfer_from_f64_samples(&exact_samples, frequency);
            let magnitude_error_db = ratio_db(measured.magnitude(), exact_continuous.magnitude());
            let phase_error_degrees = wrapped_phase_degrees(measured.phase() - exact_continuous.phase());
            let complex_relative_error =
                (measured - exact_continuous).magnitude() / exact_continuous.magnitude().max(1.0e-12);
            let sampled_magnitude_error_db = ratio_db(measured.magnitude(), exact_sampled.magnitude());
            let sampled_phase_error_degrees = wrapped_phase_degrees(measured.phase() - exact_sampled.phase());
            let sampled_complex_relative_error =
                (measured - exact_sampled).magnitude() / exact_sampled.magnitude().max(1.0e-12);

            writeln!(
                output,
                "{METHOD},{COMMIT},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.3},{:.9},{:.9},{:.9},\
{:.6},{:.6},{:.9},{:.6},{:.6},{:.9},{:.9},{},{:.9},{:.9},{:.9},{:.9},{}",
                case.name,
                options.rays,
                options.seed,
                case.max_bounces,
                milliseconds(gpu_setup),
                milliseconds(measurement.setup),
                milliseconds(measurement.cold_gpu),
                milliseconds(measurement.warm_gpu_median),
                milliseconds(measurement.warm_gpu_p95),
                milliseconds(measurement.warm_gpu_p99),
                milliseconds(measurement.warm_gpu_max),
                milliseconds(measurement.readback_median),
                milliseconds(measurement.readback_p99),
                frequency,
                measured.magnitude(),
                exact_continuous.magnitude(),
                exact_sampled.magnitude(),
                magnitude_error_db,
                phase_error_degrees,
                complex_relative_error,
                sampled_magnitude_error_db,
                sampled_phase_error_degrees,
                sampled_complex_relative_error,
                measurement.ir.energy,
                case.exact_paths.len(),
                time_metrics.relative_l1,
                time_metrics.relative_l2,
                time_metrics.measured_pressure_sum,
                time_metrics.exact_pressure_sum,
                time_metrics.nonzero_samples,
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
    let gpu_ir = pipeline.build_ir_gpu(gpu, &mut scene, query)?;
    let cold_gpu = cold_start.elapsed();
    let readback_start = Instant::now();
    let mut ir = pipeline.download_gpu_ir(gpu, &gpu_ir)?;
    let mut readback_times = vec![readback_start.elapsed()];

    let mut warm_times = Vec::with_capacity(options.warm_repeats);
    for _ in 0..options.warm_repeats {
        let start = Instant::now();
        let gpu_ir = pipeline.build_ir_gpu(gpu, &mut scene, query)?;
        warm_times.push(start.elapsed());
        let readback_start = Instant::now();
        ir = pipeline.download_gpu_ir(gpu, &gpu_ir)?;
        readback_times.push(readback_start.elapsed());
    }

    warm_times.sort_unstable();
    readback_times.sort_unstable();
    Ok(TraceMeasurement {
        ir,
        setup,
        cold_gpu,
        warm_gpu_median: percentile_duration(&warm_times, 0.50),
        warm_gpu_p95: percentile_duration(&warm_times, 0.95),
        warm_gpu_p99: percentile_duration(&warm_times, 0.99),
        warm_gpu_max: *warm_times.last().unwrap(),
        readback_median: percentile_duration(&readback_times, 0.50),
        readback_p99: percentile_duration(&readback_times, 0.99),
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
            maximum_path_length: MAXIMUM_PATH_LENGTH as f32,
            minimum_throughput: 0.0,
            sampling_seed,
        },
        cache: IrCacheConfig::default(),
        auralization: AuralizationConfig { block_size: 1024 },
    }
}

fn benchmark_cases() -> Vec<BenchmarkCase> {
    let direct_source = Vec3::ZERO;
    let direct_listener_1m = vec3(1.003, 0.217, -0.091);
    let direct_listener_2m = direct_listener_1m.normalize() * 2.0;
    let direct_listener_5m = direct_listener_1m.normalize() * 5.0;

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
    // Thirty reflections cover every rigid-room image that can arrive inside
    // this benchmark's 341 ms IR. The test below guards that assumption if the
    // room dimensions, sample count, or speed of sound change.
    let rigid_room_bounces = 30;
    let diffraction_source = vec3(-2.0, 0.0, 3.0);
    let diffraction_listener = vec3(-2.0, 0.0, -3.0);
    let exterior_wedge_angle = 1.5 * PI;
    let diffraction_exact = btm_diffraction_paths(
        diffraction_source,
        diffraction_listener,
        exterior_wedge_angle,
        -DIFFRACTION_EDGE_HALF_LENGTH,
        DIFFRACTION_EDGE_HALF_LENGTH,
        DIFFRACTION_QUADRATURE_SEGMENTS,
    );

    vec![
        BenchmarkCase {
            name: "free_field_1m",
            source: direct_source,
            listener: direct_listener_1m,
            max_bounces: 0,
            scene: empty_scene(),
            exact_paths: vec![ExactPath {
                length_meters: direct_source.distance(direct_listener_1m) as f64,
                pressure_reflection: 1.0,
                _reflection_order: 0,
            }],
        },
        BenchmarkCase {
            name: "free_field_2m",
            source: direct_source,
            listener: direct_listener_2m,
            max_bounces: 0,
            scene: empty_scene(),
            exact_paths: vec![ExactPath {
                length_meters: 2.0,
                pressure_reflection: 1.0,
                _reflection_order: 0,
            }],
        },
        BenchmarkCase {
            name: "free_field_5m",
            source: direct_source,
            listener: direct_listener_5m,
            max_bounces: 0,
            scene: empty_scene(),
            exact_paths: vec![ExactPath {
                length_meters: 5.0,
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
            name: "absorbing_single_plane",
            source: plane_source,
            listener: plane_listener,
            max_bounces: 1,
            scene: single_plane_scene(ROOM_ABSORPTION_ENERGY),
            exact_paths: vec![
                ExactPath {
                    length_meters: plane_direct,
                    pressure_reflection: 1.0,
                    _reflection_order: 0,
                },
                ExactPath {
                    length_meters: plane_reflected,
                    pressure_reflection: ROOM_REFLECTION_PRESSURE,
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
        BenchmarkCase {
            name: "rigid_shoebox_exact",
            source: room_source,
            listener: room_listener,
            max_bounces: rigid_room_bounces,
            scene: shoebox_scene(room_size, 0.0),
            // For rigid (Neumann) rectangular boundaries, mirroring the
            // monopole in every room cell is an exact time-domain solution of
            // the scalar wave equation. Retaining all images whose arrival is
            // inside the finite IR gives an exact reference for this window.
            exact_paths: rigid_shoebox_image_paths_within_length(
                room_size,
                room_source,
                room_listener,
                MAXIMUM_PATH_LENGTH,
            ),
        },
        BenchmarkCase {
            name: "diffraction_wedge_shadow",
            source: diffraction_source,
            listener: diffraction_listener,
            max_bounces: 1,
            scene: diffraction_wedge_scene(
                exterior_wedge_angle as f32,
                DIFFRACTION_EDGE_HALF_LENGTH as f32,
                0.45,
            ),
            // The Biot-Tolstoy-Medwin secondary-edge-source integral is the
            // exact first-order solution for an infinite rigid wedge. Truncating
            // the integral to the authored edge gives the standard finite-edge
            // first-order reference used by interactive geometric acoustics.
            exact_paths: diffraction_exact.clone(),
        },
        BenchmarkCase {
            name: "diffraction_proxy_radius_020",
            source: diffraction_source,
            listener: diffraction_listener,
            max_bounces: 1,
            scene: diffraction_wedge_scene(
                exterior_wedge_angle as f32,
                DIFFRACTION_EDGE_HALF_LENGTH as f32,
                0.20,
            ),
            exact_paths: diffraction_exact.clone(),
        },
        BenchmarkCase {
            name: "diffraction_proxy_radius_090",
            source: diffraction_source,
            listener: diffraction_listener,
            max_bounces: 1,
            scene: diffraction_wedge_scene(
                exterior_wedge_angle as f32,
                DIFFRACTION_EDGE_HALF_LENGTH as f32,
                0.90,
            ),
            exact_paths: diffraction_exact,
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

fn time_domain_metrics(measured: &[IrSample], exact: &[f64]) -> TimeDomainMetrics {
    let mut l1_error = 0.0;
    let mut l2_error = 0.0;
    let mut exact_l1 = 0.0;
    let mut exact_l2 = 0.0;
    let mut measured_pressure_sum = 0.0;
    let mut exact_pressure_sum = 0.0;
    let mut nonzero_samples = 0;
    for (measured, exact) in measured.iter().zip(exact) {
        let measured = measured.left as f64;
        let difference = measured - exact;
        l1_error += difference.abs();
        l2_error += difference * difference;
        exact_l1 += exact.abs();
        exact_l2 += exact * exact;
        measured_pressure_sum += measured;
        exact_pressure_sum += exact;
        nonzero_samples += usize::from(measured != 0.0);
    }
    TimeDomainMetrics {
        relative_l1: l1_error / exact_l1.max(1.0e-15),
        relative_l2: l2_error.sqrt() / exact_l2.sqrt().max(1.0e-15),
        measured_pressure_sum,
        exact_pressure_sum,
        nonzero_samples,
    }
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

fn rigid_shoebox_image_paths_within_length(
    room_size: Vec3,
    source: Vec3,
    listener: Vec3,
    maximum_length: f64,
) -> Vec<ExactPath> {
    // Each image coordinate is 2*n*L +/- source. One extra cell beyond the
    // simple radius bound accounts for the source/listener offsets within the
    // base room; the final Euclidean-distance check removes that overreach.
    let x_cells = (maximum_length / (2.0 * room_size.x as f64)).ceil() as i32 + 1;
    let y_cells = (maximum_length / (2.0 * room_size.y as f64)).ceil() as i32 + 1;
    let z_cells = (maximum_length / (2.0 * room_size.z as f64)).ceil() as i32 + 1;
    let mut paths = Vec::new();
    for nx in -x_cells..=x_cells {
        for px in 0..=1 {
            let ox = (2 * nx - px).unsigned_abs();
            for ny in -y_cells..=y_cells {
                for py in 0..=1 {
                    let oy = (2 * ny - py).unsigned_abs();
                    for nz in -z_cells..=z_cells {
                        for pz in 0..=1 {
                            let oz = (2 * nz - pz).unsigned_abs();
                            let image = Vec3::new(
                                image_coordinate(room_size.x, source.x, nx, px),
                                image_coordinate(room_size.y, source.y, ny, py),
                                image_coordinate(room_size.z, source.z, nz, pz),
                            );
                            let length = image.distance(listener) as f64;
                            if length <= maximum_length {
                                paths.push(ExactPath {
                                    length_meters: length,
                                    pressure_reflection: 1.0,
                                    _reflection_order: ox + oy + oz,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    paths
}

fn btm_diffraction_paths(
    source: Vec3,
    listener: Vec3,
    exterior_wedge_angle: f64,
    edge_start: f64,
    edge_end: f64,
    segment_count: usize,
) -> Vec<ExactPath> {
    // This benchmark wedge is aligned to +Y. Its upper face points along
    // (-X + Z)/sqrt(2), and angles increase around +Y through the exterior
    // (air) region. Source and receiver radii are therefore measured in XZ.
    let edge_axis = Vec3::Y;
    let angle_origin = vec3(-1.0, 0.0, 1.0).normalize();
    let source_radial = vec3(source.x, 0.0, source.z);
    let listener_radial = vec3(listener.x, 0.0, listener.z);
    let source_radius = source_radial.length() as f64;
    let listener_radius = listener_radial.length() as f64;
    let source_angle = wedge_angle(angle_origin, edge_axis, source_radial);
    let listener_angle = wedge_angle(angle_origin, edge_axis, listener_radial);
    let wedge_index = PI / exterior_wedge_angle;
    let segment_length = (edge_end - edge_start) / segment_count as f64;
    let mut paths = Vec::with_capacity(segment_count);

    for segment in 0..segment_count {
        // Midpoint integration avoids evaluating the integrable BTM onset
        // singularity exactly at the shortest source-edge-receiver path.
        let edge_coordinate = edge_start + (segment as f64 + 0.5) * segment_length;
        let source_axial = edge_coordinate - source.y as f64;
        let listener_axial = edge_coordinate - listener.y as f64;
        let source_distance = source_radius.hypot(source_axial);
        let listener_distance = listener_radius.hypot(listener_axial);
        let eta_argument = (
            source_distance * listener_distance + source_axial * listener_axial
        ) / (source_radius * listener_radius);
        let eta = eta_argument.max(1.0).acosh();

        let mut beta_sum = 0.0;
        for source_sign in [-1.0, 1.0] {
            for listener_sign in [-1.0, 1.0] {
                let phi = PI + source_sign * source_angle + listener_sign * listener_angle;
                let numerator = (wedge_index * phi).sin();
                let denominator = (wedge_index * eta).cosh() - (wedge_index * phi).cos();
                beta_sum += numerator / denominator;
            }
        }

        let total_length = source_distance + listener_distance;
        // The published BTM Green function carries -nu/(4*pi). The engine is
        // calibrated to direct pressure 1/r instead of 1/(4*pi*r), so multiplying
        // the whole reference by 4*pi leaves this -nu factor.
        let pressure = -wedge_index * beta_sum * segment_length
            / (source_distance * listener_distance);
        paths.push(ExactPath {
            length_meters: total_length,
            // ExactPath stores the numerator of pressure/length.
            pressure_reflection: pressure * total_length,
            _reflection_order: 1,
        });
    }
    paths
}

fn wedge_angle(origin: Vec3, axis: Vec3, direction: Vec3) -> f64 {
    let origin = origin.normalize();
    let direction = direction.normalize();
    let sine = axis.dot(origin.cross(direction)) as f64;
    let cosine = origin.dot(direction) as f64;
    sine.atan2(cosine).rem_euclid(2.0 * PI)
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

fn diffraction_wedge_scene(
    exterior_wedge_angle: f32,
    edge_half_length: f32,
    proxy_radius: f32,
) -> SceneDescription {
    let face_length = 80.0;
    let upper = vec3(-1.0, 0.0, 1.0).normalize() * face_length;
    let lower = vec3(-1.0, 0.0, -1.0).normalize() * face_length;
    let mut scene = scene_from_walls(
        vec![
            quad(
                vec3(0.0, -edge_half_length, 0.0),
                vec3(0.0, edge_half_length, 0.0),
                vec3(upper.x, edge_half_length, upper.z),
                vec3(upper.x, -edge_half_length, upper.z),
            ),
            quad(
                vec3(0.0, -edge_half_length, 0.0),
                vec3(lower.x, -edge_half_length, lower.z),
                vec3(lower.x, edge_half_length, lower.z),
                vec3(0.0, edge_half_length, 0.0),
            ),
        ],
        1.0,
    );
    scene.diffraction_edges.push(DiffractionEdge {
        id: 1,
        start: vec3(0.0, -edge_half_length, 0.0),
        end: vec3(0.0, edge_half_length, 0.0),
        // The solid right-angle corner is centered on -X, so the exterior
        // wedge and its two authored shadow boundaries are bisected by +X.
        bisector_dir: Vec3::X,
        edge_angle_radians: exterior_wedge_angle,
        diffraction_radius: proxy_radius,
        base_strength: 1.0,
    });
    scene
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
        case_filter: None,
        drop_backend: false,
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
            "--case" => {
                index += 1;
                options.case_filter = Some(
                    args.get(index)
                        .ok_or_else(|| SoundError::invalid_argument("--case needs a name"))?
                        .clone(),
                );
            }
            "--drop-backend" => {
                options.drop_backend = true;
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

fn percentile_duration(sorted: &[Duration], fraction: f64) -> Duration {
    let index = ((sorted.len() - 1) as f64 * fraction).ceil() as usize;
    sorted[index]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_image_enumeration_has_direct_and_two_first_reflections() {
        let paths = parallel_image_paths(10.0, vec3(2.0, 0.0, 0.0), vec3(7.0, 0.0, 0.0), 1, 1.0);
        assert_eq!(paths.len(), 3);
        assert_eq!(paths.iter().filter(|path| path._reflection_order == 0).count(), 1);
        assert_eq!(paths.iter().filter(|path| path._reflection_order == 1).count(), 2);
    }

    #[test]
    fn shoebox_first_order_has_six_walls_and_direct_path() {
        let paths = shoebox_image_paths(vec3(10.0, 8.0, 6.0), vec3(2.0, 2.0, 2.0), vec3(7.0, 5.0, 3.0), 1, 1.0);
        assert_eq!(paths.len(), 7);
        assert_eq!(paths.iter().filter(|path| path._reflection_order == 0).count(), 1);
        assert_eq!(paths.iter().filter(|path| path._reflection_order == 1).count(), 6);
    }

    #[test]
    fn rigid_shoebox_ir_window_needs_at_most_thirty_reflections() {
        let paths = rigid_shoebox_image_paths_within_length(
            vec3(10.0, 8.0, 6.0),
            vec3(2.1, 2.7, 2.2),
            vec3(7.4, 5.9, 3.8),
            MAXIMUM_PATH_LENGTH,
        );
        assert!(paths.len() > 1_000);
        assert!(paths.iter().map(|path| path._reflection_order).max().unwrap() <= 30);
    }

    #[test]
    fn btm_quadrature_is_converged_at_benchmark_resolution() {
        let source = vec3(-2.0, 0.0, 3.0);
        let listener = vec3(-2.0, 0.0, -3.0);
        let coarse = btm_diffraction_paths(source, listener, 1.5 * PI, -20.0, 20.0, 65_536);
        let fine = btm_diffraction_paths(source, listener, 1.5 * PI, -20.0, 20.0, 131_072);
        let coarse_transfer = exact_transfer(&coarse, 8_000.0);
        let fine_transfer = exact_transfer(&fine, 8_000.0);
        let relative_error = (coarse_transfer - fine_transfer).magnitude()
            / fine_transfer.magnitude().max(1.0e-12);
        assert!(relative_error < 1.0e-4, "relative quadrature error was {relative_error}");
    }

    #[test]
    fn fractional_reference_preserves_path_pressure() {
        let path = ExactPath {
            length_meters: 1.2345,
            pressure_reflection: 0.75,
            _reflection_order: 0,
        };
        let expected = path.pressure_reflection / path.length_meters;
        let samples = exact_fractional_ir(&[path]);
        assert!((samples.iter().sum::<f64>() - expected).abs() < 1.0e-12);
        assert_eq!(samples.iter().filter(|sample| **sample != 0.0).count(), 2);
    }
}
