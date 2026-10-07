//! Shared correctness and timing contract for acoustic transport implementations.
//!
//! Geometry is shared with the Bell Labs Box experiment. Independent analytical
//! references live in `acoustic_method/reference.rs`; runtime transport is always
//! evaluated through AcousticPipeline. CSV timing includes synchronous submission.

use sound_engine::acoustics::AcousticPipelineConfig;
use sound_engine::{ListenerPose, PointSource};
#[path = "support/timing.rs"]
mod timing;
use acoustic_fixtures::geometry;
use timing::{milliseconds, percentile as percentile_duration};

use geometry::*;
use glam::{Vec3, vec3};
#[path = "support/provenance.rs"]
mod provenance;
use sound_engine::acoustics::{AcousticPipeline, AcousticQuery, IrSample, IrSnapshot};
use sound_engine::core::config::{AcousticsConfig, OutputChannels, SoundConfig};
use sound_engine::core::error::{SoundError, SoundResult};
use sound_engine::gpu::backend::VkBackend;
use sound_engine::scene::{SceneDescription, SceneManager};
use std::env;
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[path = "acoustic_method/reference.rs"]
mod reference;
use reference::*;

const METHOD: &str = "listener_backward";
const SAMPLE_RATE: u32 = 48_000;
const SAMPLE_COUNT: u32 = 16_384;
const SPEED_OF_SOUND: f64 = 343.0;
const MAXIMUM_PATH_LENGTH: f64 = SPEED_OF_SOUND * SAMPLE_COUNT as f64 / SAMPLE_RATE as f64;
const SOURCE_AMPLITUDE: f32 = 1.0;
const ROOM_REFLECTION_PRESSURE: f64 = 0.8;
const ROOM_ABSORPTION_ENERGY: f32 =
    1.0 - (ROOM_REFLECTION_PRESSURE * ROOM_REFLECTION_PRESSURE) as f32;
const DIFFRACTION_EDGE_HALF_LENGTH: f64 = 20.0;
const DIFFRACTION_QUADRATURE_SEGMENTS: usize = 131_072;
const FREQUENCIES_HZ: &[f64] = &[125.0, 250.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0];

#[derive(Debug, Clone)]
struct Options {
    rays: u32,
    seed: u32,
    seed_repeats: u32,
    warm_repeats: usize,
    case_filter: Option<String>,
    output: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    const ZERO: Self = Self {
        re: 0.0,
        im: 0.0,
    };

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
    measured_peak_abs: f64,
    nonzero_samples: usize,
}

fn main() -> SoundResult<()> {
    if env::args().any(|arg| arg == "--help") {
        println!(
            "acoustic_method_benchmark [--rays 65536] [--seed 7] [--seed-repeats 1] [--warm-repeats 5] [--case NAME] [--output FILE]\nUse --list-cases for available controls. Timings are synchronous host wall times."
        );
        return Ok(());
    }
    if env::args().any(|arg| arg == "--list-cases") {
        for case in benchmark_cases() {
            println!("{}", case.name);
        }
        return Ok(());
    }
    let options = parse_options()?;
    let mut cases = benchmark_cases();
    if let Some(filter) = &options.case_filter {
        cases.retain(|case| case.name == filter);
        if cases.is_empty() {
            return Err(SoundError::invalid_argument(format!(
                "unknown benchmark case: {filter}"
            )));
        }
    }
    let gpu_start = Instant::now();
    let gpu = VkBackend::new()?;
    let gpu_setup = gpu_start.elapsed();

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
    let commit = provenance::revision();
    let method = METHOD;
    writeln!(
        output,
        "method,commit,case,rays,seed,max_bounces,gpu_setup_ms,setup_ms,cold_gpu_ms,warm_gpu_ms,warm_gpu_p95_ms,\
warm_gpu_p99_ms,warm_gpu_max_ms,readback_ms,readback_p99_ms,\
frequency_hz,measured_magnitude,exact_continuous_magnitude,exact_sampled_magnitude,\
magnitude_error_db,phase_error_degrees,complex_relative_error,sampled_magnitude_error_db,\
sampled_phase_error_degrees,sampled_complex_relative_error,measured_ir_energy,exact_path_count,\
time_relative_l1,time_relative_l2,measured_pressure_sum,exact_pressure_sum,measured_peak_abs,nonzero_samples"
    )?;

    for case in cases {
        if options
            .case_filter
            .as_deref()
            .is_some_and(|filter| filter != case.name)
        {
            continue;
        }
        let exact_samples = exact_fractional_ir(&case.exact_paths);
        for seed_offset in 0..options.seed_repeats {
            let mut run_options = options.clone();
            run_options.seed = options.seed.wrapping_add(seed_offset);
            let measurement = trace_case(gpu, &run_options, &case)?;
            let time_metrics = time_domain_metrics(&measurement.ir.samples, &exact_samples);
            for &frequency in FREQUENCIES_HZ {
                let measured = transfer_from_samples(&measurement.ir.samples, frequency);
                let exact_continuous = exact_transfer(&case.exact_paths, frequency);
                let exact_sampled = transfer_from_f64_samples(&exact_samples, frequency);
                let magnitude_error_db =
                    ratio_db(measured.magnitude(), exact_continuous.magnitude());
                let phase_error_degrees =
                    wrapped_phase_degrees(measured.phase() - exact_continuous.phase());
                let complex_relative_error = (measured - exact_continuous).magnitude()
                    / exact_continuous.magnitude().max(1.0e-12);
                let sampled_magnitude_error_db =
                    ratio_db(measured.magnitude(), exact_sampled.magnitude());
                let sampled_phase_error_degrees =
                    wrapped_phase_degrees(measured.phase() - exact_sampled.phase());
                let sampled_complex_relative_error =
                    (measured - exact_sampled).magnitude() / exact_sampled.magnitude().max(1.0e-12);

                writeln!(
                    output,
                    "{method},{commit},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.3},{:.9},{:.9},{:.9},\
{:.6},{:.6},{:.9},{:.6},{:.6},{:.9},{:.9},{},{:.9},{:.9},{:.9},{:.9},{:.9},{}",
                    case.name,
                    run_options.rays,
                    run_options.seed,
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
                    time_metrics.measured_peak_abs,
                    time_metrics.nonzero_samples,
                )?;
            }
        }
    }
    output.flush()?;
    Ok(())
}

fn trace_case(
    gpu: &VkBackend,
    options: &Options,
    case: &BenchmarkCase,
) -> SoundResult<TraceMeasurement> {
    let setup_start = Instant::now();
    let mut scene = SceneManager::new();
    scene.load(case.scene.clone())?;
    let mut pipeline = AcousticPipeline::new(
        gpu,
        acoustic_config(options.rays, options.seed, case.max_bounces),
    )?;
    let setup = setup_start.elapsed();

    let query = AcousticQuery {
        query_id: 1,
        source: PointSource {
            position: case.source,
            amplitude: SOURCE_AMPLITUDE,
        },
        listener: ListenerPose::new(case.listener, -Vec3::Z, Vec3::Y),
    };

    let cold_start = Instant::now();
    let gpu_ir = pipeline.build_ir_gpu(gpu, &mut scene, query)?;
    let cold_gpu = cold_start.elapsed();
    let readback_start = Instant::now();
    let mut ir = gpu_ir.download(gpu)?;
    let mut readback_times = vec![readback_start.elapsed()];

    let mut warm_times = Vec::with_capacity(options.warm_repeats);
    for _ in 0..options.warm_repeats {
        let start = Instant::now();
        let gpu_ir = pipeline.build_ir_gpu(gpu, &mut scene, query)?;
        warm_times.push(start.elapsed());
        let readback_start = Instant::now();
        ir = gpu_ir.download(gpu)?;
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

fn acoustic_config(
    listener_ray_count: u32,
    sampling_seed: u32,
    max_bounces: u32,
) -> AcousticPipelineConfig {
    AcousticPipelineConfig {
        sound: SoundConfig {
            output_channels: OutputChannels::Mono,
            sample_rate: SAMPLE_RATE,
            ir_num_samples: SAMPLE_COUNT,
        },
        tracing: AcousticsConfig {
            listener_ray_count,
            max_bounces,
            maximum_path_length: MAXIMUM_PATH_LENGTH as f32,
            minimum_throughput: 0.0,
            sampling_seed,
        },
        binaural: Default::default(),
    }
}

fn parse_options() -> SoundResult<Options> {
    let mut options = Options {
        rays: 65_536,
        seed: 7,
        seed_repeats: 1,
        warm_repeats: 5,
        case_filter: None,
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
            "--seed-repeats" => {
                index += 1;
                options.seed_repeats = parse_value(&args, index, "--seed-repeats")?;
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
            "--output" => {
                index += 1;
                options.output =
                    Some(PathBuf::from(args.get(index).ok_or_else(|| {
                        SoundError::invalid_argument("--output needs a path")
                    })?));
            }
            unknown => {
                return Err(SoundError::invalid_argument(format!(
                    "unknown benchmark argument: {unknown}"
                )));
            }
        }
        index += 1;
    }
    if options.rays == 0 || options.seed_repeats == 0 || options.warm_repeats == 0 {
        return Err(SoundError::invalid_argument(
            "ray count, seed repeat count, and warm repeat count must be non-zero",
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

fn ratio_db(measured: f64, expected: f64) -> f64 {
    20.0 * (measured.max(1.0e-15) / expected.max(1.0e-15)).log10()
}

fn wrapped_phase_degrees(phase: f64) -> f64 {
    let wrapped = (phase + PI).rem_euclid(2.0 * PI) - PI;
    wrapped.to_degrees()
}

#[cfg(test)]
#[path = "../tests/examples/acoustic_method_benchmark.rs"]
mod tests;
