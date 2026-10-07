//! Measured BRAS RS5/RS6 geometry: every exported propagation IR comes from rays.
//! Preparation and calibration: scripts/bras_reference.py; citations: references.md.
use sound_engine::acoustics::AcousticPipelineConfig;
use sound_engine::{ListenerPose, PointSource};
#[path = "support/provenance.rs"]
mod provenance;
#[path = "support/timing.rs"]
mod timing;

use acoustic_fixtures::{bras, read_positions};
use glam::Vec3;
use sound_engine::acoustics::export_ir;
use sound_engine::acoustics::{AcousticPipeline, AcousticQuery};
use sound_engine::core::config::*;
use sound_engine::gpu::backend::VkBackend;
use sound_engine::scene::SceneManager;
use std::{
    fs,
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut scene_name = String::from("RS5");
    let mut rays = 262_144_u32;
    let mut seeds = 3_u32;
    let mut bounces = 3_u32;
    let mut radius = 0.2_f32;
    let mut diffraction = true;
    let mut reverse = false;
    let mut repeats = 1_usize;
    let mut warmups = 3_usize;
    let mut selected_case = None;
    let mut motion_height = 0.0_f32;
    let mut output = PathBuf::from("output/bras-ray-baseline");
    let mut input = PathBuf::from("output/measured-reference-research");
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--help" {
            println!(
                "bras_diffraction [--scene RS5|RS6] [--rays 262144] [--seeds 3] [--bounces 3] [--radius 0.2] [--diffraction true|false] [--reverse true|false] [--repeats 1] [--warmups 3] [--case RS5_LS1_MP1] [--motion-height 0] [--input output/measured-reference-research] [--output output/bras-ray-baseline]"
            );
            return Ok(());
        }
        let value = args.next().ok_or("missing option value")?;
        match arg.as_str() {
            "--scene" => scene_name = value,
            "--rays" => rays = value.parse()?,
            "--seeds" => seeds = value.parse()?,
            "--bounces" => bounces = value.parse()?,
            "--radius" => radius = value.parse()?,
            "--reverse" => reverse = value.parse()?,
            "--diffraction" => diffraction = value.parse()?,
            "--repeats" => repeats = value.parse()?,
            "--warmups" => warmups = value.parse()?,
            "--case" => selected_case = Some(value),
            "--motion-height" => motion_height = value.parse()?,
            "--input" => input = value.into(),
            "--output" => output = value.into(),
            _ => return Err(format!("unknown option: {arg}").into()),
        }
    }
    if !["RS5", "RS6"].contains(&scene_name.as_str())
        || rays == 0
        || seeds == 0
        || repeats == 0
        || !radius.is_finite()
        || radius <= 0.0
    {
        return Err("invalid scene, ray/seed count or radius".into());
    }
    let end_seed = 7_u32.checked_add(seeds).ok_or("seed range overflow")?;
    if !motion_height.is_finite() || motion_height < 0.0 {
        return Err("motion height must be finite and nonnegative".into());
    }
    let setup = if scene_name == "RS5" {
        bras::Setup::Partition
    } else {
        bras::Setup::Block
    };
    let positions = read_positions(input.join(format!("{scene_name}/positions.csv")))?;
    fs::create_dir_all(&output)?;

    let mut runs = BufWriter::new(fs::File::create(
        output.join(format!("{scene_name}-runs.csv")),
    )?);
    writeln!(
        runs,
        "case,seed,rays,bounces,radius,diffraction,sample_rate,cold_ms,p50_ms,p95_ms,p99_ms,max_ms,repeats,warmups,revision,reverse"
    )?;
    let mut raw_times = BufWriter::new(fs::File::create(
        output.join(format!("{scene_name}-latencies.csv")),
    )?);
    writeln!(
        raw_times,
        "case,seed,repeat,listener_y,trace_readback_ms,synthesis_ms,total_ms"
    )?;
    let gpu = VkBackend::new()?;
    let revision = provenance::revision();
    for seed in 7..end_seed {
        let mut scene = SceneManager::new();
        let description = bras::scene(setup, radius, diffraction);
        scene.load(description)?;
        let config = AcousticPipelineConfig {
            sound: SoundConfig {
                output_channels: OutputChannels::Mono,
                sample_rate: 44_100,
                ir_num_samples: 11_025,
            },
            tracing: AcousticsConfig {
                listener_ray_count: rays,
                max_bounces: bounces,
                maximum_path_length: 343.0 * 0.04,
                minimum_throughput: 0.0,
                sampling_seed: seed,
            },
            binaural: BinauralConfig::default(),
        };
        let mut pipeline = AcousticPipeline::new(&gpu, config)?;

        for (index, pair) in positions.iter().enumerate() {
            if selected_case
                .as_deref()
                .is_some_and(|name| name != pair.name)
            {
                continue;
            }
            let mut query = AcousticQuery {
                query_id: index as u32,
                source: PointSource::new(pair.source),
                listener: ListenerPose::new(pair.listener, -Vec3::X, Vec3::Y),
            };
            // Reciprocity diagnostics swap endpoints without changing geometry,
            // source gain, or calibration; the runs CSV records this explicitly.
            if reverse {
                std::mem::swap(&mut query.source.position, &mut query.listener.position);
            }
            let mut last_ir = None;
            let mut timings = Vec::with_capacity(repeats);
            let mut cold_ms = 0.0;
            // Cold query and warmups are explicit. No CSV serialization or file
            // writes occur inside any measured interval, including synthesis.
            for repetition in 0..1 + warmups + repeats {
                let mut traced_query = query;
                let step = repetition.saturating_sub(1 + warmups);
                traced_query.listener.position.y += motion_height
                    * (std::f32::consts::TAU * step as f32 / repeats.max(2) as f32).sin();
                let start = Instant::now();
                last_ir = Some(pipeline.build_ir(&gpu, &mut scene, traced_query)?);
                let trace_readback_ms = start.elapsed().as_secs_f64() * 1000.0;
                let total_ms = start.elapsed().as_secs_f64() * 1000.0;
                if repetition == 0 {
                    cold_ms = total_ms;
                }
                if repetition > warmups {
                    timings.push((
                        trace_readback_ms,
                        total_ms - trace_readback_ms,
                        total_ms,
                        traced_query.listener.position.y,
                    ));
                }
            }
            for (index, &(trace, synthesis, total, y)) in timings.iter().enumerate() {
                writeln!(
                    raw_times,
                    "{},{seed},{index},{y},{trace},{synthesis},{total}",
                    pair.name
                )?;
            }
            // Accuracy exports always use the published measurement coordinates.
            // A motion benchmark must not silently relabel a displaced receiver.
            if motion_height > 0.0 {
                last_ir = Some(pipeline.build_ir(&gpu, &mut scene, query)?);
            }

            let mut times: Vec<_> = timings.iter().map(|t| t.2).collect();
            times.sort_by(f64::total_cmp);
            export_ir(
                last_ir.as_ref().unwrap(),
                output.join(format!("{}-seed{seed}.csv", pair.name)),
            )?;

            writeln!(
                runs,
                "{},{seed},{rays},{bounces},{radius},{diffraction},44100,{cold_ms},{},{},{},{},{repeats},{warmups},{revision},{reverse}",
                pair.name,
                timing::percentile(&times, 0.5),
                timing::percentile(&times, 0.95),
                timing::percentile(&times, 0.99),
                times.last().unwrap()
            )?;
        }
        eprintln!("{scene_name}: completed seed {seed}");
    }
    runs.flush()?;
    raw_times.flush()?;
    Ok(())
}
