//! Reproducible CGA02 geometry and ray-traced early IR exports.
//! Run with --help for parameters. No measured waveforms are synthesized here.
use sound_engine::acoustics::AcousticPipelineConfig;
use sound_engine::{ListenerPose, PointSource};
#[path = "support/provenance.rs"]
mod provenance;
#[path = "support/timing.rs"]
mod timing;

use acoustic_fixtures::bell_box;
use glam::Vec3;
use sound_engine::acoustics::export_ir;
use sound_engine::acoustics::{AcousticPipeline, AcousticQuery};
use sound_engine::core::config::*;
use sound_engine::gpu::backend::VkBackend;
use sound_engine::scene::SceneManager;
use std::{fs, io::Write, path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rays = 262_144_u32;
    let mut repeats = 21_usize;
    let mut seeds = 3_u32;
    let mut absorption = 0.1_f32;
    let mut radius = 0.05_f32;
    let mut output = PathBuf::from("output/bell-labs-box-ray-local");
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--help" {
            println!(
                "bell_labs_box [--rays 262144] [--repeats 21] [--seeds 3] [--absorption 0.1] [--radius 0.05] [--output output/bell-labs-box-ray-local]"
            );
            return Ok(());
        }
        let value = args.next().ok_or("missing option value")?;
        match arg.as_str() {
            "--rays" => rays = value.parse()?,
            "--repeats" => repeats = value.parse()?,
            "--seeds" => seeds = value.parse()?,
            "--absorption" => absorption = value.parse()?,
            "--radius" => radius = value.parse()?,
            "--output" => output = value.into(),
            _ => return Err(format!("unknown option: {arg}").into()),
        }
    }
    if rays == 0
        || repeats == 0
        || seeds == 0
        || !absorption.is_finite()
        || !(0.0..=1.0).contains(&absorption)
        || !radius.is_finite()
        || radius <= 0.0
    {
        return Err("counts/radius must be positive and absorption must be in [0,1]".into());
    }
    let seed_end = 7_u32.checked_add(seeds).ok_or("seed range exceeds u32")?;
    fs::create_dir_all(&output)?;
    let mut timings = std::io::BufWriter::new(fs::File::create(output.join("runs.csv"))?);
    writeln!(
        timings,
        "revision,case,seed,rays,bounces,absorption,radius,sample_rate,cold_ms,p50_ms,p95_ms,p99_ms,max_ms,readback_ms"
    )?;
    let revision = provenance::revision();
    let gpu = VkBackend::new()?;
    for (name, baffle, diffraction, bounces) in [
        ("empty", false, false, 10),
        ("baffle_proxy", true, true, 6),
        ("baffle_no_diffraction", true, false, 8),
    ] {
        for seed in 7..seed_end {
            let mut scene = SceneManager::new();
            scene.load(bell_box::scene(baffle, diffraction, absorption, radius))?;
            let config = AcousticPipelineConfig {
                sound: SoundConfig {
                    output_channels: OutputChannels::Mono,
                    sample_rate: 48_000,
                    ir_num_samples: 2048,
                },
                tracing: AcousticsConfig {
                    listener_ray_count: rays,
                    max_bounces: bounces,
                    maximum_path_length: 343.0 * 2048.0 / 48_000.0,
                    minimum_throughput: 0.0,
                    sampling_seed: seed,
                },
                binaural: BinauralConfig::default(),
            };
            let mut pipeline = AcousticPipeline::new(&gpu, config)?;

            let query = AcousticQuery {
                query_id: seed,
                source: PointSource::new(if baffle {
                    bell_box::BAFFLE_SOURCE
                } else {
                    bell_box::EMPTY_SOURCE
                }),
                listener: ListenerPose::new(
                    if baffle {
                        bell_box::BAFFLE_LISTENER
                    } else {
                        bell_box::EMPTY_LISTENER
                    },
                    -Vec3::Z,
                    Vec3::Y,
                ),
            };
            let start = Instant::now();
            let mut ir = pipeline.build_ir_gpu(&gpu, &mut scene, query)?;
            let cold_ms = start.elapsed().as_secs_f64() * 1000.0;
            let mut times = Vec::with_capacity(repeats);
            for _ in 0..repeats {
                let start = Instant::now();
                ir = pipeline.build_ir_gpu(&gpu, &mut scene, query)?;
                times.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            let start = Instant::now();
            let snapshot = ir.download(&gpu)?;
            let readback_ms = start.elapsed().as_secs_f64() * 1000.0;
            export_ir(&snapshot, output.join(format!("{name}-seed{seed}.csv")))?;
            times.sort_by(f64::total_cmp);
            let percentile = |p: f64| timing::percentile(&times, p);
            writeln!(
                timings,
                "{revision},{name},{seed},{rays},{bounces},{absorption},{radius},48000,{cold_ms},{},{},{},{},{readback_ms}",
                percentile(0.5),
                percentile(0.95),
                percentile(0.99),
                times.last().unwrap()
            )?;
            eprintln!("{name}, seed {seed}: {:.3} ms median", percentile(0.5));
        }
    }
    timings.flush()?;
    Ok(())
}

#[cfg(test)]
#[path = "../tests/examples/bell_labs_box.rs"]
mod tests;
