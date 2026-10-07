//! File-driven BRAS validation using the unchanged scalar ray transport.
//! Fixtures contain triangles, broadband materials and authored diffraction edges.

use acoustic_fixtures::{read_positions, read_scene};
use glam::Vec3;
use sound_engine::acoustics::{AcousticPipeline, AcousticPipelineConfig, AcousticQuery};
use sound_engine::core::config::*;
use sound_engine::gpu::backend::VkBackend;
use sound_engine::scene::SceneManager;
use sound_engine::{ListenerPose, PointSource};
use std::{
    collections::BTreeMap,
    fs,
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut options = BTreeMap::new();
    let mut args = std::env::args().skip(1);
    while let Some(key) = args.next() {
        options.insert(key, args.next().ok_or("missing option value")?);
    }
    let number = |key: &str, default: u32| -> Result<u32, Box<dyn std::error::Error>> {
        Ok(options
            .get(key)
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(default))
    };
    let rays = number("--rays", 65_536)?;
    let bounces = number("--bounces", 3)?;
    let seeds = number("--seeds", 3)?;
    let repeats = number("--repeats", 5)?;
    let duration: f32 = options
        .get("--seconds")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(0.25);
    // Bound each offline query to previously exercised allocation scales.
    if rays == 0
        || rays > 65_536
        || bounces > 30
        || bounces == 0
        || seeds == 0
        || seeds > 3
        || repeats == 0
        || !duration.is_finite()
        || !(0.04..=4.5).contains(&duration)
    {
        return Err("benchmark budget outside bounds".into());
    }
    let input = PathBuf::from(options.get("--fixture").ok_or("missing --fixture")?);
    let positions = read_positions(options.get("--positions").ok_or("missing --positions")?)?;
    let output = PathBuf::from(options.get("--output").ok_or("missing --output")?);
    fs::create_dir_all(&output)?;
    let description = read_scene(&input)?;
    let mut log = BufWriter::new(fs::File::create(output.join("runs.csv"))?);
    writeln!(
        log,
        "case,seed,rays,bounces,seconds,cold_ms,warm_median_ms,warm_max_ms,nonzero_samples,peak"
    )?;
    let gpu = VkBackend::new()?;
    for seed in 7..7 + seeds {
        let config = AcousticPipelineConfig {
            sound: SoundConfig {
                output_channels: OutputChannels::Mono,
                sample_rate: 44_100,
                ir_num_samples: (duration * 44_100.0).ceil() as u32,
            },
            tracing: AcousticsConfig {
                listener_ray_count: rays,
                max_bounces: bounces,
                maximum_path_length: 343.0 * duration,
                minimum_throughput: 0.0,
                sampling_seed: seed,
            },
            binaural: BinauralConfig::default(),
        };
        let mut pipeline = AcousticPipeline::new(&gpu, config)?;
        let mut scene = SceneManager::new();
        scene.load(description.clone())?;
        for (index, pair) in positions.iter().enumerate() {
            if options.get("--case").is_some_and(|name| name != &pair.name) {
                continue;
            }
            let query = AcousticQuery {
                query_id: index as u32,
                source: PointSource::new(pair.source),
                listener: ListenerPose::new(pair.listener, -Vec3::X, Vec3::Y),
            };
            let mut times = Vec::new();
            let mut result = None;
            for _ in 0..repeats + 1 {
                let start = Instant::now();
                result = Some(pipeline.build_ir(&gpu, &mut scene, query)?);
                times.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            let cold = times.remove(0);
            times.sort_by(f64::total_cmp);
            let ir = result.unwrap();
            let mut file = BufWriter::new(fs::File::create(
                output.join(format!("{}-seed{seed}.f32", pair.name)),
            )?);
            let mut count = 0;
            let mut peak = 0.0_f32;
            for sample in ir.samples {
                if !sample.left.is_finite() {
                    return Err("non-finite ray IR".into());
                }
                count += usize::from(sample.left != 0.0);
                peak = peak.max(sample.left.abs());
                file.write_all(&sample.left.to_le_bytes())?;
            }
            writeln!(
                log,
                "{},{seed},{rays},{bounces},{duration},{cold},{},{},{count},{peak}",
                pair.name,
                times[times.len() / 2],
                times.last().unwrap()
            )?;
            log.flush()?;
        }
        eprintln!("completed seed {seed}");
    }
    Ok(())
}
