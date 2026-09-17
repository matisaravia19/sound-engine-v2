//! Reproducible CGA02 geometry and ray-traced early IR exports.
//! Run with --help for parameters. No measured waveforms are synthesized here.
#[path = "support/bell_box.rs"]
mod bell_box;
#[path = "support/geometry.rs"]
mod geometry;
#[path = "support/provenance.rs"]
mod provenance;
#[path = "support/timing.rs"]
mod timing;

use glam::Vec3;
use sound_engine::acoustics::{AcousticPipeline, AcousticQuery};
use sound_engine::core::config::*;
use sound_engine::core::debug::export_ir;
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
            let config = EngineConfig {
                sound: SoundConfig {
                    output_channels: OutputChannels::Mono,
                    sample_rate: 48_000,
                    ir_num_samples: 2048,
                },
                acoustics: AcousticsConfig {
                    listener_ray_count: rays,
                    max_bounces: bounces,
                    maximum_path_length: 343.0 * 2048.0 / 48_000.0,
                    minimum_throughput: 0.0,
                    sampling_seed: seed,
                },
                cache: IrCacheConfig::default(),
                binaural: BinauralConfig::default(),
                auralization: AuralizationConfig {
                    block_size: 1024,
                },
            };
            let mut pipeline = AcousticPipeline::new(&gpu, config)?;

            let query = AcousticQuery {
                query_id: seed,
                source_position: if baffle {
                    bell_box::BAFFLE_SOURCE
                } else {
                    bell_box::EMPTY_SOURCE
                },
                listener_position: if baffle {
                    bell_box::BAFFLE_LISTENER
                } else {
                    bell_box::EMPTY_LISTENER
                },
                listener_forward: -Vec3::Z,
                listener_up: Vec3::Y,
                source_amplitude: 1.0,
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
            let snapshot = pipeline.download_gpu_ir(&gpu, &ir)?;
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
mod tests {
    use super::*;

    // Backend setup shares process-wide Vulkan state; serialize GPU fixtures.
    static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    #[ignore = "requires a Vulkan ray-tracing device and glslangValidator"]
    fn diffraction_requires_visible_ray_traced_geometry() {
        let _guard = GPU_TEST_LOCK.lock().unwrap();
        let gpu = VkBackend::new().unwrap();
        let config = EngineConfig {
            sound: SoundConfig {
                output_channels: OutputChannels::Mono,
                sample_rate: 48_000,
                ir_num_samples: 2048,
            },
            acoustics: AcousticsConfig {
                listener_ray_count: 262_144,
                max_bounces: 1,
                maximum_path_length: 14.0,
                minimum_throughput: 0.0,
                sampling_seed: 7,
            },
            cache: IrCacheConfig::default(),
            binaural: BinauralConfig::default(),
            auralization: AuralizationConfig {
                block_size: 1024,
            },
        };
        let mut pipeline = AcousticPipeline::new(&gpu, config).unwrap();

        let query = AcousticQuery {
            query_id: 1,
            source_position: bell_box::BAFFLE_SOURCE,
            listener_position: bell_box::BAFFLE_LISTENER,
            listener_forward: -Vec3::Z,
            listener_up: Vec3::Y,
            source_amplitude: 1.0,
        };
        for variant in 0..3 {
            let mut description = bell_box::scene(true, true, 1.0, 0.05);
            if variant == 1 {
                // Removing the edge disables both connection and continuation.
                description.diffraction_edges.clear();
            } else if variant == 2 {
                // A fully absorbing cross-wall blocks listener-to-edge traversal.
                description.meshes[0].vertices.extend(geometry::quad_z(
                    1.5,
                    0.0,
                    bell_box::SIZE.x,
                    0.0,
                    bell_box::SIZE.y,
                ));
            }
            let mut scene = SceneManager::new();
            scene.load(description).unwrap();
            let ir = pipeline.build_ir(&gpu, &mut scene, query).unwrap();
            assert!(ir.samples.iter().all(|sample| sample.left.is_finite()));
            if variant == 0 {
                let first = ir
                    .samples
                    .iter()
                    .position(|sample| sample.left.abs() > 1e-8)
                    .unwrap();
                // The four-tap cubic delay starts one sample before floor(arrival).
                // Earlier pressure would indicate a path shorter than the geometry.
                let earliest = (1.6758_f32 / 343.0 * 48000.0).floor() as usize - 1;
                assert!(
                    first >= earliest,
                    "arrival {first} precedes cubic support {earliest}"
                );
                assert!(ir.energy > 0.0);
            } else {
                assert!(
                    ir.samples.iter().all(|sample| sample.left == 0.0),
                    "variant {variant} bypassed traversal"
                );
            }
        }
    }

    #[test]
    #[ignore = "requires a Vulkan ray-tracing device and glslangValidator"]
    fn multiple_edge_paths_require_each_event_and_preserve_source_scale() {
        let _guard = GPU_TEST_LOCK.lock().unwrap();
        use glam::vec3;
        use sound_engine::scene::DiffractionEdge;
        let gpu = VkBackend::new().unwrap();
        let mut config = EngineConfig {
            sound: SoundConfig {
                output_channels: OutputChannels::Mono,
                sample_rate: 48_000,
                ir_num_samples: 8192,
            },
            acoustics: AcousticsConfig {
                listener_ray_count: 1_048_576,
                max_bounces: 2,
                maximum_path_length: 50.0,
                minimum_throughput: 0.0,
                sampling_seed: 7,
            },
            cache: IrCacheConfig::default(),
            binaural: BinauralConfig::default(),
            auralization: AuralizationConfig {
                block_size: 1024,
            },
        };
        for edge_count in [2, 3] {
            let mut reference = Vec::<f32>::new();
            let variants = [
                (0, 1.0, edge_count),     // Reference needs every edge event.
                (1, 0.5, edge_count),     // Source scaling must survive all events.
                (2, 1.0, edge_count - 1), // A missing event cannot reach the source.
                (3, 0.0, edge_count),     // Silent sources must not inject pressure.
            ];

            for (index, amplitude, bounces) in variants {
                // Alternating absorbing half-screens require every edge in order.
                // The three-screen case detects an independent two-edge cutoff.
                let walls = (0..edge_count)
                    .map(|edge_index| {
                        let z = 3.0 + 4.0 * edge_index as f32;
                        let (left, right) = if edge_index % 2 == 0 {
                            (-0.5, 40.0)
                        } else {
                            (-40.0, 0.5)
                        };
                        geometry::quad_z(z, left, right, -40.0, 40.0)
                    })
                    .collect();
                let mut description = geometry::scene_from_walls(walls, 1.0);
                for edge_index in 0..edge_count {
                    let x = if edge_index % 2 == 0 {
                        -0.5
                    } else {
                        0.5
                    };
                    let z = 3.0 + 4.0 * edge_index as f32;
                    let bisector = if edge_index % 2 == 0 {
                        -Vec3::X
                    } else {
                        Vec3::X
                    };
                    description.diffraction_edges.push(DiffractionEdge {
                        id: edge_index + 1,
                        start: vec3(x, -40.0, z),
                        end: vec3(x, 40.0, z),
                        bisector_dir: bisector,
                        edge_angle_radians: std::f32::consts::TAU,
                        diffraction_radius: 0.3,
                    });
                }
                let mut scene = SceneManager::new();
                scene.load(description).unwrap();
                config.acoustics.max_bounces = bounces;
                let mut pipeline = AcousticPipeline::new(&gpu, config).unwrap();
                let ir = pipeline
                    .build_ir(
                        &gpu,
                        &mut scene,
                        AcousticQuery {
                            query_id: index,
                            source_position: vec3(
                                if edge_count % 2 == 0 {
                                    -0.5
                                } else {
                                    0.5
                                },
                                0.0,
                                4.0 * edge_count as f32 + 2.0,
                            ),
                            listener_position: vec3(0.5, 0.0, 0.0),
                            listener_forward: Vec3::Z,
                            listener_up: Vec3::Y,
                            source_amplitude: amplitude,
                        },
                    )
                    .unwrap();
                let samples = ir.samples.iter().map(|s| s.left).collect::<Vec<_>>();
                assert!(samples.iter().all(|s| s.is_finite()));
                eprintln!(
                    "edges={edge_count} budget={bounces} amplitude={amplitude} energy={}",
                    ir.energy
                );
                if index == 0 {
                    assert!(
                        ir.energy > 0.0,
                        "test must discover a {edge_count}-edge path"
                    );
                    reference = samples;
                } else if index >= 2 {
                    assert!(
                        samples.iter().all(|&s| s == 0.0),
                        "insufficient events or a silent source must contribute nothing"
                    );
                } else {
                    let expected_gain = 0.5;
                    let residual: f64 = samples
                        .iter()
                        .zip(&reference)
                        .map(|(&s, &r)| (s as f64 - expected_gain * r as f64).powi(2))
                        .sum();
                    let norm: f64 = reference.iter().map(|&s| (s as f64).powi(2)).sum();
                    assert!(
                        (residual / norm).sqrt() < 1e-4,
                        "ray pressure inheritance failed in variant {index}"
                    );
                }
            }
        }
    }
}
