//! Measured BRAS RS5/RS6 geometry: every exported propagation IR comes from rays.
//! Preparation and calibration: scripts/bras_reference.py; citations: references.md.
#[path = "support/geometry.rs"]
mod geometry;
#[path = "support/provenance.rs"]
mod provenance;
#[path = "support/timing.rs"]
mod timing;

#[cfg(test)]
#[path = "support/mixed_surface_tests.rs"]
mod mixed_surface_tests;
#[cfg(test)]
#[path = "support/surface_two_edge_tests.rs"]
mod surface_two_edge_tests;

use glam::{Mat4, Vec3, vec3};
use sound_engine::acoustics::{AcousticPipeline, AcousticQuery, ExperimentalUtdPipeline};
use sound_engine::core::config::*;
use sound_engine::core::debug::export_ir;
use sound_engine::gpu::backend::VkBackend;
use sound_engine::scene::{
    DiffractionEdge, Material, MeshAsset, SceneDescription, SceneManager, SceneObject,
};
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
    let mut edge_strength = 1.0_f32;
    let mut diffraction = true;
    let mut utd = false;
    let mut utd_bands = 0_u32;
    let mut utd_orders = 1_u32;
    let mut reverse = false;
    let mut source_surface_paths = false;
    let mut surface_two_edge_paths = false;
    let mut source_profile_path: Option<PathBuf> = None;
    let mut cone_width = 0.04_f32;
    let mut maximum_frequency_hz = 2000.0_f32;
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
                "bras_diffraction [--scene RS5|RS6] [--rays 262144] [--seeds 3] [--bounces 3] [--radius 0.2] [--edge-strength 1.0] [--diffraction true|false] [--utd true|false] [--utd-bands 16] [--utd-orders 1|2] [--reverse true|false] [--source-surface-paths true|false] [--surface-two-edge-paths true|false] [--source-directivity profile.dir1] [--cone-width 0.04] [--max-frequency 2000] [--repeats 1] [--warmups 3] [--case RS5_LS1_MP1] [--motion-height 0] [--input output/measured-reference-research] [--output output/bras-ray-baseline]"
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
            "--edge-strength" => edge_strength = value.parse()?,
            "--utd-orders" => utd_orders = value.parse()?,
            "--reverse" => reverse = value.parse()?,
            "--source-surface-paths" => source_surface_paths = value.parse()?,
            "--surface-two-edge-paths" => surface_two_edge_paths = value.parse()?,
            "--source-directivity" => source_profile_path = Some(value.into()),
            "--diffraction" => diffraction = value.parse()?,
            "--utd" => utd = value.parse()?,
            "--utd-bands" => {
                utd_bands = value.parse()?;
                utd = true;
            }
            "--cone-width" => cone_width = value.parse()?,
            "--max-frequency" => maximum_frequency_hz = value.parse()?,
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
        || !edge_strength.is_finite()
        || !(0.0..=1.0).contains(&edge_strength)
    {
        return Err("invalid scene, ray/seed count or radius".into());
    }
    let end_seed = 7_u32.checked_add(seeds).ok_or("seed range overflow")?;
    let source_surface_paths = source_surface_paths && utd && utd_orders == 2 && utd_bands > 0;
    if surface_two_edge_paths && (!utd || utd_orders != 2 || utd_bands == 0 || bounces < 3) {
        return Err("surface/two-edge paths require UTD bands, two edge orders and at least three interactions".into());
    }
    if !(1..=2).contains(&utd_orders) || (utd_orders == 2 && utd_bands == 0) {
        return Err("UTD orders must be 1 or 2; two orders require --utd-bands".into());
    }
    if !motion_height.is_finite() || motion_height < 0.0 {
        return Err("motion height must be finite and nonnegative".into());
    }
    let positions = fs::read_to_string(input.join(format!("{scene_name}/positions.csv")))?;
    if source_profile_path.is_some() && (!utd || utd_bands == 0 || reverse) {
        return Err("source directivity requires UTD bands and forward measured-source geometry; reciprocity diagnostics use omnidirectional endpoints".into());
    }
    let source_profile = source_profile_path
        .as_ref()
        .map(sound_engine::acoustics::SourceDirectivity::read)
        .transpose()?;
    let source_directivity = source_profile.is_some();
    let two_edge_seed_scheme = if utd && utd_bands > 0 && utd_orders == 2 {
        "hashed-seed-v2"
    } else {
        "not-applicable"
    };
    fs::create_dir_all(&output)?;
    if let Some(path) = &source_profile_path {
        fs::copy(path, output.join("source-directivity.dir1"))?;
        if path.with_extension("json").exists() {
            fs::copy(
                path.with_extension("json"),
                output.join("source-directivity.json"),
            )?;
        }
        fs::write(
            output.join("source-directivity-orientation.txt"),
            "BRAS source orientation from RS5_RIR.png and RS6_RIR.png: local +X forward, +Y up. RS5 LS4 tilted up 34.6 degrees (rotation around engine +Z); all other RS5 and RS6 sources face +X. Angular response is relative to on-axis calibration, which must be applied once separately.\n",
        )?;
    }
    let mut runs = BufWriter::new(fs::File::create(
        output.join(format!("{scene_name}-runs.csv")),
    )?);
    writeln!(
        runs,
        "case,seed,rays,bounces,radius,diffraction,utd,cone_width,max_frequency_hz,sample_rate,cold_ms,p50_ms,p95_ms,p99_ms,max_ms,repeats,warmups,revision,utd_bands,edge_strength,utd_orders,reverse,source_surface_paths,source_directivity,surface_two_edge_paths,two_edge_seed_scheme"
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
        let mut description = benchmark_scene(&scene_name, radius, diffraction);
        for edge in &mut description.diffraction_edges {
            edge.base_strength = edge_strength;
        }
        scene.load(description)?;
        let config = EngineConfig {
            sound: SoundConfig {
                output_channels: OutputChannels::Mono,
                sample_rate: 44_100,
                ir_num_samples: if utd {
                    8192
                } else {
                    11_025
                },
            },
            acoustics: AcousticsConfig {
                listener_ray_count: rays,
                max_bounces: bounces,
                maximum_path_length: 343.0 * 0.04,
                minimum_throughput: 0.0,
                sampling_seed: seed,
            },
            cache: IrCacheConfig::default(),
            binaural: BinauralConfig::default(),
            auralization: AuralizationConfig {
                block_size: 1024,
            },
        };
        let mut pipeline = if utd {
            None
        } else {
            Some(AcousticPipeline::new(&gpu, config)?)
        };
        let mut spectral = if utd_bands > 0 {
            Some(ExperimentalUtdPipeline::with_band_orders(
                &gpu,
                config,
                cone_width,
                maximum_frequency_hz,
                utd_bands,
                utd_orders,
            )?)
        } else if utd {
            Some(ExperimentalUtdPipeline::with_bandwidth(
                &gpu,
                config,
                cone_width,
                maximum_frequency_hz,
            )?)
        } else {
            None
        };
        if let Some(spectral) = &mut spectral {
            spectral.set_source_surface_paths(source_surface_paths);
            spectral.set_surface_two_edge_paths(surface_two_edge_paths);
        }
        let source_profile_id = if let Some(profile) = &source_profile {
            spectral
                .as_mut()
                .unwrap()
                .register_source_directivity(&gpu, profile)?
        } else {
            0
        };
        for (index, line) in positions.lines().skip(1).enumerate() {
            let fields: Vec<_> = line.split(',').collect();
            if fields.len() != 7 {
                return Err("positions.csv must have seven columns".into());
            }
            if selected_case
                .as_deref()
                .is_some_and(|name| name != fields[0])
            {
                continue;
            }
            let p: Vec<f32> = fields[1..]
                .iter()
                .map(|s| s.parse())
                .collect::<Result<_, _>>()?;
            let mut query = AcousticQuery {
                query_id: index as u32,
                source_position: vec3(p[0], p[1], p[2]),
                listener_position: vec3(p[3], p[4], p[5]),
                listener_forward: -Vec3::X,
                listener_up: Vec3::Y,
                source_amplitude: 1.0,
                source_radiation: sound_engine::acoustics::SourceRadiation {
                    profile: source_profile_id,
                    orientation: if scene_name == "RS5" && fields[0].starts_with("RS5_LS4_") {
                        glam::Quat::from_rotation_z(34.6_f32.to_radians())
                    } else {
                        glam::Quat::IDENTITY
                    },
                },
            };
            // Reciprocity diagnostics swap endpoints without changing geometry,
            // source gain, or calibration; the runs CSV records this explicitly.
            if reverse {
                std::mem::swap(&mut query.source_position, &mut query.listener_position);
            }
            let mut transfer = Vec::new();
            let mut last_ir = None;
            let mut timings = Vec::with_capacity(repeats);
            let mut cold_ms = 0.0;
            // Cold query and warmups are explicit. No CSV serialization or file
            // writes occur inside any measured interval, including synthesis.
            for repetition in 0..1 + warmups + repeats {
                let mut traced_query = query;
                let step = repetition.saturating_sub(1 + warmups);
                traced_query.listener_position.y += motion_height
                    * (std::f32::consts::TAU * step as f32 / repeats.max(2) as f32).sin();
                let start = Instant::now();
                let trace_readback_ms;
                if let Some(spectral) = &mut spectral {
                    if utd_bands > 0 {
                        let snapshot = spectral.build_ir_gpu(&gpu, &mut scene, traced_query)?;
                        trace_readback_ms = start.elapsed().as_secs_f64() * 1000.0;
                        last_ir = Some(spectral.download_gpu_ir(&gpu, &snapshot)?);
                    } else {
                        transfer = spectral.build_transfer(&gpu, &mut scene, traced_query)?;
                        trace_readback_ms = start.elapsed().as_secs_f64() * 1000.0;
                        std::hint::black_box(spectral.synthesize_ir(&transfer)?);
                    }
                } else {
                    last_ir = Some(pipeline.as_mut().unwrap().build_ir(
                        &gpu,
                        &mut scene,
                        traced_query,
                    )?);
                    trace_readback_ms = start.elapsed().as_secs_f64() * 1000.0;
                }
                let total_ms = start.elapsed().as_secs_f64() * 1000.0;
                if repetition == 0 {
                    cold_ms = total_ms;
                }
                if repetition > warmups {
                    timings.push((
                        trace_readback_ms,
                        total_ms - trace_readback_ms,
                        total_ms,
                        traced_query.listener_position.y,
                    ));
                }
            }
            for (index, &(trace, synthesis, total, y)) in timings.iter().enumerate() {
                writeln!(
                    raw_times,
                    "{},{seed},{index},{y},{trace},{synthesis},{total}",
                    fields[0]
                )?;
            }
            // Accuracy exports always use the published measurement coordinates.
            // A motion benchmark must not silently relabel a displaced receiver.
            if motion_height > 0.0 {
                if let Some(spectral) = &mut spectral {
                    if utd_bands > 0 {
                        let snapshot = spectral.build_ir_gpu(&gpu, &mut scene, query)?;
                        last_ir = Some(spectral.download_gpu_ir(&gpu, &snapshot)?);
                    } else {
                        transfer = spectral.build_transfer(&gpu, &mut scene, query)?;
                    }
                } else {
                    last_ir = Some(
                        pipeline
                            .as_mut()
                            .unwrap()
                            .build_ir(&gpu, &mut scene, query)?,
                    );
                }
            }
            let mut times: Vec<_> = timings.iter().map(|t| t.2).collect();
            times.sort_by(f64::total_cmp);
            if utd && utd_bands == 0 {
                let mut file = BufWriter::new(fs::File::create(
                    output.join(format!("{}-seed{seed}-spectrum.csv", fields[0])),
                )?);
                writeln!(file, "frequency_hz,real,imaginary")?;
                for (bin, value) in transfer.iter().enumerate() {
                    writeln!(
                        file,
                        "{},{},{}",
                        bin as f64 * 44100.0 / 8192.0,
                        value[0],
                        value[1]
                    )?;
                }
                file.flush()?;
            } else {
                export_ir(
                    last_ir.as_ref().unwrap(),
                    output.join(format!("{}-seed{seed}.csv", fields[0])),
                )?;
            }
            writeln!(
                runs,
                "{},{seed},{rays},{bounces},{radius},{diffraction},{utd},{cone_width},{maximum_frequency_hz},44100,{cold_ms},{},{},{},{},{repeats},{warmups},{revision},{utd_bands},{edge_strength},{utd_orders},{reverse},{source_surface_paths},{source_directivity},{surface_two_edge_paths},{two_edge_seed_scheme}",
                fields[0],
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

fn benchmark_scene(name: &str, radius: f32, diffraction: bool) -> SceneDescription {
    // BRAS uses Z up. Positions.csv explicitly maps (X,Y,Z) to engine (X,Z,Y).
    let mut scene =
        geometry::scene_from_walls(vec![geometry::quad_y(0.0, -20.0, 20.0, -20.0, 20.0)], 0.0);
    let (walls, edges) = if name == "RS5" {
        // Thin-screen approximation of the 25 mm partition, at its mid-plane.
        // Height and span come from RS5_RIR.png; no RIR was used to adjust them.
        let (x, height, z0, z1) = (5.4995, 2.066, 2.985 - 4.75 / 2.0, 2.985 + 4.75 / 2.0);
        (
            vec![geometry::quad_x(x, 0.0, height, z0, z1)],
            vec![
                edge(
                    1,
                    vec3(x, height, z0),
                    vec3(x, height, z1),
                    Vec3::Y,
                    std::f32::consts::TAU,
                    radius,
                ),
                edge(
                    2,
                    vec3(x, 0.0, z0),
                    vec3(x, height, z0),
                    -Vec3::Z,
                    std::f32::consts::TAU,
                    radius,
                ),
                edge(
                    3,
                    vec3(x, 0.0, z1),
                    vec3(x, height, z1),
                    Vec3::Z,
                    std::f32::consts::TAU,
                    radius,
                ),
            ],
        )
    } else {
        let (x0, x1, height, z0, z1) = (5.5 - 0.36, 5.5 + 0.36, 0.72, 2.985 - 2.07, 2.985 + 2.07);
        let n = std::f32::consts::FRAC_1_SQRT_2;
        let angle = 1.5 * std::f32::consts::PI;
        // The hollow-block assembly is represented by its measured outer envelope.
        (
            vec![
                geometry::quad_x(x0, 0.0, height, z0, z1),
                geometry::quad_x(x1, 0.0, height, z0, z1),
                geometry::quad_y(height, x0, x1, z0, z1),
                geometry::quad_z(z0, x0, x1, 0.0, height),
                geometry::quad_z(z1, x0, x1, 0.0, height),
            ],
            vec![
                edge(
                    1,
                    vec3(x0, height, z0),
                    vec3(x0, height, z1),
                    vec3(-n, n, 0.0),
                    angle,
                    radius,
                ),
                edge(
                    2,
                    vec3(x1, height, z0),
                    vec3(x1, height, z1),
                    vec3(n, n, 0.0),
                    angle,
                    radius,
                ),
                edge(
                    3,
                    vec3(x0, 0.0, z0),
                    vec3(x0, height, z0),
                    vec3(-n, 0.0, -n),
                    angle,
                    radius,
                ),
                edge(
                    4,
                    vec3(x1, 0.0, z0),
                    vec3(x1, height, z0),
                    vec3(n, 0.0, -n),
                    angle,
                    radius,
                ),
                edge(
                    5,
                    vec3(x0, 0.0, z1),
                    vec3(x0, height, z1),
                    vec3(-n, 0.0, n),
                    angle,
                    radius,
                ),
                edge(
                    6,
                    vec3(x1, 0.0, z1),
                    vec3(x1, height, z1),
                    vec3(n, 0.0, n),
                    angle,
                    radius,
                ),
                edge(
                    7,
                    vec3(x0, height, z0),
                    vec3(x1, height, z0),
                    vec3(0.0, n, -n),
                    angle,
                    radius,
                ),
                edge(
                    8,
                    vec3(x0, height, z1),
                    vec3(x1, height, z1),
                    vec3(0.0, n, n),
                    angle,
                    radius,
                ),
            ],
        )
    };
    scene.meshes.push(MeshAsset {
        id: 2,
        vertices: walls.into_iter().flatten().collect(),
        indices: Vec::new(),
        opaque: true,
    });
    // Rigid boundary is explicit pending use of the supplied MDF material data.
    scene.materials.push(Material {
        id: 2,
        absorption_bands: vec![0.0],
        scattering: 0.0,
        roughness: 0.0,
        transmission: None,
    });
    scene.objects.push(SceneObject {
        id: 2,
        mesh_id: 2,
        material_id: 2,
        transform: Mat4::IDENTITY,
        active: true,
    });
    if diffraction {
        scene.diffraction_edges = edges;
    }
    scene
}

fn edge(
    id: u32,
    start: Vec3,
    end: Vec3,
    bisector_dir: Vec3,
    angle: f32,
    radius: f32,
) -> DiffractionEdge {
    DiffractionEdge {
        id,
        start,
        end,
        bisector_dir,
        edge_angle_radians: angle,
        diffraction_radius: radius,
        base_strength: 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// These tests each initialize Vulkan and VkFFT resources. Concurrent
    /// initialization stalled on the validation host; keep hardware tests serial.
    /// Serializes GPU regressions that share driver and FFT initialization state.
    pub(super) static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    #[ignore = "requires Vulkan ray tracing and glslangValidator"]
    fn utd_two_edge_reciprocity_and_visibility() {
        let _guard = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let gpu = VkBackend::new().unwrap();
        let config = EngineConfig {
            sound: SoundConfig {
                output_channels: OutputChannels::Mono,
                sample_rate: 44100,
                ir_num_samples: 4096,
            },
            acoustics: AcousticsConfig {
                listener_ray_count: 262144,
                max_bounces: 2,
                maximum_path_length: 15.0,
                minimum_throughput: 0.0,
                sampling_seed: 7,
            },
            cache: IrCacheConfig::default(),
            binaural: BinauralConfig::default(),
            auralization: AuralizationConfig {
                block_size: 1024,
            },
        };
        // Absorbing faces isolate the edge pressure. The two exposed top edges
        // are the only route between endpoints below opposite sides of the box.
        let mut description = geometry::scene_from_walls(
            vec![
                geometry::quad_x(-0.36, -10.0, 1.0, -10.0, 10.0),
                geometry::quad_x(0.36, -10.0, 1.0, -10.0, 10.0),
                geometry::quad_y(1.0, -0.36, 0.36, -10.0, 10.0),
            ],
            1.0,
        );
        let n = std::f32::consts::FRAC_1_SQRT_2;
        description.diffraction_edges = vec![
            edge(
                1,
                vec3(-0.36, 1.0, -10.0),
                vec3(-0.36, 1.0, 10.0),
                vec3(-n, n, 0.0),
                1.5 * std::f32::consts::PI,
                0.1,
            ),
            edge(
                2,
                vec3(0.36, 1.0, -10.0),
                vec3(0.36, 1.0, 10.0),
                vec3(n, n, 0.0),
                1.5 * std::f32::consts::PI,
                0.1,
            ),
        ];
        let mut scene = SceneManager::new();
        scene.load(description.clone()).unwrap();
        let query = AcousticQuery {
            query_id: 0,
            source_position: vec3(-3.0, 0.2, 0.0),
            listener_position: vec3(1.5, 0.4, 0.0),
            listener_forward: -Vec3::X,
            listener_up: Vec3::Y,
            source_amplitude: 1.0,
            source_radiation: Default::default(),
        };
        let mut pipeline =
            ExperimentalUtdPipeline::with_band_orders(&gpu, config, 0.02, 20000.0, 16, 2).unwrap();
        let source_profile = sound_engine::acoustics::SourceDirectivity::new(
            1,
            19,
            vec![20.0, 20000.0],
            (0..19)
                .flat_map(|t| [[1.0 - 0.5 * t as f32 / 18.0, 0.0]; 2])
                .collect(),
        )
        .unwrap();
        let source_profile_id = pipeline
            .register_source_directivity(&gpu, &source_profile)
            .unwrap();
        let mut run = |scene: &mut SceneManager, query| {
            let snapshot = pipeline.build_ir_gpu(&gpu, scene, query).unwrap();
            pipeline.download_gpu_ir(&gpu, &snapshot).unwrap()
        };
        let forward = run(&mut scene, query);
        assert!(forward.energy > 1e-8, "two-edge route was missed");
        assert!(forward.samples.iter().all(|s| s.left.is_finite()));
        // A power-of-two launch must not repeat the same ray set when the seed
        // changes. XORing a small seed directly with the index used to do that.
        let mut alternate_config = config;
        alternate_config.acoustics.sampling_seed = 8;
        let mut alternate =
            ExperimentalUtdPipeline::with_band_orders(&gpu, alternate_config, 0.02, 20000.0, 16, 2)
                .unwrap();
        let alternate_raw = alternate.build_ir_gpu(&gpu, &mut scene, query).unwrap();
        let alternate_ir = alternate.download_gpu_ir(&gpu, &alternate_raw).unwrap();
        let seed_difference = (alternate_ir
            .samples
            .iter()
            .zip(&forward.samples)
            .map(|(a, b)| (a.left - b.left).powi(2))
            .sum::<f32>()
            / forward.energy)
            .sqrt();
        eprintln!("two-edge independent-seed relative difference={seed_difference}");
        assert!(
            seed_difference > 1e-4,
            "different seeds repeated the same diffraction samples"
        );
        // Independent stationary values from two_edge_stationary_reference.py
        // check absolute complex pressure, including geometric normalization.
        let fft = realfft::RealFftPlanner::<f32>::new().plan_fft_forward(4096);
        let mut spectrum = fft.make_output_vec();
        let mut samples: Vec<f32> = forward.samples.iter().map(|s| s.left).collect();
        fft.process(&mut samples, &mut spectrum).unwrap();
        for (bin, expected) in [
            (23, [-0.09844568_f32, 0.03016060]),
            (93, [0.04693347, -0.02159182]),
            (163, [-0.03284891, 0.01404044]),
        ] {
            let actual = spectrum[bin];
            let relative = ((actual.re - expected[0]).powi(2) + (actual.im - expected[1]).powi(2))
                .sqrt()
                / expected[0].hypot(expected[1]);
            eprintln!(
                "two-edge bin={bin}: {actual:?}, reference={expected:?}, relative={relative}"
            );
            assert!(
                relative < 0.08,
                "two-edge stationary normalization mismatch"
            );
        }
        let mut directional_query = query;
        directional_query.source_radiation.profile = source_profile_id;
        let directional = run(&mut scene, directional_query);
        let expected_gain = 1.0 - 0.5 * 0.8_f32.atan2(2.64) / std::f32::consts::PI;
        let direction_error = (directional
            .samples
            .iter()
            .zip(&forward.samples)
            .map(|(a, b)| (a.left - expected_gain * b.left).powi(2))
            .sum::<f32>()
            / forward.energy)
            .sqrt();
        eprintln!(
            "two-edge source direction error={direction_error}, expected gain={expected_gain}"
        );
        assert!(
            direction_error < 0.01,
            "two-edge ray used listener arrival instead of source departure"
        );
        let reverse = run(
            &mut scene,
            AcousticQuery {
                source_position: query.listener_position,
                listener_position: query.source_position,
                ..query
            },
        );
        let residual = forward
            .samples
            .iter()
            .zip(&reverse.samples)
            .map(|(a, b)| (a.left - b.left).powi(2))
            .sum::<f32>()
            / forward.energy;
        eprintln!(
            "two-edge reversal relative L2={}, energy ratio={}",
            residual.sqrt(),
            reverse.energy / forward.energy
        );
        // Finite ray counts and cone width prevent exact samplewise equality.
        assert!(
            residual.sqrt() < 0.04,
            "two-edge reciprocal pressure mismatch"
        );
        let half = run(
            &mut scene,
            AcousticQuery {
                source_amplitude: 0.5,
                ..query
            },
        );
        assert!((half.energy / forward.energy - 0.25).abs() < 0.001);
        for index in 0..2 {
            let mut disabled = description.clone();
            disabled.diffraction_edges[index].base_strength = 0.0;
            scene.load(disabled).unwrap();
            assert_eq!(
                run(&mut scene, query).energy,
                0.0,
                "pressure bypassed edge {index}"
            );
        }
        // An opaque wall across the physical middle segment must remove the IR.
        let mut blocked = description.clone();
        blocked.meshes[0]
            .vertices
            .extend(geometry::quad_x(0.0, -10.0, 10.0, -10.0, 10.0));
        scene.load(blocked).unwrap();
        assert_eq!(
            run(&mut scene, query).energy,
            0.0,
            "pressure bypassed the middle blocker"
        );

        // Exercise the same public engine and stereo snapshots used by the
        // playground, with both diffraction events required for any pressure.
        use sound_engine::acoustics::{AcousticModel, AcousticsEngine};
        use sound_engine::{ListenerPose, PointSource};
        let stereo_config = EngineConfig {
            sound: SoundConfig {
                output_channels: OutputChannels::Stereo,
                ..config.sound
            },
            ..config
        };
        let mut engine = AcousticsEngine::with_model(
            &gpu,
            stereo_config,
            AcousticModel::Utd {
                cone_width: 0.02,
                maximum_frequency_hz: 20000.0,
                diffraction_orders: 2,
                band_count: 16,
                surface_two_edge_paths: false,
            },
        )
        .unwrap();
        engine.load_scene(description).unwrap();
        let source = PointSource::new(query.source_position);
        let listener = ListenerPose::new(query.listener_position, -Vec3::Z, Vec3::Y);
        let stereo = engine
            .build_impulse_response(&gpu, source, listener)
            .unwrap();
        assert!(stereo.energy > 1e-8);
        let rotated = engine
            .build_impulse_response(
                &gpu,
                source,
                ListenerPose::new(query.listener_position, Vec3::Z, Vec3::Y),
            )
            .unwrap();
        let swap_error: f32 = stereo
            .samples
            .iter()
            .zip(&rotated.samples)
            .map(|(a, b)| (a.left - b.right).powi(2) + (a.right - b.left).powi(2))
            .sum();
        assert!(
            swap_error / stereo.energy < 1e-6,
            "two-edge ear directions did not rotate"
        );
        let moved = engine
            .build_impulse_response(
                &gpu,
                source,
                ListenerPose::new(query.listener_position + Vec3::X, -Vec3::Z, Vec3::Y),
            )
            .unwrap();
        assert!(moved.energy > 1e-8);
        let change: f32 = stereo
            .samples
            .iter()
            .zip(&moved.samples)
            .map(|(a, b)| (a.left - b.left).powi(2) + (a.right - b.right).powi(2))
            .sum();
        assert!(
            change / stereo.energy > 0.1,
            "two-edge IR failed to update after movement"
        );
    }

    #[test]
    #[ignore = "requires Vulkan ray tracing and glslangValidator"]
    fn utd_engine_stereo_tracks_listener_orientation_and_position() {
        let _guard = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        use sound_engine::acoustics::{AcousticModel, AcousticsEngine};
        use sound_engine::{ListenerPose, PointSource};
        let gpu = VkBackend::new().unwrap();
        let config = EngineConfig {
            sound: SoundConfig {
                output_channels: OutputChannels::Stereo,
                sample_rate: 44100,
                ir_num_samples: 4096,
            },
            acoustics: AcousticsConfig {
                listener_ray_count: 262144,
                max_bounces: 1,
                maximum_path_length: 20.0,
                minimum_throughput: 0.0,
                sampling_seed: 7,
            },
            cache: IrCacheConfig::default(),
            binaural: BinauralConfig::default(),
            auralization: AuralizationConfig {
                block_size: 1024,
            },
        };
        let mut engine = AcousticsEngine::with_model(
            &gpu,
            config,
            AcousticModel::Utd {
                cone_width: 0.04,
                maximum_frequency_hz: 20000.0,
                diffraction_orders: 2,
                band_count: 16,
                surface_two_edge_paths: false,
            },
        )
        .unwrap();
        let source = PointSource::new(vec3(-3.0, 1.0, 0.0));
        let listener = ListenerPose::new(vec3(3.0, 1.0, 0.0), -Vec3::Z, Vec3::Y);
        let left = engine
            .build_impulse_response(&gpu, source, listener)
            .unwrap();
        assert_eq!(left.samples.len(), 4096);
        assert!(left.samples.iter().any(|s| s.left.abs() > 0.01));
        assert!(left.samples.iter().map(|s| s.right * s.right).sum::<f32>() / left.energy < 1e-6);
        let rotated = ListenerPose::new(listener.position, Vec3::Z, Vec3::Y);
        let right = engine
            .build_impulse_response(&gpu, source, rotated)
            .unwrap();
        for (a, b) in left.samples.iter().zip(&right.samples) {
            assert!((a.left - b.right).abs() < 1e-6 && (a.right - b.left).abs() < 1e-6);
        }
        let moved = engine
            .build_impulse_response(
                &gpu,
                source,
                ListenerPose::new(vec3(2.0, 1.0, 0.0), -Vec3::Z, Vec3::Y),
            )
            .unwrap();
        let peak = |ir: &sound_engine::acoustics::IrSnapshot| {
            ir.samples
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.left.abs().total_cmp(&b.left.abs()))
                .unwrap()
                .0
        };
        assert!((peak(&left) as f32 - 6.0 / 343.0 * 44100.0).abs() < 1.0);
        assert!((peak(&moved) as f32 - 5.0 / 343.0 * 44100.0).abs() < 1.0);
        let mut scene =
            geometry::scene_from_walls(vec![geometry::quad_x(0.0, -20.0, 2.0, -20.0, 20.0)], 1.0);
        scene.diffraction_edges.push(edge(
            1,
            vec3(0.0, 2.0, -20.0),
            vec3(0.0, 2.0, 20.0),
            Vec3::Y,
            std::f32::consts::TAU,
            0.2,
        ));
        engine.load_scene(scene.clone()).unwrap();
        let diffracted = engine
            .build_impulse_response(&gpu, source, listener)
            .unwrap();
        assert!(
            diffracted
                .samples
                .iter()
                .all(|s| s.left.is_finite() && s.right.is_finite())
        );
        assert!(diffracted.energy > 1e-6);
        let swapped = engine
            .build_impulse_response(&gpu, source, rotated)
            .unwrap();
        let residual: f32 = diffracted
            .samples
            .iter()
            .zip(&swapped.samples)
            .map(|(a, b)| (a.left - b.right).powi(2) + (a.right - b.left).powi(2))
            .sum();
        assert!(residual / diffracted.energy < 1e-6);
        scene.diffraction_edges.clear();
        engine.load_scene(scene).unwrap();
        let blocked = engine
            .build_impulse_response(&gpu, source, listener)
            .unwrap();
        assert_eq!(
            blocked.energy, 0.0,
            "blocked pressure bypassed ray visibility"
        );
    }

    #[test]
    #[ignore = "requires Vulkan ray tracing and glslangValidator"]
    fn utd_pressure_obeys_visibility_source_gain_and_direct_phase() {
        let _guard = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let gpu = VkBackend::new().unwrap();
        let config = EngineConfig {
            sound: SoundConfig {
                output_channels: OutputChannels::Mono,
                sample_rate: 44100,
                ir_num_samples: 4096,
            },
            acoustics: AcousticsConfig {
                listener_ray_count: 1_048_576,
                max_bounces: 1,
                maximum_path_length: 20.0,
                minimum_throughput: 0.0,
                sampling_seed: 7,
            },
            cache: IrCacheConfig::default(),
            binaural: BinauralConfig::default(),
            auralization: AuralizationConfig {
                block_size: 1024,
            },
        };
        let mut pipeline = ExperimentalUtdPipeline::new(&gpu, config, 0.04).unwrap();
        let mut query = AcousticQuery {
            query_id: 0,
            source_position: vec3(-3.0, 1.0, 0.0),
            listener_position: vec3(3.0, 1.0, 0.0),
            listener_forward: -Vec3::X,
            listener_up: Vec3::Y,
            source_amplitude: 1.0,
            source_radiation: Default::default(),
        };
        let mut empty = SceneManager::new();
        empty
            .load(SceneDescription {
                meshes: Vec::new(),
                materials: Vec::new(),
                objects: Vec::new(),
                diffraction_edges: Vec::new(),
            })
            .unwrap();
        let direct = pipeline.build_transfer(&gpu, &mut empty, query).unwrap();
        let ir = pipeline.synthesize_ir(&direct).unwrap();
        assert_eq!(ir.len(), 4096);
        let forward = realfft::RealFftPlanner::<f32>::new().plan_fft_forward(ir.len());
        let mut restored = forward.make_output_vec();
        forward.process(&mut ir.clone(), &mut restored).unwrap();
        for (actual, expected) in restored.iter().zip(&direct) {
            assert!((actual.re - expected[0]).abs() < 1e-6);
            assert!((actual.im - expected[1]).abs() < 1e-6);
        }
        assert!(pipeline.synthesize_ir(&direct[..10]).is_err());
        for (bin, value) in direct.iter().enumerate().skip(1) {
            let frequency = bin as f32 * 44100.0 / 4096.0;
            if frequency > 2000.0 {
                assert_eq!(*value, [0.0, 0.0]);
                continue;
            }
            let phase = -std::f32::consts::TAU * frequency * 6.0 / 343.0;
            assert!((value[0] - phase.cos() / 6.0).abs() < 2e-5);
            assert!((value[1] - phase.sin() / 6.0).abs() < 2e-5);
        }
        let mut reference = Vec::new();
        for variant in 0..5 {
            let mut description = geometry::scene_from_walls(
                vec![geometry::quad_x(0.0, -20.0, 2.0, -20.0, 20.0)],
                1.0,
            );
            description.diffraction_edges.push(edge(
                1,
                vec3(0.0, 2.0, -20.0),
                vec3(0.0, 2.0, 20.0),
                Vec3::Y,
                std::f32::consts::TAU,
                0.2,
            ));
            if variant == 2 {
                description.diffraction_edges[0].base_strength = 0.0;
            }
            if variant == 3 {
                description.meshes[0]
                    .vertices
                    .extend(geometry::quad_x(1.5, -20.0, 20.0, -20.0, 20.0));
            }
            query.source_amplitude = if variant == 1 {
                0.5
            } else if variant == 4 {
                0.0
            } else {
                1.0
            };
            let mut scene = SceneManager::new();
            scene.load(description).unwrap();
            let data = pipeline.build_transfer(&gpu, &mut scene, query).unwrap();
            assert!(data.iter().flatten().all(|v| v.is_finite()));
            if variant == 0 {
                assert!(data.iter().flatten().any(|v| v.abs() > 1e-4));
                // Independent stationary single-wedge values, calculated with
                // complex erfc (Tsingos Appendix A, Eqs. 4–8). This checks the
                // Monte Carlo strip/cone normalization, not only source scaling.
                for (bin, expected) in [
                    (23, [-0.023324789, 0.038156537]),
                    (93, [-0.021261286, 0.011303850]),
                    (163, [-0.018164788, -0.002462949]),
                ] {
                    let actual = data[bin];
                    let residual = ((actual[0] - expected[0]).powi(2)
                        + (actual[1] - expected[1]).powi(2))
                    .sqrt();
                    let norm = (expected[0].powi(2) + expected[1].powi(2)).sqrt();
                    assert!(
                        residual / norm < 0.05,
                        "UTD normalization bin {bin}: {actual:?}, expected {expected:?}"
                    );
                }
                let mut previous_error = f64::INFINITY;
                for bands in [4, 16, 32] {
                    let mut band_pipeline =
                        ExperimentalUtdPipeline::with_bands(&gpu, config, 0.04, 2000.0, bands)
                            .unwrap();
                    let snapshot = band_pipeline.build_ir_gpu(&gpu, &mut scene, query).unwrap();
                    let reconstructed = band_pipeline.download_gpu_ir(&gpu, &snapshot).unwrap();
                    let mut wave: Vec<_> = reconstructed.samples.iter().map(|s| s.left).collect();
                    let mut response = forward.make_output_vec();
                    forward.process(&mut wave, &mut response).unwrap();
                    let residual: f64 = response
                        .iter()
                        .zip(&data)
                        .map(|(a, b)| {
                            (a.re as f64 - b[0] as f64).powi(2)
                                + (a.im as f64 - b[1] as f64).powi(2)
                        })
                        .sum();
                    let norm: f64 = data.iter().flatten().map(|v| (*v as f64).powi(2)).sum();
                    let error = (residual / norm).sqrt();
                    eprintln!("UTD {bands}-band complex reconstruction relative L2: {error}");
                    assert!(
                        error <= previous_error + 0.001,
                        "band interpolation did not converge"
                    );
                    if bands >= 16 {
                        assert!(error < 0.02, "band synthesis changed ray pressure: {error}");
                    }
                    previous_error = error;
                }
                reference = data;
            } else if variant == 1 {
                let error: f64 = data
                    .iter()
                    .flatten()
                    .zip(reference.iter().flatten())
                    .map(|(&a, &b)| (a as f64 - 0.5 * b as f64).powi(2))
                    .sum();
                let norm: f64 = reference
                    .iter()
                    .flatten()
                    .map(|v| (*v as f64).powi(2))
                    .sum();
                assert!((error / norm).sqrt() < 1e-4);
            } else {
                assert!(
                    data.iter().flatten().all(|&v| v == 0.0),
                    "variant {variant} bypassed ray pressure or visibility"
                );
            }
        }
    }
}
