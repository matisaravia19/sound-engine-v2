//! Measured BRAS RS5/RS6 geometry: every exported propagation IR comes from rays.
//! Preparation and calibration: scripts/bras_reference.py; citations: references.md.
#[path = "support/geometry.rs"]
mod geometry;
#[path = "support/provenance.rs"]
mod provenance;
#[path = "support/timing.rs"]
mod timing;

use glam::{Mat4, Vec3, vec3};
use sound_engine::acoustics::{AcousticPipeline, AcousticQuery};
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
    let positions = fs::read_to_string(input.join(format!("{scene_name}/positions.csv")))?;
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
        let description = benchmark_scene(&scene_name, radius, diffraction);
        scene.load(description)?;
        let config = EngineConfig {
            sound: SoundConfig {
                output_channels: OutputChannels::Mono,
                sample_rate: 44_100,
                ir_num_samples: 11_025,
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
        let mut pipeline = AcousticPipeline::new(&gpu, config)?;

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
            };
            // Reciprocity diagnostics swap endpoints without changing geometry,
            // source gain, or calibration; the runs CSV records this explicitly.
            if reverse {
                std::mem::swap(&mut query.source_position, &mut query.listener_position);
            }
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
                last_ir = Some(pipeline.build_ir(&gpu, &mut scene, query)?);
            }

            let mut times: Vec<_> = timings.iter().map(|t| t.2).collect();
            times.sort_by(f64::total_cmp);
            export_ir(
                last_ir.as_ref().unwrap(),
                output.join(format!("{}-seed{seed}.csv", fields[0])),
            )?;

            writeln!(
                runs,
                "{},{seed},{rays},{bounces},{radius},{diffraction},44100,{cold_ms},{},{},{},{},{repeats},{warmups},{revision},{reverse}",
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
        absorption: 0.0,
        scattering: 0.0,
        roughness: 0.0,
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
    }
}
