//! File-driven BRAS validation using the unchanged scalar ray transport.
//! Fixtures contain triangles, broadband materials and authored diffraction edges.
#[path = "support/geometry.rs"]
mod geometry;

use glam::{Mat4, Vec3, vec3};
use sound_engine::acoustics::{AcousticPipeline, AcousticQuery};
use sound_engine::core::config::*;
use sound_engine::gpu::backend::VkBackend;
use sound_engine::scene::{
    DiffractionEdge, Material, MeshAsset, SceneDescription, SceneManager, SceneObject,
};
use std::{
    collections::BTreeMap,
    fs,
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

fn fixture(path: &std::path::Path) -> Result<SceneDescription, Box<dyn std::error::Error>> {
    let mut scene = geometry::empty_scene();
    let mut vertices: BTreeMap<u32, Vec<Vec3>> = BTreeMap::new();
    for line in fs::read_to_string(path)?
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let fields: Vec<_> = line.split_whitespace().collect();
        let p: Vec<f32> = fields[1..]
            .iter()
            .map(|x| x.parse())
            .collect::<Result<_, _>>()?;
        if !p.iter().all(|v| v.is_finite()) {
            return Err("non-finite fixture value".into());
        }
        match fields[0] {
            "material"
                if p.len() == 3 && (0.0..=1.0).contains(&p[1]) && (0.0..=1.0).contains(&p[2]) =>
            {
                scene.materials.push(Material {
                    id: p[0] as u32,
                    absorption_bands: vec![p[1]],
                    scattering: p[2],
                    roughness: 0.0,
                    transmission: None,
                });
            }
            "tri" if p.len() == 10 => {
                vertices
                    .entry(p[0] as u32)
                    .or_default()
                    .extend(p[1..].chunks_exact(3).map(|v| vec3(v[0], v[1], v[2])));
            }
            "edge" if (p.len() == 11 || p.len() == 12) && p[10] > 0.0 => {
                // Legacy fixtures end in an energy scale. Only their disabled
                // state survives migration; positive values no longer scale rays.
                if p.len() == 12 {
                    if !(0.0..=1.0).contains(&p[11]) {
                        return Err(format!("invalid legacy edge scale: {line}").into());
                    }

                    if p[11] == 0.0 {
                        continue;
                    }
                }

                scene.diffraction_edges.push(DiffractionEdge {
                    id: scene.diffraction_edges.len() as u32 + 1,
                    start: vec3(p[0], p[1], p[2]),
                    end: vec3(p[3], p[4], p[5]),
                    bisector_dir: vec3(p[6], p[7], p[8]),
                    edge_angle_radians: p[9],
                    diffraction_radius: p[10],
                });
            }
            _ => return Err(format!("invalid fixture row: {line}").into()),
        }
    }
    for (id, points) in vertices {
        if !scene.materials.iter().any(|m| m.id == id) {
            return Err("triangle has no material".into());
        }
        scene.meshes.push(MeshAsset {
            id,
            vertices: points,
            indices: Vec::new(),
            opaque: true,
        });
        scene.objects.push(SceneObject {
            id,
            mesh_id: id,
            material_id: id,
            transform: Mat4::IDENTITY,
            active: true,
        });
    }
    Ok(scene)
}

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
    let positions = fs::read_to_string(options.get("--positions").ok_or("missing --positions")?)?;
    let output = PathBuf::from(options.get("--output").ok_or("missing --output")?);
    fs::create_dir_all(&output)?;
    let description = fixture(&input)?;
    let mut log = BufWriter::new(fs::File::create(output.join("runs.csv"))?);
    writeln!(
        log,
        "case,seed,rays,bounces,seconds,cold_ms,warm_median_ms,warm_max_ms,nonzero_samples,peak"
    )?;
    let gpu = VkBackend::new()?;
    for seed in 7..7 + seeds {
        let config = EngineConfig {
            sound: SoundConfig {
                output_channels: OutputChannels::Mono,
                sample_rate: 44_100,
                ir_num_samples: (duration * 44_100.0).ceil() as u32,
            },
            acoustics: AcousticsConfig {
                listener_ray_count: rays,
                max_bounces: bounces,
                maximum_path_length: 343.0 * duration,
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
        let mut scene = SceneManager::new();
        scene.load(description.clone())?;
        for (index, line) in positions.lines().skip(1).enumerate() {
            let f: Vec<_> = line.split(',').collect();
            if f.len() != 7 || !f[0].chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                return Err("invalid positions row".into());
            }
            if options.get("--case").is_some_and(|c| c != f[0]) {
                continue;
            }
            let p: Vec<f32> = f[1..].iter().map(|x| x.parse()).collect::<Result<_, _>>()?;
            let query = AcousticQuery {
                query_id: index as u32,
                source_position: vec3(p[0], p[1], p[2]),
                listener_position: vec3(p[3], p[4], p[5]),
                listener_forward: -Vec3::X,
                listener_up: Vec3::Y,
                source_amplitude: 1.0,
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
                output.join(format!("{}-seed{seed}.f32", f[0])),
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
                f[0],
                times[times.len() / 2],
                times.last().unwrap()
            )?;
            log.flush()?;
        }
        eprintln!("completed seed {seed}");
    }
    Ok(())
}
