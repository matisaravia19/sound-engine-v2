//! Interactive scene catalog. Measurement files remain optional runtime assets.

mod bell_box;
mod bras;
mod geometry;

use glam::{Mat4, Vec3, vec3};
use sound_engine::scene::{
    DiffractionEdge, Material, MeshAsset, SceneDescription, SceneManager, SceneObject,
};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

/// Source and receiver coordinates in engine axes, copied from a benchmark pair.
pub(crate) struct Pair {
    /// Benchmark case identifier displayed in the sandbox.
    pub name: String,
    /// Source position in meters.
    pub source: Vec3,
    /// Receiver position in meters.
    pub listener: Vec3,
}

/// Catalog entry; geometry is read only when the user selects it.
pub(crate) struct Entry {
    /// Stable command-line identifier.
    pub id: String,
    /// Short display name.
    pub title: String,
    /// Optional input scene and paired-position files.
    files: Option<(PathBuf, PathBuf)>,
    /// Optional measured RS5/RS6 position table.
    positions: Option<PathBuf>,
}

/// Owned scene and positions, validated before replacing the active scene.
pub(crate) struct Loaded {
    /// Acoustic geometry and original authored materials.
    pub description: SceneDescription,
    /// Nonempty paired positions for interactive exploration.
    pub pairs: Vec<Pair>,
    /// Transformed triangles cached for rendering; no measurement audio is loaded.
    pub triangles: Vec<[Vec3; 3]>,
}

/// Lists built-in controls and available BRAS fixtures beneath the output root.
pub(crate) fn catalog(root: &Path) -> Vec<Entry> {
    let names = [
        ("playground", "Habitacion con puerta"),
        ("free-field", "Campo libre"),
        ("single-plane", "Una pared rigida"),
        ("absorbing-plane", "Una pared absorbente"),
        ("parallel-planes", "Dos paredes paralelas"),
        ("shoebox", "Caja rectangular"),
        ("wedge", "Difraccion: una esquina"),
        ("bell-empty", "Bell Labs: caja vacia"),
        ("bell-baffle", "Bell Labs: panel con difraccion"),
        ("bell-no-diffraction", "Bell Labs: panel sin difraccion"),
    ];
    let mut entries: Vec<_> = names
        .into_iter()
        .map(|(id, title)| Entry {
            id: id.into(),
            title: title.into(),
            files: None,
            positions: None,
        })
        .collect();

    for (id, title) in [("RS5", "BRAS RS5: tabique"), ("RS6", "BRAS RS6: bloque")] {
        let positions = root
            .join("measured-reference-research")
            .join(id)
            .join("positions.csv");
        entries.push(Entry {
            id: id.into(),
            title: title.into(),
            files: None,
            positions: Some(positions),
        });
    }

    let folder = root.join("bras-extended-20260909/fixtures");

    if let Ok(files) = std::fs::read_dir(folder) {
        let mut paths: Vec<_> = files
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "scene"))
            .collect();
        paths.sort();

        for scene in paths {
            let Some(id) = scene
                .file_stem()
                .and_then(|s| s.to_str())
                .map(str::to_owned)
            else {
                continue;
            };

            let positions = scene.with_extension("csv");

            if positions.is_file() {
                entries.push(Entry {
                    title: fixture_title(&id),
                    id,
                    files: Some((scene, positions)),
                    positions: None,
                });
            }
        }
    }

    entries
}

fn fixture_title(id: &str) -> String {
    let description = match id {
        "RS1_Absorbing" => "RS1: suelo absorbente",
        "RS1_Diffuse" => "RS1: suelo con perfiles",
        "RS1_Rigid" => "RS1: suelo rigido",
        "RS2_1mPlate_absorbing" => "RS2: placa absorbente de 1 m",
        "RS2_1mPlate_rigid" => "RS2: placa rigida de 1 m",
        "RS2_2mPlate_absorbing" => "RS2: placa absorbente de 2 m",
        "RS2_2mPlate_rigid" => "RS2: placa rigida de 2 m",
        "RS3_default" => "RS3: dos placas enfrentadas",
        "RS4_offCenter" => "RS4: paneles, fuera del centro",
        "RS4_onCenter" => "RS4: paneles, centrado",
        "RS7_default" => "RS7: fila de obstaculos",
        "CR1_DoorAngle1" => "CR1: recintos acoplados, puerta 1",
        "CR1_DoorAngle3" => "CR1: recintos acoplados, puerta 3",
        "CR2_room" => "CR2: sala de seminarios",
        "CR3_room" => "CR3: sala de musica de camara",
        "CR4_room" => "CR4: auditorio",
        _ => id,
    };
    format!("BRAS {description}")
}

impl Entry {
    /// Loads geometry and pair coordinates without changing the running engine.
    pub(crate) fn load(&self) -> Result<Loaded, String> {
        let pair = |name: &str, source, listener| {
            vec![Pair {
                name: name.into(),
                source,
                listener,
            }]
        };
        let (description, pairs) = if let Some((scene, positions)) = &self.files {
            (read_fixture(scene)?, read_pairs(positions)?)
        } else {
            match self.id.as_str() {
                "free-field" => (
                    geometry::empty_scene(),
                    [1.0, 2.0, 5.0]
                        .into_iter()
                        .map(|distance| Pair {
                            name: format!("Campo libre {distance} m"),
                            source: Vec3::ZERO,
                            listener: if distance == 1.0 {
                                vec3(1.003, 0.217, -0.091)
                            } else {
                                vec3(1.003, 0.217, -0.091).normalize() * distance
                            },
                        })
                        .collect(),
                ),
                "single-plane" | "absorbing-plane" => (
                    geometry::single_plane_scene(if self.id == "absorbing-plane" {
                        0.36
                    } else {
                        0.0
                    }),
                    pair("Pared", vec3(1.1, -0.3, 0.2), vec3(2.7, 0.8, -0.4)),
                ),
                "parallel-planes" => (
                    geometry::parallel_plane_scene(10.0, 0.36),
                    pair("Planos", vec3(2.1, -0.4, 0.3), vec3(7.3, 0.6, -0.2)),
                ),
                "shoebox" => (
                    geometry::shoebox_scene(vec3(10.0, 8.0, 6.0), 0.36),
                    pair("Caja", vec3(2.1, 2.7, 2.2), vec3(7.4, 5.9, 3.8)),
                ),
                "wedge" => (
                    geometry::diffraction_wedge_scene(1.5 * std::f32::consts::PI, 20.0, 0.45),
                    pair("Sombra", vec3(-2.0, 0.0, 3.0), vec3(-2.0, 0.0, -3.0)),
                ),
                "bell-empty" => (
                    bell_box::scene(false, false, 0.1, 0.05),
                    pair(
                        "Caja vacia",
                        bell_box::EMPTY_SOURCE,
                        bell_box::EMPTY_LISTENER,
                    ),
                ),
                "bell-baffle" | "bell-no-diffraction" => (
                    bell_box::scene(true, self.id == "bell-baffle", 0.1, 0.05),
                    pair("Panel", bell_box::BAFFLE_SOURCE, bell_box::BAFFLE_LISTENER),
                ),
                "RS5" | "RS6" => (
                    bras::benchmark_scene(&self.id, 0.2, true),
                    read_pairs(self.positions.as_ref().ok_or("Missing positions path")?)?,
                ),
                _ => return Err(format!("Unknown benchmark scene: {}", self.id)),
            }
        };
        // Validate material references and geometry before deriving render indices.
        SceneManager::new()
            .load(description.clone())
            .map_err(|e| e.to_string())?;
        let mut triangles = Vec::new();
        for object in description.objects.iter().filter(|o| o.active) {
            let mesh = description
                .meshes
                .iter()
                .find(|m| m.id == object.mesh_id)
                .ok_or("Missing mesh")?;
            if mesh.indices.is_empty() {
                for t in mesh.vertices.chunks_exact(3) {
                    triangles.push([0, 1, 2].map(|i| object.transform.transform_point3(t[i])));
                }
            } else {
                for t in mesh.indices.chunks_exact(3) {
                    triangles.push([0, 1, 2].map(|i| {
                        object
                            .transform
                            .transform_point3(mesh.vertices[t[i] as usize])
                    }));
                }
            }
        }
        Ok(Loaded {
            description,
            pairs,
            triangles,
        })
    }
}

fn read_pairs(path: &Path) -> Result<Vec<Pair>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut pairs = Vec::new();
    for line in text.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<_> = line.split(',').collect();
        if f.len() != 7 {
            return Err(format!("Invalid position row in {}", path.display()));
        }
        let p: Vec<f32> = f[1..]
            .iter()
            .map(|x| x.parse::<f32>())
            .collect::<Result<_, _>>()
            .map_err(|e| e.to_string())?;
        if !p.iter().all(|x| x.is_finite()) {
            return Err("Non-finite position".into());
        }
        pairs.push(Pair {
            name: f[0].to_owned(),
            source: vec3(p[0], p[1], p[2]),
            listener: vec3(p[3], p[4], p[5]),
        });
    }
    if pairs.is_empty() {
        return Err(format!("No positions in {}", path.display()));
    }
    Ok(pairs)
}

fn read_fixture(path: &Path) -> Result<SceneDescription, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut scene = geometry::empty_scene();
    let mut vertices: BTreeMap<u32, Vec<Vec3>> = BTreeMap::new();
    for line in text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
    {
        let f: Vec<_> = line.split_whitespace().collect();
        let p: Vec<f32> = f[1..]
            .iter()
            .map(|x| x.parse::<f32>())
            .collect::<Result<_, _>>()
            .map_err(|e| e.to_string())?;
        if !p.iter().all(|v| v.is_finite()) {
            return Err("Non-finite scene value".into());
        }
        match f[0] {
            "material" if p.len() == 3 => {
                scene.materials.push(Material {
                    id: material_id(p[0])?,
                    absorption_bands: vec![p[1]],
                    scattering: p[2],
                    roughness: 0.0,
                    transmission: None,
                });
            }
            "tri" if p.len() == 10 => {
                vertices
                    .entry(material_id(p[0])?)
                    .or_default()
                    .extend(p[1..].chunks_exact(3).map(|v| vec3(v[0], v[1], v[2])));
            }
            "edge" if p.len() == 12 => {
                scene.diffraction_edges.push(DiffractionEdge {
                    id: scene.diffraction_edges.len() as u32 + 1,
                    start: vec3(p[0], p[1], p[2]),
                    end: vec3(p[3], p[4], p[5]),
                    bisector_dir: vec3(p[6], p[7], p[8]),
                    edge_angle_radians: p[9],
                    diffraction_radius: p[10],
                    base_strength: p[11],
                });
            }
            _ => return Err(format!("Invalid scene row: {line}")),
        }
    }
    for (id, v) in vertices {
        scene.meshes.push(MeshAsset {
            id,
            vertices: v,
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

fn material_id(value: f32) -> Result<u32, String> {
    if value >= 1.0 && value < u32::MAX as f32 && value.fract() == 0.0 {
        Ok(value as u32)
    } else {
        Err("Invalid material identifier".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn builtins_have_valid_geometry_and_pairs() {
        for entry in catalog(Path::new("missing-runtime-assets"))
            .into_iter()
            .filter(|e| e.id != "playground" && e.positions.is_none())
        {
            let scene = entry.load().unwrap();
            assert!(!scene.pairs.is_empty());
            if entry.id == "bell-baffle" {
                assert_eq!(scene.description.diffraction_edges.len(), 1);
            }
            if entry.id == "bell-no-diffraction" {
                assert!(scene.description.diffraction_edges.is_empty());
            }
        }
    }
}
