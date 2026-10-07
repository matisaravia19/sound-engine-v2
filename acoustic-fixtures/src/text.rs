//! Readers for the scene and position files produced by the measurement scripts.

use glam::{Mat4, Vec3};
use sound_engine::scene::{
    DiffractionEdge, Material, MeshAsset, SceneDescription, SceneManager, SceneObject,
};
use std::collections::{BTreeMap, HashSet};
use std::io;
use std::path::Path;

/// Source and receiver coordinates in engine axes, copied from a measured pair.
#[derive(Debug, PartialEq)]
pub struct PositionPair {
    /// Unique case identifier, also safe to use as an exported filename stem.
    pub name: String,
    /// Source position in meters.
    pub source: Vec3,
    /// Receiver position in meters.
    pub listener: Vec3,
}

/// Reads and validates a text scene containing materials, triangles, and edges.
/// Errors include the input path and, for malformed records, their line number.
pub fn read_scene(path: impl AsRef<Path>) -> io::Result<SceneDescription> {
    read_file(path.as_ref(), parse_scene)
}

/// Reads a nonempty table of unique, finite source/receiver positions.
/// Accepts the column headers emitted by both BRAS preparation scripts.
pub fn read_positions(path: impl AsRef<Path>) -> io::Result<Vec<PositionPair>> {
    read_file(path.as_ref(), parse_positions)
}

fn read_file<T>(path: &Path, parse: impl FnOnce(&str) -> Result<T, String>) -> io::Result<T> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| io::Error::new(error.kind(), format!("{}: {error}", path.display())))?;
    parse(&text).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{}: {error}", path.display()),
        )
    })
}

fn parse_scene(text: &str) -> Result<SceneDescription, String> {
    let mut scene = SceneDescription::default();
    let mut vertices = BTreeMap::<u32, Vec<Vec3>>::new();
    for (index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields = line.split_whitespace().collect::<Vec<_>>();
        parse_scene_row(&fields, &mut scene, &mut vertices)
            .map_err(|error| format!("line {}: {error}", index + 1))?;
    }

    for (id, vertices) in vertices {
        scene.meshes.push(MeshAsset {
            id,
            vertices,
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
    // Share the engine's geometry/material invariants without creating GPU state.
    SceneManager::new()
        .load(scene.clone())
        .map_err(|error| error.to_string())?;
    Ok(scene)
}

fn parse_scene_row(
    fields: &[&str],
    scene: &mut SceneDescription,
    vertices: &mut BTreeMap<u32, Vec<Vec3>>,
) -> Result<(), String> {
    match fields {
        ["material", id, absorption, scattering] => {
            let [absorption, scattering] = numbers(&[absorption, scattering])?;
            scene.materials.push(Material {
                id: material_id(id)?,
                absorption,
                scattering,
                roughness: 0.0,
            });
        }
        ["tri", id, coordinates @ ..] => {
            let coordinates = numbers::<9>(coordinates)?;
            vertices.entry(material_id(id)?).or_default().extend(
                coordinates
                    .chunks_exact(3)
                    .map(|point| Vec3::new(point[0], point[1], point[2])),
            );
        }
        ["edge", values @ ..] if values.len() == 11 || values.len() == 12 => {
            let values_without_scale = numbers::<11>(&values[..11])?;
            // Cached fixtures may contain the retired energy-scale column.
            // Zero disabled an edge; positive values no longer change its response.
            if values.len() == 12 {
                let [scale] = numbers(&values[11..])?;
                if !(0.0..=1.0).contains(&scale) {
                    return Err("legacy edge scale must be in [0, 1]".into());
                }
                if scale == 0.0 {
                    return Ok(());
                }
            }
            let [sx, sy, sz, ex, ey, ez, bx, by, bz, angle, radius] = values_without_scale;
            let id = u32::try_from(scene.diffraction_edges.len())
                .ok()
                .and_then(|count| count.checked_add(1))
                .ok_or("too many diffraction edges")?;
            scene.diffraction_edges.push(DiffractionEdge {
                id,
                start: Vec3::new(sx, sy, sz),
                end: Vec3::new(ex, ey, ez),
                bisector_dir: Vec3::new(bx, by, bz),
                edge_angle_radians: angle,
                diffraction_radius: radius,
            });
        }
        _ => return Err("expected a material, triangle, or edge record".into()),
    }
    Ok(())
}

fn material_id(value: &str) -> Result<u32, String> {
    value
        .parse::<u32>()
        .ok()
        .filter(|id| *id != 0)
        .ok_or_else(|| format!("invalid material identifier: {value}"))
}

fn numbers<const N: usize>(fields: &[&str]) -> Result<[f32; N], String> {
    if fields.len() != N {
        return Err(format!("expected {N} numeric fields, got {}", fields.len()));
    }
    let mut values = [0.0_f32; N];
    for (value, field) in values.iter_mut().zip(fields) {
        *value = field
            .parse()
            .map_err(|_| format!("invalid number: {field}"))?;
        if !value.is_finite() {
            return Err(format!("non-finite number: {field}"));
        }
    }
    Ok(values)
}

fn parse_positions(text: &str) -> Result<Vec<PositionPair>, String> {
    let mut lines = text.lines();
    let header = lines.next().ok_or("missing position header")?;
    let fields = header.split(',').map(str::trim).collect::<Vec<_>>();
    if fields != ["case", "sx", "sy", "sz", "rx", "ry", "rz"]
        && fields
            != [
                "case",
                "source_x",
                "source_y",
                "source_z",
                "listener_x",
                "listener_y",
                "listener_z",
            ]
    {
        return Err("unexpected position columns".into());
    }

    let mut pairs = Vec::new();
    let mut names = HashSet::new();
    for (index, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let pair =
            parse_position_row(line).map_err(|error| format!("line {}: {error}", index + 2))?;
        if !names.insert(pair.name.clone()) {
            return Err(format!("line {}: duplicate case {}", index + 2, pair.name));
        }
        pairs.push(pair);
    }
    if pairs.is_empty() {
        return Err("no positions".into());
    }
    Ok(pairs)
}

fn parse_position_row(line: &str) -> Result<PositionPair, String> {
    let fields = line.split(',').map(str::trim).collect::<Vec<_>>();
    let name = fields[0];
    if name.is_empty()
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err("case names must contain only letters, digits, underscores, or hyphens".into());
    }
    let [sx, sy, sz, lx, ly, lz] = numbers(&fields[1..])?;
    Ok(PositionPair {
        name: name.to_owned(),
        source: Vec3::new(sx, sy, sz),
        listener: Vec3::new(lx, ly, lz),
    })
}

#[cfg(test)]
#[path = "../tests/unit/text.rs"]
mod tests;
