//! Hardware regression for the proxy measure on consecutive edge interactions.

use super::{AcousticPipeline, AcousticQuery};
use crate::core::config::*;
use crate::core::error::SoundResult;
use crate::gpu::backend::VkBackend;
use crate::scene::{
    DiffractionEdge, Material, MeshAsset, SceneDescription, SceneManager, SceneObject,
};
use glam::{Mat4, Vec3, vec3};

fn alternating_screens(scale: f32, first_radius: f32, last_radius: f32) -> SceneDescription {
    let mut vertices = Vec::new();
    let mut edges = Vec::new();

    for index in 0..2 {
        let z = 3.0 + 4.0 * index as f32;
        let (left, right, edge_x, bisector) = if index == 0 {
            (-0.5, 40.0, -0.5, -Vec3::X)
        } else {
            (-40.0, 0.5, 0.5, Vec3::X)
        };
        let a = vec3(left, -40.0, z) * scale;
        let b = vec3(right, -40.0, z) * scale;
        let c = vec3(right, 40.0, z) * scale;
        let d = vec3(left, 40.0, z) * scale;
        vertices.extend([a, b, c, a, c, d]);

        edges.push(DiffractionEdge {
            id: index + 1,
            start: vec3(edge_x, -40.0, z) * scale,
            end: vec3(edge_x, 40.0, z) * scale,
            bisector_dir: bisector,
            edge_angle_radians: std::f32::consts::TAU,
            diffraction_radius: if index == 0 {
                first_radius
            } else {
                last_radius
            } * scale,
        });
    }

    // Absorbing screens block direct and reflected paths: a nonzero result
    // requires both diffraction events within the two-interaction budget.
    SceneDescription {
        meshes: vec![MeshAsset {
            id: 1,
            vertices,
            indices: Vec::new(),
            opaque: true,
        }],
        materials: vec![Material {
            id: 1,
            absorption_bands: vec![1.0],
            scattering: 0.0,
            roughness: 0.0,
            transmission: None,
        }],
        objects: vec![SceneObject {
            id: 1,
            mesh_id: 1,
            material_id: 1,
            transform: Mat4::IDENTITY,
            active: true,
        }],
        diffraction_edges: edges,
    }
}

#[test]
#[ignore = "requires Vulkan ray tracing; run hardware tests serially"]
fn consecutive_edges_preserve_proxy_measure_and_scene_scale() -> SoundResult<()> {
    let gpu = VkBackend::new()?;
    let cases = [
        (1.0, 3.0, 3.0),
        (2.0, 3.0, 3.0),
        (1.0, 0.3, 3.0),
        (1.0, 6.0, 3.0),
        (1.0, 3.0, 0.3),
        (1.0, 3.0, 6.0),
    ];
    let mut pressures = [0.0_f64; 6];

    for seed in [31, 32, 33, 34] {
        let config = EngineConfig {
            sound: SoundConfig {
                output_channels: OutputChannels::Mono,
                sample_rate: 48_000,
                ir_num_samples: 65_536,
            },
            acoustics: AcousticsConfig {
                listener_ray_count: 8_388_608,
                max_bounces: 2,
                maximum_path_length: 800.0,
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

        for (index, &(scale, first_radius, last_radius)) in cases.iter().enumerate() {
            let mut manager = SceneManager::new();
            manager.load(alternating_screens(scale, first_radius, last_radius))?;
            let query = AcousticQuery {
                query_id: index as u32 + 1,
                source_position: vec3(-0.5, 0.0, 10.0) * scale,
                listener_position: vec3(0.5, 0.0, 0.0) * scale,
                listener_forward: Vec3::Z,
                listener_up: Vec3::Y,
                source_amplitude: 1.0,
            };
            let ir = pipeline.build_ir(&gpu, &mut manager, query)?;
            assert!(ir.samples.iter().all(|sample| sample.left.is_finite()));
            pressures[index] += ir
                .samples
                .iter()
                .map(|sample| sample.left as f64)
                .sum::<f64>()
                / 4.0;
        }
    }

    // Independent line quadrature of D1*D2/(a*b*c), y1,y2 in [-40,40].
    // Increasing Gauss-Legendre order from 512 to 1024 changes this by <1e-12.
    // It checks the scalar model's measure, not agreement with measured physics.
    let reference = 0.044_341_657_879_09;
    for ((scale, first_radius, last_radius), pressure) in cases.into_iter().zip(pressures) {
        let relative_error = pressure * scale as f64 / reference - 1.0;
        println!(
            "scale={scale} proxies={first_radius}/{last_radius} pressure={pressure} error={relative_error}"
        );
        assert!(
            relative_error.abs() < 0.15,
            "edge-line measure error: {relative_error}"
        );
    }

    assert!((2.0 * pressures[1] / pressures[0] - 1.0).abs() < 0.005);
    Ok(())
}
