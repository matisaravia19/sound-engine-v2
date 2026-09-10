//! Independent image-plane regression for physical edge-prefix relocation.
use super::{AcousticPipeline, AcousticQuery, IrSnapshot};
use crate::core::config::*;
use crate::gpu::backend::VkBackend;
use crate::scene::{DiffractionEdge, Material, MeshAsset, SceneDescription, SceneManager, SceneObject};
use glam::{Mat4, Vec3, vec3};

fn fixture(radius: f32, floor: bool, roughness: f32) -> SceneDescription {
    let mut scene = SceneDescription {
        meshes: vec![],
        materials: vec![],
        objects: vec![],
        diffraction_edges: vec![DiffractionEdge {
            id: 1,
            start: vec3(0.0, 2.0, -4.0),
            end: vec3(0.0, 2.0, 4.0),
            bisector_dir: Vec3::Y,
            edge_angle_radians: std::f32::consts::TAU,
            diffraction_radius: radius,
            base_strength: 1.0,
        }],
    };
    let screen = [
        vec3(0.0, -20.0, -20.0),
        vec3(0.0, 2.0, -20.0),
        vec3(0.0, 2.0, 20.0),
        vec3(0.0, -20.0, 20.0),
    ];
    let mirror = [
        vec3(-20.0, 0.0, -20.0),
        vec3(-0.001, 0.0, -20.0),
        vec3(-0.001, 0.0, 20.0),
        vec3(-20.0, 0.0, 20.0),
    ];
    for (id, corners, absorption) in [(1, screen, 1.0), (2, mirror, 0.36)] {
        if id == 2 && !floor {
            continue;
        }
        scene.meshes.push(MeshAsset {
            id,
            vertices: vec![corners[0], corners[1], corners[2], corners[0], corners[2], corners[3]],
            indices: vec![],
            opaque: true,
        });
        scene.materials.push(Material {
            id,
            absorption_bands: vec![absorption],
            scattering: 0.0,
            roughness,
            transmission: None,
        });
        scene.objects.push(SceneObject {
            id,
            mesh_id: id,
            material_id: id,
            transform: Mat4::IDENTITY,
            active: true,
        });
    }
    scene
}

fn trace(gpu: &VkBackend, radius: f32, floor: bool, roughness: f32, listener: Vec3, bounces: u32) -> IrSnapshot {
    let settings = EngineConfig {
        sound: SoundConfig {
            sample_rate: 48000,
            ir_num_samples: 4096,
            output_channels: OutputChannels::Mono,
        },
        acoustics: AcousticsConfig {
            listener_ray_count: 524288,
            max_bounces: bounces,
            maximum_path_length: 20.0,
            minimum_throughput: 0.0,
            sampling_seed: 17,
        },
        cache: IrCacheConfig::default(),
        binaural: BinauralConfig::default(),
        auralization: AuralizationConfig { block_size: 1024 },
    };
    let mut scene = SceneManager::new();
    scene.load(fixture(radius, floor, roughness)).unwrap();
    let mut pipeline = AcousticPipeline::new(gpu, settings).unwrap();
    pipeline
        .build_ir(
            gpu,
            &mut scene,
            AcousticQuery {
                query_id: 1,
                source_position: vec3(3.0, 1.0, 0.0),
                listener_position: listener,
                listener_forward: Vec3::X,
                listener_up: Vec3::Y,
                source_amplitude: 1.0,
                source_radiation: Default::default(),
            },
        )
        .unwrap()
}

#[test]
#[ignore = "requires Vulkan ray tracing; run hardware tests serially"]
fn relocated_mirror_edge_matches_independent_image_listener() {
    let gpu = VkBackend::new().unwrap();
    for radius in [0.2, 0.8] {
        let direct = trace(&gpu, radius, true, 0.0, vec3(-2.0, 1.0, 0.0), 1);
        let mixed = trace(&gpu, radius, true, 0.0, vec3(-2.0, 1.0, 0.0), 2);
        let image = trace(&gpu, radius, false, 0.0, vec3(-2.0, -1.0, 0.0), 1);
        let mut actual = [0.0_f64; 2];
        let mut expected = [0.0_f64; 2];
        for (index, ((a, b), c)) in mixed
            .samples
            .iter()
            .zip(&direct.samples)
            .zip(&image.samples)
            .enumerate()
        {
            let p = f64::from(a.left - b.left);
            let q = 0.8 * f64::from(c.left);
            actual[0] += p;
            actual[1] += index as f64 * p;
            expected[0] += q;
            expected[1] += index as f64 * q;
        }
        let ratio = actual[0] / expected[0];
        let delay_error = actual[1] / actual[0] - expected[1] / expected[0];
        eprintln!("mirror-edge radius={radius} pressure_ratio={ratio} delay_error_samples={delay_error}");
        assert!(expected[0] > 0.01, "image fixture must carry pressure");
        assert!((ratio - 1.0).abs() < 0.04, "mirror-edge pressure ratio {ratio}");
        assert!(delay_error.abs() < 1.0, "mirror-edge delay error {delay_error}");
    }
}

#[test]
fn relocated_anisotropic_strip_matches_direct_edge_integral() {
    // Integrate the original surface proposal over the capture strip and
    // compare against its independently evaluated density at the actual edge.
    for exponent in [1.0_f64, 8.0, 64.0] {
        let axis = [0.8_f64, 0.6];
        let response = |y: f64| ((axis[0] + axis[1] * y) / (1.0 + y * y).sqrt()).max(0.0).powf(exponent);
        for radius in [0.05_f64, 0.4, 1.0] {
            let mut integral = 0.0;
            for index in 0..16384 {
                let r = radius * (index as f64 + 0.5) / 16384.0;
                let distance = (1.0 + r * r).sqrt();
                let original_response = response(r);
                let proposal_area_density = original_response / distance.powi(3);
                let relocation = distance.powi(3) * response(0.0) / original_response;
                integral += proposal_area_density * relocation * 2.0 * (1.0 - r / radius) / radius * radius / 16384.0;
            }
            assert!((integral / response(0.0) - 1.0).abs() < 1e-9);
        }
    }
}

#[test]
fn axial_proposal_integrates_solid_angle_instead_of_a_cone() {
    use std::f64::consts::PI;
    for incoming_slope in [-3.0_f64, 0.0, 2.0] {
        for width in [0.02_f64, 0.3, 1.0] {
            let mut moments = [0.0_f64; 3];
            let steps = 262144;
            for index in 0..steps {
                let uniform = (index as f64 + 0.5) / steps as f64;
                let delta = width * (PI * (uniform - 0.5)).tan();
                let slope = incoming_slope + delta;
                let radial = 1.0 / (1.0 + slope * slope).sqrt();
                let cosine = slope * radial;
                let pdf = width / (PI * (width * width + delta * delta) * radial.powi(3));
                moments[0] += 1.0 / pdf / steps as f64;
                moments[1] += cosine / pdf / steps as f64;
                moments[2] += cosine * cosine / pdf / steps as f64;
            }
            // dOmega = d(azimuth) d(cosine); conditional cosine measure is 2.
            assert!((moments[0] - 2.0).abs() < 0.001);
            assert!(moments[1].abs() < 0.001);
            assert!((moments[2] - 2.0 / 3.0).abs() < 0.001);
        }
    }
}

#[test]
#[ignore = "requires Vulkan ray tracing; run hardware tests serially"]
fn relocated_rough_surface_is_stable_across_capture_widths() {
    let gpu = VkBackend::new().unwrap();
    let mut pressures = Vec::new();
    for radius in [0.2, 0.8] {
        let direct = trace(&gpu, radius, true, 0.5, vec3(-2.0, 1.0, 0.0), 1);
        let mixed = trace(&gpu, radius, true, 0.5, vec3(-2.0, 1.0, 0.0), 2);
        let pressure: f64 = mixed
            .samples
            .iter()
            .zip(&direct.samples)
            .map(|(a, b)| f64::from(a.left - b.left))
            .sum();
        eprintln!("rough-edge radius={radius} pressure={pressure}");
        pressures.push(pressure);
    }
    assert!(pressures.iter().all(|p| p.is_finite() && *p > 0.01));
    assert!((pressures[1] / pressures[0] - 1.0).abs() < 0.08);
}
