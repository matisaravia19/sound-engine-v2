//! Bounded, serial measurements for scene edits and convolution hot paths.
//! Run in release mode; pass `--gpu` to include GPU work after the CPU measurements.

use glam::{Mat4, Vec3};
use sound_engine::acoustics::IrSample;
use sound_engine::auralization::convolver::FftConvolver;
use sound_engine::core::config::*;
use sound_engine::core::error::SoundResult;
use sound_engine::gpu::backend::VkBackend;
use sound_engine::scene::*;
use sound_engine::{ListenerPose, PointSource};
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

fn main() -> SoundResult<()> {
    let mut scene = SceneStore::new();
    scene.load(SceneDescription {
        meshes: vec![MeshAsset {
            id: 1,
            vertices: [Vec3::ZERO, Vec3::X, Vec3::Y].repeat(300_000),
            indices: Vec::new(),
            opaque: true,
        }],
        materials: vec![Material {
            id: 1,
            absorption: 0.2,
            scattering: 0.0,
            roughness: 0.0,
        }],
        objects: vec![SceneObject {
            id: 1,
            mesh_id: 1,
            material_id: 1,
            transform: Mat4::IDENTITY,
            active: true,
        }],
        diffraction_edges: Vec::new(),
    })?;
    measure("scene_transform_900k_vertices", 32, || {
        scene.apply_updates(SceneUpdates {
            updates: vec![SceneUpdate::SetTransform {
                id: 1,
                transform: Mat4::from_translation(Vec3::X * (scene.version() % 10) as f32),
            }],
        })?;
        black_box(scene.version());
        Ok(())
    })?;
    measure("scene_empty_update_900k_vertices", 32, || {
        scene.apply_updates(SceneUpdates {
            updates: Vec::new(),
        })
    })?;

    if std::env::args().any(|argument| argument == "--gpu") {
        measure_gpu()?;
    }
    Ok(())
}

fn measure_gpu() -> SoundResult<()> {
    let config = EngineConfig {
        sound: SoundConfig {
            output_channels: OutputChannels::Stereo,
            sample_rate: 48_000,
            ir_num_samples: 1024,
        },
        acoustics: AcousticsConfig {
            listener_ray_count: 1,
            max_bounces: 0,
            maximum_path_length: 100.0,
            minimum_throughput: 0.0,
            sampling_seed: 0,
        },
        cache: Default::default(),
        binaural: Default::default(),
        auralization: AuralizationConfig {
            block_size: 1024,
        },
    };
    let gpu = Arc::new(VkBackend::new()?);
    let mut convolver = FftConvolver::new(gpu.clone(), config.sound, config.auralization)?;
    let samples = [IrSample::new(1.0, 0.5)];
    let ir = convolver.create_impulse_response(&samples)?;
    let mut stream = convolver.create_stream(ir)?;
    let input = vec![0.25; 1024];
    let mut output = vec![0.0; 2048];
    measure("convolution_1024_frames", 32, || {
        convolver.process_block(&mut stream, &input, &mut output)?;
        black_box(&output);
        Ok(())
    })?;
    measure("prepare_ir_2048_fft", 16, || {
        black_box(convolver.create_impulse_response(&samples)?);
        Ok(())
    })?;
    measure("reload_bundled_compute_shader", 4, || {
        black_box(gpu.shaders().load_glsl_file(
            sound_engine::gpu::shader::ShaderStage::Compute,
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/auralization/shaders/multiply.comp.glsl"
            ),
        )?);
        Ok(())
    })?;
    let mut acoustic = sound_engine::acoustics::AcousticPipeline::new(
        &gpu,
        sound_engine::acoustics::AcousticPipelineConfig {
            sound: config.sound,
            tracing: config.acoustics,
            binaural: config.binaural,
        },
    )?;
    let mut scene = SceneManager::new();
    let query = sound_engine::acoustics::AcousticQuery {
        query_id: 1,
        source: PointSource::new(Vec3::X),
        listener: ListenerPose::new(Vec3::ZERO, -Vec3::Z, Vec3::Y),
    };
    measure("empty_scene_ir_1024_samples", 16, || {
        black_box(acoustic.build_ir_gpu(&gpu, &mut scene, query)?);
        Ok(())
    })?;
    measure_scene_sync(&gpu)
}

fn measure_scene_sync(gpu: &VkBackend) -> SoundResult<()> {
    let material = Material {
        id: 1,
        absorption: 0.2,
        scattering: 0.0,
        roughness: 0.0,
    };
    let edge = DiffractionEdge {
        id: 1,
        start: Vec3::ZERO,
        end: Vec3::Y,
        bisector_dir: Vec3::X,
        edge_angle_radians: std::f32::consts::TAU,
        diffraction_radius: 0.1,
    };
    let mut scene = SceneManager::new();
    scene.load(SceneDescription {
        meshes: vec![MeshAsset {
            id: 1,
            vertices: vec![Vec3::ZERO, Vec3::X, Vec3::Y],
            indices: vec![],
            opaque: true,
        }],
        materials: vec![material.clone()],
        objects: (1..=64)
            .map(|id| SceneObject {
                id,
                mesh_id: 1,
                material_id: 1,
                transform: Mat4::from_translation(Vec3::X * id as f32 * 2.0),
                active: true,
            })
            .collect(),
        diffraction_edges: (1..=16)
            .map(|id| DiffractionEdge {
                id,
                start: Vec3::X * id as f32,
                end: Vec3::X * id as f32 + Vec3::Y,
                ..edge.clone()
            })
            .collect(),
    })?;
    scene.sync_gpu_if_needed(gpu)?;
    measure("scene_material_update_64_objects_16_edges", 16, || {
        let mut replacement = material.clone();
        replacement.absorption = if scene.version().is_multiple_of(2) {
            0.1
        } else {
            0.8
        };
        scene.apply_updates(SceneUpdates {
            updates: vec![SceneUpdate::ReplaceMaterial(replacement)],
        })?;
        black_box(scene.sync_gpu_if_needed(gpu)?);
        Ok(())
    })?;
    measure("scene_transform_update_64_objects_16_edges", 16, || {
        let transform = Mat4::from_translation(Vec3::Z * (scene.version() % 2) as f32);
        scene.apply_updates(SceneUpdates {
            updates: vec![SceneUpdate::SetTransform {
                id: 1,
                transform,
            }],
        })?;
        black_box(scene.sync_gpu_if_needed(gpu)?);
        Ok(())
    })?;
    measure("scene_edge_update_64_objects_16_edges", 16, || {
        let mut replacement = edge.clone();
        replacement.diffraction_radius = if scene.version().is_multiple_of(2) {
            0.1
        } else {
            0.2
        };
        scene.apply_updates(SceneUpdates {
            updates: vec![SceneUpdate::ReplaceDiffractionEdge(replacement)],
        })?;
        black_box(scene.sync_gpu_if_needed(gpu)?);
        Ok(())
    })
}

fn measure(
    name: &str,
    iterations: usize,
    mut operation: impl FnMut() -> SoundResult<()>,
) -> SoundResult<()> {
    for _ in 0..4 {
        operation()?;
    }
    let mut microseconds = Vec::with_capacity(5);
    for _ in 0..5 {
        let started = Instant::now();
        for _ in 0..iterations {
            operation()?;
        }
        microseconds.push(started.elapsed().as_secs_f64() * 1e6 / iterations as f64);
    }
    microseconds.sort_by(f64::total_cmp);
    println!(
        "{name}: median_us={:.3}, min_us={:.3}, max_us={:.3}, iterations={iterations}",
        microseconds[2], microseconds[0], microseconds[4]
    );
    Ok(())
}
