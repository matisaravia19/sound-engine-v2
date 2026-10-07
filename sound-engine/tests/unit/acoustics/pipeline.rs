//! Acoustic pipeline resource reuse and the host/shader ABI.

use super::*;
use crate::{ListenerPose, PointSource};
use glam::{Mat4, Vec3};

fn test_config() -> AcousticPipelineConfig {
    AcousticPipelineConfig {
        sound: crate::core::config::SoundConfig {
            output_channels: OutputChannels::Binaural,
            sample_rate: 48_000,
            ir_num_samples: 16_384,
        },
        tracing: crate::core::config::AcousticsConfig {
            listener_ray_count: 1,
            max_bounces: 0,
            maximum_path_length: 343.0,
            minimum_throughput: 0.0,
            sampling_seed: 0,
        },
        binaural: crate::core::config::BinauralConfig::default(),
    }
}

fn test_query() -> AcousticQuery {
    AcousticQuery {
        query_id: 1,
        source: PointSource::new(Vec3::Z),
        listener: ListenerPose::new(Vec3::ZERO, -Vec3::Z, Vec3::Y),
    }
}

#[test]
#[ignore = "requires Vulkan ray tracing; run hardware tests serially"]
fn audit_repeated_queries_reuse_tlas_and_snapshot_retains_device() {
    let gpu = VkBackend::new().unwrap();
    let pipeline = AcousticPipeline::new(&gpu, test_config()).unwrap();
    assert_eq!(gpu.rt().resource_counts(), (0, 0));
    let mut scene = SceneManager::new();
    let device = Arc::downgrade(&gpu.device_context());
    let mut previous_tlas = None;
    for _ in 0..4 {
        let ir = pipeline.build_ir(&gpu, &mut scene, test_query()).unwrap();
        assert!(
            ir.samples
                .iter()
                .all(|sample| sample.left.is_finite() && sample.right.is_finite())
        );
        assert!(ir.energy > 0.0);
        assert_eq!(gpu.rt().resource_counts(), (0, 1));
        let id = scene.sync_gpu_if_needed(&gpu).unwrap().tlas_id();
        assert!(previous_tlas.is_none_or(|previous| previous == id));
        previous_tlas = Some(id);
    }
    let snapshot = pipeline
        .build_ir_gpu(&gpu, &mut scene, test_query())
        .unwrap();
    drop(pipeline);
    assert_eq!(gpu.rt().pipeline_count(), 0);
    for _ in 0..2 {
        let temporary = AcousticPipeline::new(&gpu, test_config()).unwrap();
        assert_eq!(gpu.rt().pipeline_count(), 1);
        assert!(
            temporary
                .build_ir(&gpu, &mut scene, test_query())
                .unwrap()
                .energy
                > 0.0
        );
        assert_eq!(
            scene.sync_gpu_if_needed(&gpu).unwrap().tlas_id(),
            previous_tlas.unwrap()
        );
        assert_eq!(gpu.rt().resource_counts(), (0, 1));
        drop(temporary);
        assert_eq!(gpu.rt().pipeline_count(), 0);
    }
    drop(scene);
    assert_eq!(gpu.rt().resource_counts(), (0, 0));
    drop(gpu);
    assert!(device.upgrade().is_some());
    drop(snapshot);
    assert!(device.upgrade().is_none());
}

#[test]
#[ignore = "requires Vulkan ray tracing; run hardware tests serially"]
fn audit_scene_tlas_distinguishes_owners_and_geometry_updates() {
    use crate::scene::*;
    let gpu = VkBackend::new().unwrap();
    let mut config = test_config();
    config.sound.ir_num_samples = 1024;
    config.tracing.max_bounces = 1;
    config.tracing.listener_ray_count = 1024;
    let pipeline = AcousticPipeline::new(&gpu, config).unwrap();
    let wall = SceneDescription {
        meshes: vec![MeshAsset {
            id: 1,
            vertices: vec![
                Vec3::new(-10.0, -10.0, 0.5),
                Vec3::new(10.0, -10.0, 0.5),
                Vec3::new(0.0, 10.0, 0.5),
            ],
            indices: vec![],
            opaque: true,
        }],
        materials: vec![Material {
            id: 1,
            absorption: 1.0,
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
        diffraction_edges: vec![],
    };
    let mut obstructed = SceneManager::new();
    obstructed.load(wall.clone()).unwrap();
    let mut open_description = wall;
    open_description.objects[0].transform = Mat4::from_translation(Vec3::X * 100.0);
    let mut open = SceneManager::new();
    open.load(open_description).unwrap();
    assert_eq!(obstructed.version(), open.version());
    let obstructed_tlas = obstructed.sync_gpu_if_needed(&gpu).unwrap().tlas_id();
    let open_tlas = open.sync_gpu_if_needed(&gpu).unwrap().tlas_id();
    assert_ne!(obstructed_tlas, open_tlas);
    for _ in 0..4 {
        assert_eq!(
            pipeline
                .build_ir(&gpu, &mut obstructed, test_query())
                .unwrap()
                .energy,
            0.0
        );
        assert!(
            pipeline
                .build_ir(&gpu, &mut open, test_query())
                .unwrap()
                .energy
                > 0.0
        );
        // Each scene owns one complete traversal structure, shared across queries.
        assert_eq!(gpu.rt().resource_counts(), (2, 2));
        assert_eq!(
            obstructed.sync_gpu_if_needed(&gpu).unwrap().tlas_id(),
            obstructed_tlas
        );
        assert_eq!(open.sync_gpu_if_needed(&gpu).unwrap().tlas_id(), open_tlas);
    }
    // Immutable pipeline sharing must keep each scene's descriptors and IR separate.
    std::thread::scope(|scope| {
        let blocked = scope.spawn(|| pipeline.build_ir(&gpu, &mut obstructed, test_query()));
        let audible = scope.spawn(|| pipeline.build_ir(&gpu, &mut open, test_query()));
        assert_eq!(blocked.join().unwrap().unwrap().energy, 0.0);
        assert!(audible.join().unwrap().unwrap().energy > 0.0);
    });
    pipeline
        .build_ir(&gpu, &mut obstructed, test_query())
        .unwrap();
    obstructed
        .apply_updates(SceneUpdates {
            updates: vec![SceneUpdate::ReplaceMaterial(Material {
                id: 1,
                absorption: 0.0,
                scattering: 0.0,
                roughness: 0.0,
            })],
        })
        .unwrap();
    let reflected = pipeline
        .build_ir(&gpu, &mut obstructed, test_query())
        .unwrap();
    // A reflecting surface still blocks a source on the opposite side.
    assert_eq!(reflected.energy, 0.0);
    assert_eq!(
        obstructed.sync_gpu_if_needed(&gpu).unwrap().tlas_id(),
        obstructed_tlas
    );
    obstructed
        .apply_updates(SceneUpdates {
            updates: vec![SceneUpdate::SetActive {
                id: 1,
                active: false,
            }],
        })
        .unwrap();
    assert!(
        pipeline
            .build_ir(&gpu, &mut obstructed, test_query())
            .unwrap()
            .energy
            > 0.0
    );
    assert_ne!(
        obstructed.sync_gpu_if_needed(&gpu).unwrap().tlas_id(),
        obstructed_tlas
    );
    drop(open);
    drop(obstructed);
    assert_eq!(gpu.rt().resource_counts(), (0, 0));
    // Edge edits must rebuild traversal even when there are no surface objects.
    let mut edges = SceneManager::new();
    let mut edge = DiffractionEdge {
        id: 1,
        start: Vec3::new(0.0, -1.0, 0.5),
        end: Vec3::new(0.0, 1.0, 0.5),
        bisector_dir: Vec3::X,
        edge_angle_radians: std::f32::consts::TAU,
        diffraction_radius: 0.1,
    };
    edges
        .load(SceneDescription {
            meshes: vec![],
            materials: vec![],
            objects: vec![],
            diffraction_edges: vec![edge.clone()],
        })
        .unwrap();
    let mut previous = None;
    for step in 0..4 {
        edge.diffraction_radius = 0.1 + step as f32 * 0.1;
        edges
            .apply_updates(SceneUpdates {
                updates: vec![SceneUpdate::ReplaceDiffractionEdge(edge.clone())],
            })
            .unwrap();
        let ir = pipeline.build_ir(&gpu, &mut edges, test_query()).unwrap();
        assert!(
            ir.samples
                .iter()
                .all(|sample| sample.left.is_finite() && sample.right.is_finite())
        );
        let id = edges.sync_gpu_if_needed(&gpu).unwrap().tlas_id();
        assert!(previous.is_none_or(|previous| previous != id));
        previous = Some(id);
        assert_eq!(gpu.rt().resource_counts(), (1, 1));
    }
    drop(edges);
    assert_eq!(gpu.rt().resource_counts(), (0, 0));
    drop(pipeline);
}

#[test]
fn push_constants_match_shader_layout() {
    assert_eq!(size_of::<ListenerTracePushConstants>(), 84);
    assert_eq!(
        std::mem::offset_of!(ListenerTracePushConstants, minimum_throughput),
        72
    );
    assert_eq!(
        std::mem::offset_of!(ListenerTracePushConstants, head_radius_meters),
        76
    );
    assert_eq!(
        std::mem::offset_of!(ListenerTracePushConstants, max_ild_db),
        80
    );
}
