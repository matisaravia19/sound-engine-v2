use super::*;
use acoustic_fixtures::geometry;

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
