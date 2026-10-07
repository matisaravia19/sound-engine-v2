use super::*;
use glam::Vec3;

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn audit_cache_hits_reject_a_foreign_backend() {
    use crate::core::config::*;
    let gpu = VkBackend::new().unwrap();
    let foreign = VkBackend::new().unwrap();
    let mut engine = AcousticsEngine::new(
        &gpu,
        EngineConfig {
            sound: SoundConfig {
                sample_rate: 48_000,
                output_channels: OutputChannels::Stereo,
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
                block_size: 64,
            },
        },
    )
    .unwrap();
    let source = PointSource::new(Vec3::X);
    let listener = ListenerPose::default();
    engine
        .build_impulse_response_gpu(&gpu, source, listener)
        .unwrap();
    assert_eq!(engine.cache.len(), 1);
    assert!(
        engine
            .build_impulse_response_gpu(&foreign, source, listener)
            .is_err()
    );
    assert!(
        engine
            .build_impulse_response_gpu(&gpu, source, listener)
            .is_ok()
    );
}

#[test]
fn builds_cache_query_from_listener_and_source() {
    let listener = ListenerPose {
        position: Vec3::new(1.0, 2.0, 3.0),
        forward: -Vec3::Z,
        up: Vec3::Y,
    };
    let source = PointSource {
        position: Vec3::new(4.0, 5.0, 6.0),
        amplitude: 0.75,
    };

    let query = cache_query(9, source, listener);

    assert_eq!(query.scene_version, 9);
    assert_eq!(query.query_id, 0);
    assert_eq!(query.source_position, source.position);
    assert_eq!(query.source_amplitude, source.amplitude);
    assert_eq!(query.listener_position, listener.position);
    assert_eq!(query.listener_forward, listener.forward);
    assert_eq!(query.listener_up, listener.up);
}
