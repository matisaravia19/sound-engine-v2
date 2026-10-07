use super::*;
use glam::Vec3;

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn audit_cache_hits_reject_a_foreign_backend() {
    let gpu = VkBackend::new().unwrap();
    let foreign = VkBackend::new().unwrap();
    let mut engine = direct_engine(&gpu, IrCacheConfig::default());
    let source = PointSource::new(Vec3::X);
    let listener = ListenerPose::default();
    let first = engine
        .build_impulse_response_gpu(&gpu, source, listener)
        .unwrap();
    assert!(
        engine
            .build_impulse_response_gpu(&foreign, source, listener)
            .is_err()
    );
    let reused = engine
        .build_impulse_response_gpu(&gpu, source, listener)
        .unwrap();
    assert!(std::sync::Arc::ptr_eq(first.buffer(), reused.buffer()));
    assert!(first.download(&foreign).is_err());
}

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn cached_queries_preserve_the_traced_basis_and_distinguish_silence() {
    let gpu = VkBackend::new().unwrap();
    let mut engine = direct_engine(
        &gpu,
        IrCacheConfig {
            reuse_distance_meters: 0.0,
            listener_orientation_dot_threshold: 1.0,
            ..IrCacheConfig::default()
        },
    );
    let source = PointSource::new(Vec3::X);
    for step in 0..24 {
        let rotation =
            glam::Quat::from_rotation_y(step as f32 * 0.17) * glam::Quat::from_rotation_x(0.37);
        let listener = ListenerPose::new(Vec3::ZERO, rotation * -Vec3::Z, rotation * Vec3::Y);
        let expected = AcousticQuery {
            query_id: 0,
            source,
            listener,
        }
        .canonicalized()
        .unwrap()
        .as_query();
        let first = engine
            .build_impulse_response_gpu(&gpu, source, listener)
            .unwrap();
        assert_eq!(
            first.query(),
            expected,
            "normalization changed after tracing"
        );
        let reused = engine
            .build_impulse_response_gpu(&gpu, source, listener)
            .unwrap();
        assert!(std::sync::Arc::ptr_eq(first.buffer(), reused.buffer()));
    }

    let listener = ListenerPose::default();
    let mut source = source;
    source.amplitude = f32::EPSILON * 0.5;
    let quiet = engine
        .build_impulse_response_gpu(&gpu, source, listener)
        .unwrap();
    assert!(quiet.download(&gpu).unwrap().energy > 0.0);
    source.amplitude = 0.0;
    let silent = engine
        .build_impulse_response_gpu(&gpu, source, listener)
        .unwrap();
    assert!(!std::sync::Arc::ptr_eq(quiet.buffer(), silent.buffer()));
    assert!(
        silent
            .download(&gpu)
            .unwrap()
            .samples
            .iter()
            .all(|sample| sample.left == 0.0 && sample.right == 0.0)
    );
}

fn direct_engine(gpu: &VkBackend, cache: IrCacheConfig) -> AcousticsEngine {
    use crate::core::config::*;
    AcousticsEngine::new(
        gpu,
        AcousticPipelineConfig {
            sound: SoundConfig {
                sample_rate: 48_000,
                output_channels: OutputChannels::Stereo,
                ir_num_samples: 1024,
            },
            tracing: AcousticsConfig {
                listener_ray_count: 1,
                max_bounces: 0,
                maximum_path_length: 100.0,
                minimum_throughput: 0.0,
                sampling_seed: 0,
            },
            binaural: Default::default(),
        },
        cache,
    )
    .unwrap()
}
