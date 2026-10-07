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
        IrCacheConfig::default(),
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
