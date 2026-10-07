use super::*;
use crate::core::config::*;
use crate::core::error::ErrorCode;
use glam::Vec3;

#[test]
fn play_request_defaults_to_unit_amplitude_and_volume() {
    let request = PlaySpatialSoundRequest::new(SoundId(7), Vec3::new(1.0, 2.0, 3.0));

    assert_eq!(request.sound_id, SoundId(7));
    assert_eq!(request.source.position, Vec3::new(1.0, 2.0, 3.0));
    assert_eq!(request.source.amplitude, 1.0);
    assert_eq!(request.volume, 1.0);
}

#[test]
fn rejects_non_finite_playback_volume() {
    let mut request = PlaySpatialSoundRequest::new(SoundId(1), Vec3::ZERO);
    request.volume = f32::NAN;
    assert!(validate_play_request(request).is_err());
}

#[test]
fn invalid_configuration_is_rejected_before_initializing_vulkan() {
    let valid = EngineConfig {
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
        cache: IrCacheConfig::default(),
        binaural: BinauralConfig::default(),
        auralization: AuralizationConfig {
            block_size: 64,
        },
    };
    valid.validate().unwrap();
    let invalidate: [fn(&mut EngineConfig); 12] = [
        |config| config.sound.sample_rate = 0,
        |config| config.sound.ir_num_samples = 0,
        |config| config.acoustics.listener_ray_count = 0,
        |config| config.acoustics.maximum_path_length = f32::NAN,
        |config| config.acoustics.minimum_throughput = -1.0,
        |config| config.cache.reuse_distance_meters = -1.0,
        |config| config.cache.listener_orientation_dot_threshold = 1.1,
        |config| config.binaural.head_radius_meters = 0.0,
        |config| config.binaural.max_ild_db = f32::INFINITY,
        |config| config.auralization.block_size = 0,
        |config| config.auralization.block_size = usize::MAX,
        |config| config.sound.ir_num_samples = u32::MAX,
    ];
    for invalidate in invalidate {
        let mut config = valid;
        invalidate(&mut config);
        let error = SoundEngine::new(config)
            .err()
            .expect("Invalid settings were accepted");
        assert_eq!(
            error.code(),
            ErrorCode::InvalidArgument,
            "{config:?}: {error}"
        );
    }
}
