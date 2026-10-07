use super::*;
use crate::core::config::OutputChannels;

fn test_config() -> AcousticPipelineConfig {
    AcousticPipelineConfig {
        sound: SoundConfig {
            output_channels: OutputChannels::Binaural,
            sample_rate: 48_000,
            ir_num_samples: 16_384,
        },
        tracing: AcousticsConfig {
            listener_ray_count: 1,
            max_bounces: 0,
            maximum_path_length: 343.0,
            minimum_throughput: 0.0,
            sampling_seed: 0,
        },
        binaural: BinauralConfig::default(),
    }
}

#[test]
fn rejects_invalid_binaural_configuration() {
    let mut config = test_config();
    config.binaural.head_radius_meters = 0.0;
    assert!(config.validate().is_err());

    config = test_config();
    config.binaural.head_radius_meters = f32::NAN;
    assert!(config.validate().is_err());

    config = test_config();
    config.binaural.max_ild_db = -1.0;
    assert!(config.validate().is_err());

    config = test_config();
    config.binaural.max_ild_db = f32::INFINITY;
    assert!(config.validate().is_err());
}
