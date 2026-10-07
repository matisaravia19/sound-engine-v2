use super::*;

#[test]
fn fft_dimensions_reject_zero_and_overflow_before_allocating() {
    let mut sound = SoundConfig {
        output_channels: OutputChannels::Stereo,
        sample_rate: 48_000,
        ir_num_samples: 1024,
    };
    let mut config = AuralizationConfig {
        block_size: 1024,
    };
    assert_eq!(config.fft_size(sound).unwrap(), 2048);
    config.block_size = 0;
    assert!(config.fft_size(sound).is_err());
    config.block_size = usize::MAX;
    assert!(config.fft_size(sound).is_err());
    config.block_size = 1024;
    sound.ir_num_samples = 0;
    assert!(config.fft_size(sound).is_err());
    sound.ir_num_samples = 1024;
    sound.sample_rate = 0;
    assert!(config.fft_size(sound).is_err());
}
