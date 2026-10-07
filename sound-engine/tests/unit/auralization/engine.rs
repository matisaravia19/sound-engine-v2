use super::*;

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn audit_voices_retain_assets_and_reject_invalid_registry_state() {
    let gpu = Arc::new(VkBackend::new().unwrap());
    let mut engine = AuralizationEngine::new(
        gpu,
        SoundConfig {
            output_channels: crate::core::config::OutputChannels::Stereo,
            sample_rate: 48_000,
            ir_num_samples: 128,
        },
        AuralizationConfig {
            block_size: 64,
        },
    )
    .unwrap();
    engine
        .register_impulse_response(1, &[IrSample::new(1.0, 0.5)])
        .unwrap();
    let sound = engine
        .insert_sound(SoundAsset::from_mono_samples(48_000, vec![0.25; 128]).unwrap())
        .unwrap();
    let voice = engine.play_sound(PlaySoundRequest::new(sound, 1)).unwrap();
    engine.sound_bank_mut().remove(sound).unwrap();
    let mut block = vec![0.0; 128];
    engine.render_block(&mut block).unwrap();
    for frame in block.chunks_exact(2) {
        assert!((frame[0] - 0.25).abs() < 1e-4);
        assert!((frame[1] - 0.125).abs() < 1e-4);
    }
    let wrong_rate = engine
        .sound_bank_mut()
        .insert(SoundAsset::from_mono_samples(44_100, vec![1.0; 64]).unwrap())
        .unwrap();
    assert!(
        engine
            .play_sound(PlaySoundRequest::new(wrong_rate, 1))
            .is_err()
    );
    let valid = engine
        .insert_sound(SoundAsset::from_mono_samples(48_000, vec![0.0; 64]).unwrap())
        .unwrap();
    engine.next_voice_id = u64::MAX;
    assert!(engine.play_sound(PlaySoundRequest::new(valid, 1)).is_err());
    assert_eq!(engine.active_voice_count(), 1);
    assert!(engine.has_voice(voice));
}

#[test]
fn renders_asset_block_with_gain_and_padding() {
    let asset = SoundAsset::from_mono_samples(48_000, vec![0.5, -0.25]).unwrap();
    let mut cursor = 0;
    let mut block = vec![1.0; 4];

    render_asset_samples(&asset, &mut cursor, 2.0, &mut block);

    assert_eq!(block, vec![1.0, -0.5, 0.0, 0.0]);
    assert_eq!(cursor, 2);
}
