use super::*;

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn prepared_filters_are_shared_owners_and_reject_incompatible_sample_rates() {
    let gpu = Arc::new(VkBackend::new().unwrap());
    let config = SoundConfig {
        output_channels: crate::core::config::OutputChannels::Mono,
        sample_rate: 48_000,
        ir_num_samples: 8,
    };
    let mut engine = AuralizationEngine::new(
        gpu.clone(),
        config,
        AuralizationConfig {
            block_size: 4,
        },
    )
    .unwrap();
    let sound = engine
        .insert_sound(SoundAsset::from_mono_samples(48_000, vec![0.25; 8]).unwrap())
        .unwrap();
    let impulse_response = engine
        .prepare_impulse_response(&[IrSample::new(1.0, 0.5)])
        .unwrap();
    let weak = Arc::downgrade(&impulse_response);
    let first = engine
        .play_sound(PlaySoundRequest::new(sound, impulse_response.clone()))
        .unwrap();
    let second = engine
        .play_sound(PlaySoundRequest::new(sound, impulse_response))
        .unwrap();
    assert_eq!(
        weak.strong_count(),
        2,
        "only the two voices retain the filter"
    );
    engine.stop_voice(first);
    assert_eq!(weak.strong_count(), 1);
    engine
        .update_voice_impulse_response(second, weak.upgrade().unwrap())
        .unwrap();
    assert_eq!(
        weak.strong_count(),
        1,
        "an unchanged filter must not start a crossfade"
    );
    let mut block = [0.0; 4];
    engine.render_block(&mut block).unwrap();
    assert!(block.iter().all(|sample| (*sample - 0.25).abs() < 1e-5));

    let mut other = AuralizationEngine::new(
        gpu,
        SoundConfig {
            sample_rate: 44_100,
            ..config
        },
        AuralizationConfig {
            block_size: 4,
        },
    )
    .unwrap();
    let wrong_rate = other
        .prepare_impulse_response(&[IrSample::new(1.0, 0.5)])
        .unwrap();
    let next_voice = engine.next_voice_id;
    assert!(
        engine
            .play_sound(PlaySoundRequest::new(sound, wrong_rate.clone()))
            .is_err()
    );
    assert_eq!(engine.next_voice_id, next_voice);
    assert!(
        engine
            .update_voice_impulse_response(second, wrong_rate)
            .is_err()
    );
    engine.render_block(&mut block).unwrap();
    assert!(block.iter().all(|sample| (*sample - 0.25).abs() < 1e-5));
    assert_eq!(engine.active_voice_count(), 1);
    engine.stop_voice(second);
    assert!(weak.upgrade().is_none());
}

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn mixes_partial_asset_blocks_and_drains_each_tail() {
    let gpu = Arc::new(VkBackend::new().unwrap());
    let mut engine = AuralizationEngine::new(
        gpu,
        SoundConfig {
            output_channels: crate::core::config::OutputChannels::Stereo,
            sample_rate: 48_000,
            ir_num_samples: 8,
        },
        AuralizationConfig {
            block_size: 4,
        },
    )
    .unwrap();
    let taps = [
        IrSample::new(1.0, 0.5),
        IrSample::zero(),
        IrSample::new(-0.5, 0.25),
    ];
    let impulse_response = engine.prepare_impulse_response(&taps).unwrap();
    let sounds = [
        (vec![0.25, -0.5, 1.0], 0.5),
        (vec![1.0, 0.0, -0.25, 0.5, -1.0, 0.25], -0.25),
    ];
    let mut expected = vec![[0.0; 2]; 8];
    for (samples, gain) in sounds {
        for (frame, &sample) in samples.iter().enumerate() {
            for (delay, tap) in taps.iter().enumerate() {
                expected[frame + delay][0] += sample * gain * tap.left;
                expected[frame + delay][1] += sample * gain * tap.right;
            }
        }
        let id = engine
            .insert_sound(SoundAsset::from_mono_samples(48_000, samples).unwrap())
            .unwrap();
        engine
            .play_sound(PlaySoundRequest {
                sound_id: id,
                impulse_response: impulse_response.clone(),
                gain,
            })
            .unwrap();
        engine.sound_bank_mut().remove(id).unwrap();
    }
    drop(impulse_response);

    let mut actual = Vec::new();
    let mut block = [77.0; 8];
    for _ in 0..16 {
        engine.render_block(&mut block).unwrap();
        actual.extend(block.chunks_exact(2).map(|frame| [frame[0], frame[1]]));
        if !engine.has_active_voices() {
            break;
        }
    }
    assert!(!engine.has_active_voices(), "all finite tails must finish");
    assert!(actual.len() >= expected.len());
    expected.resize(actual.len(), [0.0; 2]);
    for (actual, expected) in actual.iter().zip(expected) {
        for channel in 0..2 {
            assert!((actual[channel] - expected[channel]).abs() < 1e-5);
        }
    }
    block.fill(77.0);
    engine.render_block(&mut block).unwrap();
    assert_eq!(block, [0.0; 8]);
}

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
    let impulse_response = engine
        .prepare_impulse_response(&[IrSample::new(1.0, 0.5)])
        .unwrap();
    let sound = engine
        .insert_sound(SoundAsset::from_mono_samples(48_000, vec![0.25; 128]).unwrap())
        .unwrap();
    let voice = engine
        .play_sound(PlaySoundRequest::new(sound, impulse_response.clone()))
        .unwrap();
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
            .play_sound(PlaySoundRequest::new(wrong_rate, impulse_response.clone()))
            .is_err()
    );
    let valid = engine
        .insert_sound(SoundAsset::from_mono_samples(48_000, vec![0.0; 64]).unwrap())
        .unwrap();
    engine.next_voice_id = u64::MAX;
    assert!(
        engine
            .play_sound(PlaySoundRequest::new(valid, impulse_response))
            .is_err()
    );
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
