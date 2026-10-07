use super::*;

#[test]
fn rejects_non_finite_audio_before_conversion() {
    for sample in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(SoundAsset::from_mono_samples(48_000, vec![sample]).is_err());
        assert!(
            SoundAsset::from_interleaved_samples(48_000, 2, vec![sample, 0.0], 44_100).is_err()
        );
    }
}

#[test]
fn finite_extreme_samples_do_not_overflow_mixing_or_interpolation() {
    let mixed = SoundAsset::from_interleaved_samples(2, 2, vec![f32::MAX, f32::MAX], 2).unwrap();
    assert_eq!(mixed.samples(), &[f32::MAX]);
    let resampled =
        SoundAsset::from_interleaved_samples(2, 1, vec![f32::MAX, -f32::MAX], 4).unwrap();
    assert!(resampled.samples().iter().all(|sample| sample.is_finite()));
    assert_eq!(resampled.samples()[1], 0.0);
}

#[test]
fn default_bank_uses_the_same_identifier_sequence_as_new() {
    let asset = SoundAsset::from_mono_samples(48_000, vec![1.0]).unwrap();
    assert_eq!(
        SoundBank::default().insert(asset.clone()).unwrap(),
        SoundBank::new().insert(asset).unwrap()
    );
}

#[test]
fn resampling_capacity_is_checked_before_allocation() {
    assert_eq!(resampled_len(441, 44_100, 48_000).unwrap(), 480);
    assert_eq!(resampled_len(1, 48_000, 44_100).unwrap(), 1);
    assert_eq!(resampled_len(0, 48_000, 44_100).unwrap(), 0);
    assert!(resampled_len(usize::MAX, 1, u32::MAX).is_err());
}

#[test]
fn exhausted_bank_preserves_existing_assets() {
    let mut bank = SoundBank::new();
    bank.next_id = u64::MAX - 1;
    let id = bank
        .insert(SoundAsset::from_mono_samples(48_000, vec![0.25]).unwrap())
        .unwrap();
    assert!(
        bank.insert(SoundAsset::from_mono_samples(48_000, vec![0.5]).unwrap())
            .is_err()
    );
    assert_eq!(bank.get(id).unwrap().samples(), &[0.25]);
    assert_eq!(bank.len(), 1);
}

#[test]
fn mixes_interleaved_stereo_to_mono() {
    let asset =
        SoundAsset::from_interleaved_samples(48_000, 2, vec![1.0, -1.0, 0.25, 0.75], 48_000)
            .unwrap();

    assert_eq!(asset.samples(), &[0.0, 0.5]);
}

#[test]
fn resamples_to_target_rate() {
    let asset = SoundAsset::from_interleaved_samples(2, 1, vec![0.0, 1.0], 4).unwrap();

    assert_eq!(asset.sample_rate(), 4);
    assert_eq!(asset.samples(), &[0.0, 0.5, 1.0, 1.0]);
}
