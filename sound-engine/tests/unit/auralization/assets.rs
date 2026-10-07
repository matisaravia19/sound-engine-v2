use super::*;
use std::io::Cursor;

#[test]
fn wav_decoding_preserves_supported_integer_and_float_levels() {
    for (bits, peak, quarter, expected_quarter) in [
        (16, 32_767, 8_192, 0.250_007_63),
        (24, 8_388_607, 2_097_152, 0.250_000_03),
        (32, 2_147_483_647, 536_870_912, 0.25),
    ] {
        let bytes = encoded_wav(
            bits,
            SampleFormat::Int,
            &[0, quarter, -quarter, peak, -peak],
        );
        let reader = WavReader::new(Cursor::new(bytes)).unwrap();
        let samples = read_wav_samples(reader).unwrap();
        let expected = [0.0, expected_quarter, -expected_quarter, 1.0, -1.0];
        assert_eq!(samples.len(), expected.len());
        for (actual, expected) in samples.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-7,
                "{bits}-bit PCM: {samples:?}"
            );
        }
    }

    let expected = [0.0, 0.25, -0.75, 1.0, -1.0];
    let bytes = encoded_wav(32, SampleFormat::Float, &expected);
    let reader = WavReader::new(Cursor::new(bytes)).unwrap();
    assert_eq!(read_wav_samples(reader).unwrap(), expected);
}

#[test]
fn unsupported_and_truncated_wav_data_retain_distinct_errors() {
    let unsupported = encoded_wav(8, SampleFormat::Int, &[0, 127_i32]);
    let reader = WavReader::new(Cursor::new(unsupported)).unwrap();
    assert_eq!(
        read_wav_samples(reader).unwrap_err().code(),
        ErrorCode::UnsupportedFeature
    );

    let mut truncated = encoded_wav(16, SampleFormat::Int, &[0, 32_767_i32]);
    truncated.pop();
    let reader = WavReader::new(Cursor::new(truncated)).unwrap();
    let error = read_wav_samples(reader).unwrap_err();
    assert_eq!(error.code(), ErrorCode::Io);
    assert!(std::error::Error::source(&error).is_some());
}

fn encoded_wav<T: hound::Sample + Copy>(
    bits_per_sample: u16,
    sample_format: SampleFormat,
    samples: &[T],
) -> Vec<u8> {
    let mut output = Cursor::new(Vec::new());
    let mut writer = hound::WavWriter::new(
        &mut output,
        hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample,
            sample_format,
        },
    )
    .unwrap();
    for &sample in samples {
        writer.write_sample(sample).unwrap();
    }
    writer.finalize().unwrap();
    output.into_inner()
}

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
