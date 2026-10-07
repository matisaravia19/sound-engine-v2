use crate::core::error::{ErrorCode, SoundError, SoundResult};
use hound::{SampleFormat, WavReader};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Identifier for a playable dry sound asset.
pub type SoundId = u64;

/// Decoded mono sound data stored at the engine sample rate.
///
/// The asset owns finite `f32` samples so active voices can render blocks
/// without I/O, channel conversion, or sample-rate conversion on the audio path.
#[derive(Debug, Clone)]
pub struct SoundAsset {
    sample_rate: u32,
    samples: Arc<[f32]>,
}

impl SoundAsset {
    /// Creates an asset from mono samples at the engine sample rate.
    pub fn from_mono_samples(sample_rate: u32, samples: Vec<f32>) -> SoundResult<Self> {
        validate_sample_rate(sample_rate)?;
        validate_samples(&samples)?;
        Ok(Self {
            sample_rate,
            samples: Arc::from(samples),
        })
    }

    /// Loads a WAV file, mixes it to mono, and resamples it to `target_sample_rate`.
    pub fn from_wav_file(path: impl AsRef<Path>, target_sample_rate: u32) -> SoundResult<Self> {
        validate_sample_rate(target_sample_rate)?;
        let decoded = DecodedWav::read(path)?;
        Self::from_interleaved_samples(
            decoded.sample_rate,
            decoded.channels,
            decoded.samples,
            target_sample_rate,
        )
    }

    /// Creates a mono asset from interleaved samples, resampling when needed.
    pub fn from_interleaved_samples(
        source_sample_rate: u32,
        channels: u16,
        samples: Vec<f32>,
        target_sample_rate: u32,
    ) -> SoundResult<Self> {
        validate_sample_rate(source_sample_rate)?;
        validate_sample_rate(target_sample_rate)?;
        if channels == 0 {
            return Err(SoundError::invalid_argument("channel count must be > 0"));
        }
        if !samples.len().is_multiple_of(channels as usize) {
            return Err(SoundError::invalid_argument(
                "interleaved sample count must be divisible by channel count",
            ));
        }

        validate_samples(&samples)?;
        let mono = mix_interleaved_to_mono(samples, channels as usize);
        let samples = if source_sample_rate == target_sample_rate {
            mono
        } else {
            resample_linear(&mono, source_sample_rate, target_sample_rate)?
        };

        Ok(Self {
            sample_rate: target_sample_rate,
            samples: Arc::from(samples),
        })
    }

    /// Returns the asset sample rate in Hz.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Returns the mono sample data.
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// Returns the number of mono frames in the asset.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Returns whether the asset contains no samples.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// In-memory registry of dry sound assets.
///
/// The bank owns immutable assets and hands out shared references to the
/// auralization engine. Active voices retain their own asset clone and keep
/// playing when the corresponding bank entry is removed.
#[derive(Debug)]
pub struct SoundBank {
    next_id: SoundId,
    assets: HashMap<SoundId, SoundAsset>,
}

impl Default for SoundBank {
    fn default() -> Self {
        Self::new()
    }
}

impl SoundBank {
    /// Creates an empty sound bank.
    pub fn new() -> Self {
        Self {
            next_id: 1,
            assets: HashMap::new(),
        }
    }

    /// Inserts a decoded asset, or reports exhaustion without replacing a live id.
    pub fn insert(&mut self, asset: SoundAsset) -> SoundResult<SoundId> {
        let id = crate::core::id::allocate_id(&mut self.next_id, "Sound")?;
        self.assets.insert(id, asset);
        Ok(id)
    }

    /// Loads a WAV file into the bank at `target_sample_rate`.
    pub fn load_wav(
        &mut self,
        path: impl AsRef<Path>,
        target_sample_rate: u32,
    ) -> SoundResult<SoundId> {
        let asset = SoundAsset::from_wav_file(path, target_sample_rate)?;
        self.insert(asset)
    }

    /// Returns an asset by id.
    pub fn get(&self, id: SoundId) -> SoundResult<&SoundAsset> {
        self.assets
            .get(&id)
            .ok_or_else(|| SoundError::not_found(format!("Unknown sound id {id}")))
    }

    /// Removes an asset from the bank.
    pub fn remove(&mut self, id: SoundId) -> Option<SoundAsset> {
        self.assets.remove(&id)
    }

    /// Returns the number of assets in the bank.
    pub fn len(&self) -> usize {
        self.assets.len()
    }

    /// Returns whether the bank contains no assets.
    pub fn is_empty(&self) -> bool {
        self.assets.is_empty()
    }
}

struct DecodedWav {
    sample_rate: u32,
    channels: u16,
    samples: Vec<f32>,
}

impl DecodedWav {
    fn read(path: impl AsRef<Path>) -> SoundResult<Self> {
        let mut reader = WavReader::open(path.as_ref()).map_err(|source| {
            SoundError::with_source(
                ErrorCode::Io,
                format!("Failed to open WAV file {}", path.as_ref().display()),
                source,
            )
        })?;
        let spec = reader.spec();
        let samples = match (spec.sample_format, spec.bits_per_sample) {
            (SampleFormat::Float, 32) => reader
                .samples::<f32>()
                .collect::<Result<Vec<_>, _>>()
                .map_err(|source| {
                    SoundError::with_source(ErrorCode::Io, "Failed to read WAV samples", source)
                })?,
            (SampleFormat::Int, 16) => reader
                .samples::<i16>()
                .map(|sample| sample.map(|x| x as f32 / i16::MAX as f32))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|source| {
                    SoundError::with_source(ErrorCode::Io, "Failed to read WAV samples", source)
                })?,
            (SampleFormat::Int, 24) => reader
                .samples::<i32>()
                .map(|sample| sample.map(|x| x as f32 / 8_388_607.0))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|source| {
                    SoundError::with_source(ErrorCode::Io, "Failed to read WAV samples", source)
                })?,
            (SampleFormat::Int, 32) => reader
                .samples::<i32>()
                .map(|sample| sample.map(|x| x as f32 / i32::MAX as f32))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|source| {
                    SoundError::with_source(ErrorCode::Io, "Failed to read WAV samples", source)
                })?,
            _ => {
                return Err(SoundError::unsupported_feature(format!(
                    "Unsupported WAV format: {:?} {} bits",
                    spec.sample_format, spec.bits_per_sample
                )));
            }
        };

        Ok(Self {
            sample_rate: spec.sample_rate,
            channels: spec.channels,
            samples,
        })
    }
}

fn validate_sample_rate(sample_rate: u32) -> SoundResult<()> {
    if sample_rate == 0 {
        return Err(SoundError::invalid_argument("sample_rate must be > 0"));
    }
    Ok(())
}

fn validate_samples(samples: &[f32]) -> SoundResult<()> {
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(SoundError::invalid_argument("Sound samples must be finite"));
    }
    Ok(())
}

fn mix_interleaved_to_mono(mut samples: Vec<f32>, channels: usize) -> Vec<f32> {
    if channels == 1 {
        return samples;
    }
    let frames = samples.len() / channels;
    for frame in 0..frames {
        // Compact behind the unread frames, avoiding a second full-size buffer.
        // Wider accumulation prevents finite channels overflowing before division.
        let sum: f64 = samples[frame * channels..(frame + 1) * channels]
            .iter()
            .map(|sample| *sample as f64)
            .sum();
        samples[frame] = (sum / channels as f64) as f32;
    }
    samples.truncate(frames);
    samples
}

fn resampled_len(frames: usize, source_rate: u32, target_rate: u32) -> SoundResult<usize> {
    let length = (frames as u128 * target_rate as u128).div_ceil(source_rate as u128);
    usize::try_from(length)
        .ok()
        .filter(|length| *length <= isize::MAX as usize / std::mem::size_of::<f32>())
        .ok_or_else(|| SoundError::invalid_argument("Resampled audio exceeds addressable capacity"))
}

fn resample_linear(
    samples: &[f32],
    source_sample_rate: u32,
    target_sample_rate: u32,
) -> SoundResult<Vec<f32>> {
    let output_len = resampled_len(samples.len(), source_sample_rate, target_sample_rate)?;
    let source_step = source_sample_rate as f64 / target_sample_rate as f64;
    let mut output = Vec::new();
    output.try_reserve_exact(output_len).map_err(|error| {
        SoundError::resource_allocation_failed(format!("Cannot allocate resampled audio: {error}"))
    })?;

    for i in 0..output_len {
        let source_pos = i as f64 * source_step;
        let left_index = (source_pos.floor() as usize).min(samples.len() - 1);
        let right_index = (left_index + 1).min(samples.len() - 1);
        let fraction = (source_pos - left_index as f64).clamp(0.0, 1.0);
        let left = samples[left_index] as f64;
        let right = samples[right_index] as f64;
        output.push((left * (1.0 - fraction) + right * fraction) as f32);
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
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
        let mixed =
            SoundAsset::from_interleaved_samples(2, 2, vec![f32::MAX, f32::MAX], 2).unwrap();
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
}
