use crate::core::error::{ErrorCode, SoundError, SoundResult};
use hound::{SampleFormat, WavReader};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Identifier for a playable dry sound asset.
pub type SoundId = u64;

/// Decoded mono sound data stored at the engine sample rate.
///
/// The asset owns normalized `f32` samples so active voices can render blocks
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
        Ok(Self {
            sample_rate,
            samples: Arc::from(samples),
        })
    }

    /// Loads a WAV file, mixes it to mono, and resamples it to `target_sample_rate`.
    pub fn from_wav_file(path: impl AsRef<Path>, target_sample_rate: u32) -> SoundResult<Self> {
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
        if samples.len() % channels as usize != 0 {
            return Err(SoundError::invalid_argument(
                "interleaved sample count must be divisible by channel count",
            ));
        }

        let mono = mix_interleaved_to_mono(&samples, channels as usize);
        let samples = if source_sample_rate == target_sample_rate {
            mono
        } else {
            resample_linear(&mono, source_sample_rate, target_sample_rate)
        };

        Self::from_mono_samples(target_sample_rate, samples)
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
/// auralization engine, allowing voices to keep playing after new assets are
/// inserted into the bank.
#[derive(Debug, Default)]
pub struct SoundBank {
    next_id: SoundId,
    assets: HashMap<SoundId, SoundAsset>,
}

impl SoundBank {
    /// Creates an empty sound bank.
    pub fn new() -> Self {
        Self {
            next_id: 1,
            assets: HashMap::new(),
        }
    }

    /// Inserts an already decoded asset and returns its generated id.
    pub fn insert(&mut self, asset: SoundAsset) -> SoundId {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        self.assets.insert(id, asset);
        id
    }

    /// Loads a WAV file into the bank at `target_sample_rate`.
    pub fn load_wav(&mut self, path: impl AsRef<Path>, target_sample_rate: u32) -> SoundResult<SoundId> {
        let asset = SoundAsset::from_wav_file(path, target_sample_rate)?;
        Ok(self.insert(asset))
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
                .map_err(|source| SoundError::with_source(ErrorCode::Io, "Failed to read WAV samples", source))?,
            (SampleFormat::Int, 16) => reader
                .samples::<i16>()
                .map(|sample| sample.map(|x| x as f32 / i16::MAX as f32))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|source| SoundError::with_source(ErrorCode::Io, "Failed to read WAV samples", source))?,
            (SampleFormat::Int, 24) => reader
                .samples::<i32>()
                .map(|sample| sample.map(|x| x as f32 / 8_388_607.0))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|source| SoundError::with_source(ErrorCode::Io, "Failed to read WAV samples", source))?,
            (SampleFormat::Int, 32) => reader
                .samples::<i32>()
                .map(|sample| sample.map(|x| x as f32 / i32::MAX as f32))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|source| SoundError::with_source(ErrorCode::Io, "Failed to read WAV samples", source))?,
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

fn mix_interleaved_to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32)
        .collect()
}

fn resample_linear(samples: &[f32], source_sample_rate: u32, target_sample_rate: u32) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }

    let ratio = target_sample_rate as f64 / source_sample_rate as f64;
    let output_len = ((samples.len() as f64) * ratio).ceil().max(1.0) as usize;
    let source_step = source_sample_rate as f64 / target_sample_rate as f64;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let source_pos = i as f64 * source_step;
        let left_index = source_pos.floor() as usize;
        let right_index = (left_index + 1).min(samples.len() - 1);
        let fraction = (source_pos - left_index as f64) as f32;
        let left = samples[left_index.min(samples.len() - 1)];
        let right = samples[right_index];
        output.push(left + (right - left) * fraction);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixes_interleaved_stereo_to_mono() {
        let asset = SoundAsset::from_interleaved_samples(48_000, 2, vec![1.0, -1.0, 0.25, 0.75], 48_000).unwrap();

        assert_eq!(asset.samples(), &[0.0, 0.5]);
    }

    #[test]
    fn resamples_to_target_rate() {
        let asset = SoundAsset::from_interleaved_samples(2, 1, vec![0.0, 1.0], 4).unwrap();

        assert_eq!(asset.sample_rate(), 4);
        assert_eq!(asset.samples(), &[0.0, 0.5, 1.0, 1.0]);
    }
}
