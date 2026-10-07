use crate::core::error::{ErrorCode, SoundError, SoundResult};
use hound::{SampleFormat, WavReader};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Opaque identifier for a dry asset in the sound bank that allocated it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SoundId(
    /// Monotonic bank-local value; never reused after removal.
    pub(crate) u64,
);

impl std::fmt::Display for SoundId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

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
        let reader = WavReader::open(path.as_ref()).map_err(|source| {
            SoundError::with_source(
                ErrorCode::Io,
                format!("Failed to open WAV file {}", path.as_ref().display()),
                source,
            )
        })?;
        let spec = reader.spec();
        Self::from_interleaved_samples(
            spec.sample_rate,
            spec.channels,
            read_wav_samples(reader)?,
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
    next_id: u64,
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
        let id = SoundId(crate::core::id::allocate_id(&mut self.next_id, "Sound")?);
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

fn read_wav_samples(mut reader: WavReader<impl std::io::Read>) -> SoundResult<Vec<f32>> {
    let spec = reader.spec();
    match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Float, 32) => reader.samples::<f32>().collect::<Result<Vec<_>, _>>(),
        (SampleFormat::Int, bits @ (16 | 24 | 32)) => {
            let peak = ((1_u64 << (bits - 1)) - 1) as f32;
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 / peak))
                .collect()
        }
        _ => {
            return Err(SoundError::unsupported_feature(format!(
                "Unsupported WAV format: {:?} {} bits",
                spec.sample_format, spec.bits_per_sample
            )));
        }
    }
    .map_err(|source| SoundError::with_source(ErrorCode::Io, "Failed to read WAV samples", source))
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
#[path = "../../tests/unit/auralization/assets.rs"]
mod tests;
