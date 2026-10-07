use crate::core::error::{SoundError, SoundResult};

/// End-to-end runtime configuration for acoustic rendering.
#[derive(Debug, Clone, Copy)]
pub struct EngineConfig {
    /// Configuration for sound generation and IR construction.
    pub sound: SoundConfig,
    /// Configuration for acoustic simulation parameters.
    pub acoustics: AcousticsConfig,
    /// Configuration for acoustic IR caching and reuse.
    pub cache: IrCacheConfig,
    /// Configuration for analytic binaural rendering.
    pub binaural: BinauralConfig,
    /// Configuration for the auralization engine and output generation.
    pub auralization: AuralizationConfig,
}

impl EngineConfig {
    /// Checks all settings that do not depend on the selected GPU or audio device.
    /// Device limits remain checked when resources are created or work is dispatched.
    pub fn validate(self) -> SoundResult<()> {
        self.auralization.fft_size(self.sound)?;
        self.acoustics.validate()?;
        self.cache.validate()?;
        self.binaural.validate()
    }
}

/// Parameters for the angle-derived binaural estimator.
#[derive(Debug, Clone, Copy)]
pub struct BinauralConfig {
    /// Finite, positive radius of the spherical head model in meters.
    pub head_radius_meters: f32,
    /// Finite, nonnegative maximum broadband interaural level difference in decibels.
    pub max_ild_db: f32,
}

impl BinauralConfig {
    /// Checks finite, physical parameters for the analytic head model.
    pub(crate) fn validate(self) -> SoundResult<()> {
        if !self.head_radius_meters.is_finite() || self.head_radius_meters <= 0.0 {
            return Err(SoundError::invalid_argument(
                "binaural head radius must be finite and positive",
            ));
        }
        if !self.max_ild_db.is_finite() || self.max_ild_db < 0.0 {
            return Err(SoundError::invalid_argument(
                "binaural maximum ILD must be finite and non-negative",
            ));
        }
        Ok(())
    }
}

impl Default for BinauralConfig {
    fn default() -> Self {
        Self {
            head_radius_meters: 0.0875,
            max_ild_db: 12.0,
        }
    }
}

/// Runtime configuration for sound generation and impulse response construction.
#[derive(Debug, Clone, Copy)]
pub struct SoundConfig {
    /// Channels used for output audio generation.
    pub output_channels: OutputChannels,
    /// Nonzero sample rate shared by sound generation and IR construction.
    pub sample_rate: u32,
    /// Nonzero number of samples in each generated impulse response.
    pub ir_num_samples: u32,
}

impl SoundConfig {
    /// Requires a nonzero sample rate and impulse-response length.
    pub(crate) fn validate(self) -> SoundResult<()> {
        if self.sample_rate == 0 || self.ir_num_samples == 0 {
            return Err(SoundError::invalid_argument(
                "IR sample rate and sample count must be non-zero",
            ));
        }
        Ok(())
    }
}

/// Output channel layout used by the auralization engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputChannels {
    /// Mono output.
    Mono,
    /// Two-channel stereo output.
    Stereo,
    /// Two-channel binaural output.
    Binaural,
}

impl OutputChannels {
    /// Returns the number of interleaved output channels.
    pub fn count(self) -> usize {
        match self {
            OutputChannels::Mono => 1,
            OutputChannels::Stereo | OutputChannels::Binaural => 2,
        }
    }
}

/// Runtime configuration for acoustic simulation and ray tracing.
#[derive(Debug, Clone, Copy)]
pub struct AcousticsConfig {
    /// Nonzero number of Monte Carlo paths launched from the listener for each query.
    pub listener_ray_count: u32,
    /// Maximum surface hits and diffraction-proxy hits per path.
    /// A proxy consumes one interaction whether passed straight or diffracted.
    /// Diffraction has no separate event limit.
    pub max_bounces: u32,
    /// Finite, positive limit on completed or continued path length in world meters.
    pub maximum_path_length: f32,
    /// Finite, nonnegative absolute path-throughput threshold below which tracing stops.
    pub minimum_throughput: f32,
    /// Stable seed used to scramble listener directions and continuation samples.
    pub sampling_seed: u32,
}

impl AcousticsConfig {
    /// Checks ray count and finite path termination settings.
    pub(crate) fn validate(self) -> SoundResult<()> {
        if self.listener_ray_count == 0 {
            return Err(SoundError::invalid_argument(
                "listener_ray_count must be non-zero",
            ));
        }
        if !self.maximum_path_length.is_finite() || self.maximum_path_length <= 0.0 {
            return Err(SoundError::invalid_argument(
                "maximum_path_length must be finite and positive",
            ));
        }
        if !self.minimum_throughput.is_finite() || self.minimum_throughput < 0.0 {
            return Err(SoundError::invalid_argument(
                "minimum_throughput must be finite and non-negative",
            ));
        }
        Ok(())
    }
}

/// Runtime policy for acoustic impulse-response reuse.
#[derive(Debug, Clone, Copy)]
pub struct IrCacheConfig {
    /// Maximum source/listener movement in meters; finite and nonnegative.
    pub reuse_distance_meters: f32,
    /// Minimum dot product for both listener orientation axes; finite and in [-1, 1].
    pub listener_orientation_dot_threshold: f32,
    /// Maximum cached IR snapshots per engine; zero disables reuse.
    pub max_entries: usize,
    /// Maximum GPU bytes retained by the cache; zero disables reuse.
    pub max_gpu_cache_bytes: usize,
}

impl IrCacheConfig {
    /// Checks finite spatial tolerances; zero cache capacity remains a valid disabled policy.
    pub(crate) fn validate(self) -> SoundResult<()> {
        if !self.reuse_distance_meters.is_finite() || self.reuse_distance_meters < 0.0 {
            return Err(SoundError::invalid_argument(
                "IR reuse distance must be finite and nonnegative",
            ));
        }
        if !self.listener_orientation_dot_threshold.is_finite()
            || !(-1.0..=1.0).contains(&self.listener_orientation_dot_threshold)
        {
            return Err(SoundError::invalid_argument(
                "IR orientation threshold must be finite and in [-1, 1]",
            ));
        }
        Ok(())
    }
}

impl Default for IrCacheConfig {
    fn default() -> Self {
        Self {
            reuse_distance_meters: 0.2,
            listener_orientation_dot_threshold: 0.98,
            max_entries: 32,
            max_gpu_cache_bytes: 512 * 1024 * 1024,
        }
    }
}

/// Runtime configuration for the auralization engine.
#[derive(Debug, Clone, Copy)]
pub struct AuralizationConfig {
    /// Nonzero number of audio frames processed per render block.
    pub block_size: usize,
}

impl AuralizationConfig {
    /// Derives the padded FFT length for linear convolution before any allocation.
    pub(crate) fn fft_size(self, sound: SoundConfig) -> SoundResult<usize> {
        sound.validate()?;
        if self.block_size == 0 {
            return Err(SoundError::invalid_argument(
                "FFT block size must be non-zero",
            ));
        }
        self.block_size
            .checked_add(sound.ir_num_samples as usize - 1)
            .and_then(usize::checked_next_power_of_two)
            // Spectral multiplication uses a 32-bit count; complex buffers must
            // also fit Rust's addressable slice size before GPU initialization.
            .filter(|size| {
                *size <= u32::MAX as usize && *size <= isize::MAX as usize / size_of::<[f32; 2]>()
            })
            .ok_or_else(|| SoundError::invalid_argument("FFT size exceeds addressable capacity"))
    }
}

#[cfg(test)]
#[path = "../../tests/unit/core/config.rs"]
mod tests;
