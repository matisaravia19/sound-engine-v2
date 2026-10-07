//! Configuration consumed by impulse-response generation.

use crate::core::config::{AcousticsConfig, BinauralConfig, SoundConfig};
use crate::core::error::{SoundError, SoundResult};

/// Output format, transport settings, and binaural model for acoustic ray tracing.
#[derive(Debug, Clone, Copy)]
pub struct AcousticPipelineConfig {
    /// Sample rate, response length, and channel layout of generated IRs.
    pub sound: SoundConfig,
    /// Ray sampling and propagation termination parameters.
    pub tracing: AcousticsConfig,
    /// Analytic head model used for binaural output.
    pub binaural: BinauralConfig,
}

impl AcousticPipelineConfig {
    /// Rejects invalid rendering settings before allocating GPU resources.
    pub(crate) fn validate(self) -> SoundResult<()> {
        if self.sound.sample_rate == 0 || self.sound.ir_num_samples == 0 {
            return Err(SoundError::invalid_argument(
                "IR sample rate and sample count must be non-zero",
            ));
        }
        if self.tracing.listener_ray_count == 0 {
            return Err(SoundError::invalid_argument(
                "listener_ray_count must be non-zero",
            ));
        }
        if !self.tracing.maximum_path_length.is_finite() || self.tracing.maximum_path_length <= 0.0
        {
            return Err(SoundError::invalid_argument(
                "maximum_path_length must be finite and positive",
            ));
        }
        if !self.tracing.minimum_throughput.is_finite() || self.tracing.minimum_throughput < 0.0 {
            return Err(SoundError::invalid_argument(
                "minimum_throughput must be finite and non-negative",
            ));
        }
        if !self.binaural.head_radius_meters.is_finite() || self.binaural.head_radius_meters <= 0.0
        {
            return Err(SoundError::invalid_argument(
                "binaural head radius must be finite and positive",
            ));
        }
        if !self.binaural.max_ild_db.is_finite() || self.binaural.max_ild_db < 0.0 {
            return Err(SoundError::invalid_argument(
                "binaural maximum ILD must be finite and non-negative",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/acoustics/config.rs"]
mod tests;
