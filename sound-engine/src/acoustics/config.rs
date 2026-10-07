//! Configuration consumed by impulse-response generation.

use crate::core::config::{AcousticsConfig, BinauralConfig, SoundConfig};
use crate::core::error::SoundResult;

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
        self.sound.validate()?;
        self.tracing.validate()?;
        self.binaural.validate()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/acoustics/config.rs"]
mod tests;
