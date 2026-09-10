#[cfg(debug_assertions)]
use crate::core::error::ErrorCode;
use crate::core::error::{SoundError, SoundResult};
use crate::playback::device::{build_output_stream, default_output_sample_rate};
use crate::playback::queue::SampleQueue;
use crate::playback::state::PlaybackState;
use cpal::traits::StreamTrait;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// Queue and prebuffering policy for device playback.
#[derive(Debug, Clone, Copy)]
pub struct PlaybackConfig {
    /// Number of rendered blocks preferred before low-latency playback begins.
    pub prefill_blocks: usize,
    /// Maximum rendered audio retained ahead of the device callback, in seconds.
    pub max_queue_seconds: f32,
}

impl Default for PlaybackConfig {
    fn default() -> Self {
        Self {
            prefill_blocks: 8,
            max_queue_seconds: 0.1,
        }
    }
}

/// Audio-device output for rendered sound-engine blocks.
///
/// The player owns the CPAL stream and a bounded sample queue. It does not own
/// or render an engine; producers push interleaved blocks into it.
pub struct SoundPlayer {
    config: PlaybackConfig,
    state: Arc<PlaybackState>,
    queue: Arc<SampleQueue>,
    stream: cpal::Stream,
    #[cfg(debug_assertions)]
    block_samples: usize,
    /// Initial queue occupancy required before device consumption begins.
    prefill_samples: usize,
}

impl SoundPlayer {
    /// Creates and starts an output stream for interleaved engine blocks.
    pub fn new(
        sample_rate: u32,
        output_channels: usize,
        block_size: usize,
        config: PlaybackConfig,
    ) -> SoundResult<Self> {
        validate_config(config)?;
        if output_channels == 0 {
            return Err(SoundError::invalid_argument("output channel count must be > 0"));
        }
        if block_size == 0 {
            return Err(SoundError::invalid_argument("block_size must be > 0"));
        }

        let block_samples = block_size * output_channels;
        let state = Arc::new(PlaybackState::new());
        if config.prefill_blocks > 0 {
            state.begin_prebuffering();
        }
        let max_queued_blocks = max_queued_blocks(config, sample_rate, block_size);
        let queue = state.prepare_queue(max_queued_blocks * block_samples)?;
        let stream = build_output_stream(sample_rate, output_channels, queue.clone(), state.clone())?;

        stream
            .play()
            .map_err(|source| SoundError::external(source.to_string()))?;

        Ok(Self {
            config,
            state,
            queue,
            stream,
            #[cfg(debug_assertions)]
            block_samples,
            prefill_samples: config.prefill_blocks * block_samples,
        })
    }

    /// Returns the default output device sample rate.
    pub fn default_output_sample_rate() -> SoundResult<u32> {
        default_output_sample_rate()
    }

    /// Returns the queue and prebuffering policy used by this player.
    pub fn config(&self) -> PlaybackConfig {
        self.config
    }

    /// Returns underrun callbacks and missing engine frames after startup.
    /// Idle intervals and explicitly finished playback do not count as starvation.
    pub fn starvation_counts(&self) -> (u64, u64) {
        self.state.starvation_counts()
    }

    /// First/last active underrun seconds from this player's monotonic creation clock.
    /// Used by completed-run diagnostics; the clock does not control scheduling.
    pub(crate) fn starvation_time_range_seconds(&self) -> Option<(f64, f64)> {
        self.state.starvation_time_range_seconds()
    }

    /// Allows the engine worker to exclude intervals with no active voices.
    pub(crate) fn set_expects_audio(&self, active: bool) {
        self.state.set_expects_audio(active);
    }

    /// Releases a short voice's queued tail when no further blocks will arrive.
    pub(crate) fn release_prebuffer(&self) {
        self.state.release_prebuffer();
    }

    /// Pushes one interleaved rendered block into the playback queue.
    pub fn push_block(&self, block: &[f32]) -> SoundResult<bool> {
        crate::debug_validate!(
            block.len() == self.block_samples,
            ErrorCode::InvalidArgument,
            "playback block length {} must equal {}",
            block.len(),
            self.block_samples
        );

        let accepted = self.queue.push_block(block, self.state.stop_requested())?;
        if accepted && !self.state.ready_to_play() && self.queue.queued_samples()? >= self.prefill_samples {
            self.state.release_prebuffer();
        }
        Ok(accepted)
    }

    /// Marks the producer as finished so the callback can complete after draining.
    pub fn finish(&self) -> SoundResult<()> {
        self.state.mark_producer_done()
    }

    /// Blocks until the device callback drains all queued samples after `finish`.
    pub fn wait_until_finished(&self) {
        while !self.state.is_playback_done() {
            thread::sleep(Duration::from_millis(20));
        }
    }

    /// Requests playback stop.
    pub fn stop(&self) -> SoundResult<()> {
        self.state.request_stop()
    }

    /// Returns whether playback has been asked to stop.
    pub(crate) fn is_stop_requested(&self) -> bool {
        self.state.is_stop_requested()
    }
}

impl Drop for SoundPlayer {
    fn drop(&mut self) {
        let _ = self.state.request_stop();
        let _ = &self.stream;
    }
}

fn validate_config(config: PlaybackConfig) -> SoundResult<()> {
    if !config.max_queue_seconds.is_finite() || config.max_queue_seconds <= 0.0 {
        return Err(SoundError::invalid_argument("max_queue_seconds must be finite and > 0"));
    }
    Ok(())
}

fn max_queued_blocks(config: PlaybackConfig, sample_rate: u32, block_size: usize) -> usize {
    let queued_frames = config.max_queue_seconds * sample_rate as f32;
    let queued_blocks = (queued_frames / block_size as f32).ceil() as usize;
    queued_blocks.max(config.prefill_blocks).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_queue_blocks_from_seconds() {
        let config = PlaybackConfig {
            prefill_blocks: 4,
            max_queue_seconds: 0.5,
        };

        assert_eq!(max_queued_blocks(config, 48_000, 1024), 24);
    }

    #[test]
    fn queue_blocks_respect_prefill_floor() {
        let config = PlaybackConfig {
            prefill_blocks: 8,
            max_queue_seconds: 0.01,
        };

        assert_eq!(max_queued_blocks(config, 48_000, 1024), 8);
    }

    #[test]
    fn rejects_invalid_queue_seconds() {
        let mut config = PlaybackConfig::default();
        config.max_queue_seconds = f32::NAN;
        assert!(validate_config(config).is_err());

        config.max_queue_seconds = 0.0;
        assert!(validate_config(config).is_err());
    }
}
