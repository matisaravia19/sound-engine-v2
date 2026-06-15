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
    /// Maximum number of rendered blocks retained ahead of the device callback.
    pub max_queued_blocks: usize,
}

impl Default for PlaybackConfig {
    fn default() -> Self {
        Self {
            prefill_blocks: 8,
            max_queued_blocks: 32,
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
    block_samples: usize,
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
        let queue = state.prepare_queue(config.max_queued_blocks * block_samples)?;
        let stream = build_output_stream(sample_rate, output_channels, queue.clone(), state.clone())?;
        stream
            .play()
            .map_err(|source| SoundError::external(source.to_string()))?;

        Ok(Self {
            config,
            state,
            queue,
            stream,
            block_samples,
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

    /// Pushes one interleaved rendered block into the playback queue.
    pub fn push_block(&self, block: &[f32]) -> SoundResult<bool> {
        if block.len() != self.block_samples {
            return Err(SoundError::invalid_argument(format!(
                "playback block length {} must equal {}",
                block.len(),
                self.block_samples
            )));
        }
        self.queue.push_block(block, self.state.stop_requested())
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
    if config.max_queued_blocks == 0 {
        return Err(SoundError::invalid_argument("max_queued_blocks must be > 0"));
    }
    if config.prefill_blocks > config.max_queued_blocks {
        return Err(SoundError::invalid_argument(
            "prefill_blocks must be less than or equal to max_queued_blocks",
        ));
    }
    Ok(())
}
