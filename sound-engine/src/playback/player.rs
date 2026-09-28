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
    // Production players always own a stream; tests drive the same queue/callback manually.
    stream: Option<cpal::Stream>,
    block_samples: usize,
    /// Initial queue occupancy required before device consumption begins.
    prefill_samples: usize,
}

impl SoundPlayer {
    /// Shares only cancellation state, allowing shutdown to wake a blocked producer.
    pub(crate) fn stop_handle(&self) -> PlaybackStopHandle {
        PlaybackStopHandle {
            state: self.state.clone(),
        }
    }

    /// Creates and starts an output stream for interleaved engine blocks.
    pub fn new(
        sample_rate: u32,
        output_channels: usize,
        block_size: usize,
        config: PlaybackConfig,
    ) -> SoundResult<Self> {
        validate_config(config)?;
        if sample_rate == 0 {
            return Err(SoundError::invalid_argument("sample_rate must be > 0"));
        }
        if output_channels == 0 {
            return Err(SoundError::invalid_argument(
                "output channel count must be > 0",
            ));
        }
        if block_size == 0 {
            return Err(SoundError::invalid_argument("block_size must be > 0"));
        }

        let block_samples = block_size
            .checked_mul(output_channels)
            .ok_or_else(|| SoundError::invalid_argument("Playback block size overflow"))?;
        let state = Arc::new(PlaybackState::new());
        if config.prefill_blocks > 0 {
            state.begin_prebuffering();
        }
        let max_queued_blocks = max_queued_blocks(config, sample_rate, block_size);
        let capacity = max_queued_blocks
            .checked_mul(block_samples)
            .filter(|samples| *samples <= isize::MAX as usize / std::mem::size_of::<f32>())
            .ok_or_else(|| SoundError::invalid_argument("Playback queue capacity overflow"))?;
        let queue = state.prepare_queue(capacity)?;
        let stream =
            build_output_stream(sample_rate, output_channels, queue.clone(), state.clone())?;

        stream
            .play()
            .map_err(|source| SoundError::external(source.to_string()))?;
        state.check_error()?;

        Ok(Self {
            config,
            state,
            queue,
            stream: Some(stream),
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
    /// Device failure returns an error; explicit cancellation returns `false`.
    /// Once `finish` closes production, further blocks return an invalid-state error.
    pub fn push_block(&self, block: &[f32]) -> SoundResult<bool> {
        crate::validate!(
            block.len() == self.block_samples,
            ErrorCode::InvalidArgument,
            "playback block length {} must equal {}",
            block.len(),
            self.block_samples
        );

        self.check_error()?;
        let accepted = self.queue.push_block(
            block,
            self.state.stop_requested(),
            self.state.producer_done(),
        );
        // Failure can arrive while this producer waits on a full queue.
        self.check_error()?;
        let accepted = accepted?;
        if accepted
            && !self.state.ready_to_play()
            && self.queue.queued_samples()? >= self.prefill_samples
        {
            self.state.release_prebuffer();
        }
        Ok(accepted)
    }

    /// Closes production while allowing the callback to drain queued samples.
    /// Pending pushes wake with an invalid-state error; repeated finishes are harmless.
    /// Returns any device/callback failure already reported by this stream.
    pub fn finish(&self) -> SoundResult<()> {
        self.state.mark_producer_done()
    }

    /// Blocks until the device callback drains all queued samples after `finish`.
    /// Returns the retained device/callback failure if playback was interrupted.
    pub fn wait_until_finished(&self) -> SoundResult<()> {
        while !self.state.is_playback_done() {
            thread::sleep(Duration::from_millis(20));
        }
        self.check_error()
    }

    /// Reports the retained device/callback failure without changing playback state.
    pub fn check_error(&self) -> SoundResult<()> {
        self.state.check_error()
    }

    /// Requests playback stop.
    /// Wakes blocked producers and returns any retained device/callback failure.
    pub fn stop(&self) -> SoundResult<()> {
        self.state.request_stop()
    }

    /// Returns whether playback has been asked to stop.
    pub(crate) fn is_stop_requested(&self) -> bool {
        self.state.is_stop_requested()
    }
}

/// Cross-thread cancellation without sharing ownership of the platform audio stream.
#[derive(Clone)]
pub(crate) struct PlaybackStopHandle {
    state: Arc<PlaybackState>,
}

impl PlaybackStopHandle {
    /// Sets the stop flag and wakes producers blocked on a full sample queue.
    pub(crate) fn request_stop(&self) -> SoundResult<()> {
        self.state.request_stop()
    }

    /// Returns any device/callback error retained by the stream owner.
    pub(crate) fn check_error(&self) -> SoundResult<()> {
        self.state.check_error()
    }
}

impl Drop for SoundPlayer {
    fn drop(&mut self) {
        let _ = self.state.request_stop();
        let _ = &self.stream;
    }
}

#[cfg(test)]
impl SoundPlayer {
    /// Creates a device-free player whose real callback is driven explicitly by tests.
    pub(crate) fn test_player(block_samples: usize) -> (Self, TestPlaybackControl) {
        let state = Arc::new(PlaybackState::new());
        let queue = state.prepare_queue(block_samples).unwrap();
        let control = TestPlaybackControl {
            state: state.clone(),
            queue: queue.clone(),
        };
        (
            Self {
                config: PlaybackConfig::default(),
                state,
                queue,
                stream: None,
                block_samples,
                prefill_samples: 0,
            },
            control,
        )
    }
}

/// Test-only access to device callbacks and failures without a platform stream.
#[cfg(test)]
pub(crate) struct TestPlaybackControl {
    state: Arc<PlaybackState>,
    queue: Arc<SampleQueue>,
}

#[cfg(test)]
impl TestPlaybackControl {
    /// Drains one stereo callback through production conversion and queue code.
    pub(crate) fn drain(&self, samples: &mut [f32]) {
        super::device::write_output(samples, 2, 2, &self.queue, &self.state);
    }

    /// Injects the same retained failure and wakeup used by a device error callback.
    pub(crate) fn fail(&self, error: SoundError) {
        self.state.report_failure(error);
    }

    /// Returns buffered samples so tests can establish backpressure before shutdown.
    pub(crate) fn queued_samples(&self) -> usize {
        self.queue.queued_samples().unwrap()
    }
}

fn validate_config(config: PlaybackConfig) -> SoundResult<()> {
    if !config.max_queue_seconds.is_finite() || config.max_queue_seconds <= 0.0 {
        return Err(SoundError::invalid_argument(
            "max_queue_seconds must be finite and > 0",
        ));
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
    fn finished_player_rejects_late_blocks_without_losing_its_queued_tail() {
        for prebuffer in [false, true] {
            let (mut player, control) = SoundPlayer::test_player(2);
            if prebuffer {
                player.prefill_samples = 8;
                player.state.begin_prebuffering();
            }
            assert!(player.push_block(&[0.25, -0.25]).unwrap());
            player.finish().unwrap();
            player.finish().unwrap();
            let mut output = [1.0; 4];
            control.drain(&mut output);
            assert_eq!(output, [0.25, -0.25, 0.0, 0.0]);
            assert!(player.state.is_playback_done());
            player.wait_until_finished().unwrap();
            assert_eq!(
                player.push_block(&[0.5, -0.5]).unwrap_err().code(),
                ErrorCode::InvalidState
            );
            assert_eq!(control.queued_samples(), 0);
            player.stop().unwrap();
            assert!(!player.push_block(&[0.0; 2]).unwrap());
            control.fail(SoundError::external("retained device failure"));
            assert!(
                player
                    .push_block(&[0.0; 2])
                    .unwrap_err()
                    .to_string()
                    .contains("retained device failure")
            );
        }
    }

    #[test]
    fn finish_releases_a_blocked_producer_without_draining() {
        let state = Arc::new(PlaybackState::new());
        let queue = state.prepare_queue(1).unwrap();
        queue
            .push_block(&[0.25], state.stop_requested(), state.producer_done())
            .unwrap();
        let producer_state = state.clone();
        let producer_queue = queue.clone();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            let result = producer_queue.push_block(
                &[0.5],
                producer_state.stop_requested(),
                producer_state.producer_done(),
            );
            result_tx.send(result).unwrap();
        });
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        state.mark_producer_done().unwrap();
        let finished = result_rx.recv_timeout(Duration::from_secs(1));
        // Always release and join the owned producer, even against the old bug.
        state.request_stop().unwrap();
        worker.join().unwrap();
        assert_eq!(
            finished.unwrap().unwrap_err().code(),
            ErrorCode::InvalidState
        );
        assert_eq!(queue.queued_samples().unwrap(), 1);
    }

    #[test]
    fn device_failure_wakes_a_full_queue_and_remains_visible_to_the_runtime() {
        let state = Arc::new(PlaybackState::new());
        let queue = state.prepare_queue(1).unwrap();
        queue
            .push_block(&[1.0], state.stop_requested(), state.producer_done())
            .unwrap();
        let handle = PlaybackStopHandle {
            state: state.clone(),
        };
        let worker_state = state.clone();
        let (sent, received) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let accepted = queue
                .push_block(
                    &[2.0],
                    worker_state.stop_requested(),
                    worker_state.producer_done(),
                )
                .unwrap();
            sent.send((accepted, worker_state.check_error())).unwrap();
        });
        state.report_failure(SoundError::external("device disconnected"));
        let (accepted, status) = received.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(!accepted);
        assert!(
            status
                .unwrap_err()
                .to_string()
                .contains("device disconnected")
        );
        assert!(handle.check_error().is_err());
        assert!(handle.request_stop().is_err());
        worker.join().unwrap();
    }

    #[test]
    fn stop_handle_releases_a_producer_without_an_audio_callback() {
        let state = Arc::new(PlaybackState::new());
        let queue = state.prepare_queue(1).unwrap();
        queue
            .push_block(&[1.0], state.stop_requested(), state.producer_done())
            .unwrap();
        let handle = PlaybackStopHandle {
            state: state.clone(),
        };
        let (sent, received) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            sent.send(
                queue
                    .push_block(&[2.0], state.stop_requested(), state.producer_done())
                    .unwrap(),
            )
            .unwrap();
        });
        handle.request_stop().unwrap();
        assert!(!received.recv_timeout(Duration::from_secs(1)).unwrap());
        worker.join().unwrap();
    }

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
