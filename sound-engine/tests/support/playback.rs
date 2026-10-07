//! Device-free callback control for playback and runtime tests.

use super::*;

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
pub(crate) struct TestPlaybackControl {
    state: Arc<PlaybackState>,
    queue: Arc<SampleQueue>,
}

impl TestPlaybackControl {
    /// Drains one stereo callback through production conversion and queue code.
    pub(crate) fn drain(&self, samples: &mut [f32]) {
        crate::playback::device::write_output(samples, 2, 2, &self.queue, &self.state);
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
