//! Device-free callback control for playback and runtime tests.

use super::*;

impl SoundPlayer {
    /// Creates a device-free player whose real callback is driven explicitly by tests.
    pub(crate) fn test_player(block_samples: usize) -> (Self, TestPlaybackControl) {
        Self::test_player_with_prefill(block_samples, 0)
    }

    /// Creates a player with room for its prebuffer; callbacks remain under test control.
    pub(super) fn test_player_with_prefill(
        block_samples: usize,
        prefill_blocks: usize,
    ) -> (Self, TestPlaybackControl) {
        let state = Arc::new(
            PlaybackState::new(
                block_samples * prefill_blocks.max(1),
                block_samples * prefill_blocks,
            )
            .unwrap(),
        );
        let control = TestPlaybackControl {
            state: state.clone(),
        };
        (
            Self {
                config: PlaybackConfig {
                    prefill_blocks,
                    ..Default::default()
                },
                state,
                _stream: None,
                block_samples,
            },
            control,
        )
    }
}

/// Test-only access to device callbacks and failures without a platform stream.
pub(crate) struct TestPlaybackControl {
    state: Arc<PlaybackState>,
}

impl TestPlaybackControl {
    /// Drains one stereo callback through production conversion and queue code.
    pub(crate) fn drain(&self, samples: &mut [f32]) {
        crate::playback::device::write_output(samples, 2, 2, &self.state);
    }

    /// Injects the same retained failure and wakeup used by a device error callback.
    pub(crate) fn fail(&self, error: SoundError) {
        self.state.report_failure(error);
    }

    /// Returns buffered samples so tests can establish backpressure before shutdown.
    pub(crate) fn queued_samples(&self) -> usize {
        self.state.queue().queued_samples().unwrap()
    }
}
