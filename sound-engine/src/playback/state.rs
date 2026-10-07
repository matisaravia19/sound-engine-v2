use crate::core::error::{SoundError, SoundResult};
use crate::playback::queue::SampleQueue;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Owns one stream's sample queue and shared producer/callback lifecycle.
///
/// Construction installs the queue and prebuffer policy before the state can be
/// shared through `Arc`. Atomic flags remain callback-safe; the first failure
/// and cancellation cannot be reset for another stream.
pub(super) struct PlaybackState {
    /// Queue retained through callback completion and producer cancellation.
    queue: SampleQueue,
    /// Initial queue occupancy required before device consumption begins.
    prefill_samples: usize,
    /// First stream/callback error, retained for every subsequent caller.
    failure: OnceLock<SoundError>,
    /// Set by the producer once no more blocks will be pushed.
    producer_done: AtomicBool,
    /// Set once the callback has drained final samples or stop was requested.
    playback_done: AtomicBool,
    /// Set when producers and the callback should stop as soon as possible.
    stop_requested: AtomicBool,
    /// Producer expectation distinguishes active starvation from an idle engine.
    expects_audio: AtomicBool,
    /// Startup silence is excluded until the callback has received audio once.
    received_audio: AtomicBool,
    /// Callback count with missing frames during active playback.
    underruns: AtomicU64,
    /// Missing engine frames during active playback, independent of device channels.
    missing_frames: AtomicU64,
    /// Monotonic origin for underrun diagnostics; timing never affects playback.
    created_at: std::time::Instant,
    /// First active underrun in nanoseconds since creation, or u64::MAX if none.
    first_underrun_ns: AtomicU64,
    /// Most recent active underrun on the same clock as first_underrun_ns.
    last_underrun_ns: AtomicU64,
    /// Callback gate while the producer fills the configured initial prebuffer.
    ready: AtomicBool,
}

impl PlaybackState {
    /// Allocates a stream's queue and installs its initial prebuffer threshold.
    pub(super) fn new(capacity: usize, prefill_samples: usize) -> SoundResult<Self> {
        if prefill_samples > capacity {
            return Err(SoundError::invalid_argument(
                "Playback prebuffer exceeds queue capacity",
            ));
        }
        Ok(Self {
            queue: SampleQueue::new(capacity)?,
            prefill_samples,
            failure: OnceLock::new(),
            producer_done: AtomicBool::new(false),
            playback_done: AtomicBool::new(false),
            stop_requested: AtomicBool::new(false),
            expects_audio: AtomicBool::new(true),
            received_audio: AtomicBool::new(false),
            underruns: AtomicU64::new(0),
            missing_frames: AtomicU64::new(0),
            created_at: std::time::Instant::now(),
            first_underrun_ns: AtomicU64::new(u64::MAX),
            last_underrun_ns: AtomicU64::new(0),
            ready: AtomicBool::new(prefill_samples == 0),
        })
    }

    /// Borrows this stream's queue for callback draining and occupancy queries.
    pub(super) fn queue(&self) -> &SampleQueue {
        &self.queue
    }

    /// Enqueues a complete block and releases the initial prebuffer when full.
    /// Failure takes precedence over cancellation, including while waiting for space.
    pub(super) fn push_block(&self, block: &[f32]) -> SoundResult<bool> {
        self.check_error()?;
        let accepted = self
            .queue
            .push_block(block, &self.stop_requested, &self.producer_done);
        // Failure can arrive while this producer waits on a full queue.
        self.check_error()?;
        let accepted = accepted?;
        if accepted && !self.ready_to_play() && self.queue.queued_samples()? >= self.prefill_samples
        {
            self.release_prebuffer();
        }
        Ok(accepted)
    }

    /// Retains the first callback failure before waking stopped producers.
    pub(super) fn report_failure(&self, error: SoundError) {
        let _ = self.failure.set(error);
        let _ = self.request_stop();
    }

    /// Returns the original failure without consuming it; healthy checks do not lock.
    pub(super) fn check_error(&self) -> SoundResult<()> {
        self.failure.get().cloned().map_or(Ok(()), Err)
    }

    /// Observes producer completion before a callback drains the final queued samples.
    pub(super) fn is_producer_done(&self) -> bool {
        self.producer_done.load(Ordering::Acquire)
    }

    /// Releases the callback after the producer fills its prebuffer.
    pub(super) fn release_prebuffer(&self) {
        self.ready.store(true, Ordering::Release);
    }

    /// Returns whether the device callback may consume queued frames.
    pub(super) fn ready_to_play(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    /// Marks whether an active producer should currently keep the queue fed.
    pub(super) fn set_expects_audio(&self, active: bool) {
        self.expects_audio.store(active, Ordering::Release);
    }

    /// Records active callback starvation without allocation, locks, or logging.
    /// Explicit-stop silence is excluded even while the device stream is alive.
    pub(super) fn record_callback(&self, received_frames: usize, requested_frames: usize) {
        let started = self.received_audio.load(Ordering::Relaxed);
        if received_frames > 0 {
            self.received_audio.store(true, Ordering::Relaxed);
        }
        if started
            && self.expects_audio.load(Ordering::Acquire)
            && !self.producer_done.load(Ordering::Acquire)
            && !self.stop_requested.load(Ordering::Acquire)
            && received_frames < requested_frames
        {
            // Read the clock only on starvation. The callback still performs
            // no allocation, lock acquisition, or diagnostic file I/O.
            let elapsed = self
                .created_at
                .elapsed()
                .as_nanos()
                .min(u64::MAX as u128 - 1) as u64;
            let _ = self.first_underrun_ns.compare_exchange(
                u64::MAX,
                elapsed,
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
            self.last_underrun_ns.store(elapsed, Ordering::Relaxed);
            self.underruns.fetch_add(1, Ordering::Relaxed);
            self.missing_frames.fetch_add(
                (requested_frames - received_frames) as u64,
                Ordering::Relaxed,
            );
        }
    }

    /// Returns active-playback underrun callbacks and missing frame totals.
    pub(super) fn starvation_counts(&self) -> (u64, u64) {
        (
            self.underruns.load(Ordering::Relaxed),
            self.missing_frames.load(Ordering::Relaxed),
        )
    }

    /// Returns first/last active underrun seconds since this state was created.
    /// A live snapshot may lag a concurrent callback; read after stopping for a final report.
    pub(super) fn starvation_time_range_seconds(&self) -> Option<(f64, f64)> {
        let first = self.first_underrun_ns.load(Ordering::Relaxed);
        (first != u64::MAX).then(|| {
            (
                first as f64 * 1e-9,
                self.last_underrun_ns.load(Ordering::Relaxed) as f64 * 1e-9,
            )
        })
    }

    /// Marks the producer as finished and wakes any blocked queue users.
    pub(super) fn mark_producer_done(&self) -> SoundResult<()> {
        self.producer_done.store(true, Ordering::Release);
        // Short sounds may finish before reaching the prefill threshold.
        self.release_prebuffer();
        // Taking the queue mutex also finishes any enqueue that had already
        // accepted its block, before this method returns to the producer.
        self.queue.notify_available_space();
        self.check_error()
    }

    /// Marks playback as drained from the device callback side.
    pub(super) fn mark_playback_done(&self) {
        self.playback_done.store(true, Ordering::Release);
    }

    /// Requests immediate playback stop and wakes blocked producers.
    pub(super) fn request_stop(&self) -> SoundResult<()> {
        self.stop_requested.store(true, Ordering::Release);
        self.playback_done.store(true, Ordering::Release);
        self.queue.notify_available_space();
        self.check_error()
    }

    /// Returns whether playback has completed or was stopped.
    pub(super) fn is_playback_done(&self) -> bool {
        self.playback_done.load(Ordering::Acquire)
    }

    /// Returns whether playback has been asked to stop.
    pub(super) fn is_stop_requested(&self) -> bool {
        self.stop_requested.load(Ordering::Acquire)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/playback/state.rs"]
mod tests;
