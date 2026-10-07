use crate::core::error::{SoundError, SoundResult};
use crate::playback::queue::SampleQueue;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

/// Shared playback lifecycle flags used by producers and the device callback.
///
/// The state is shared through `Arc<PlaybackState>`. Atomic flags are used for
/// callback-safe status checks. Each state belongs to one stream; its queue and
/// first failure are installed once and remain available through shutdown.
pub(super) struct PlaybackState {
    /// Queue currently associated with this playback run, if one was prepared.
    queue: OnceLock<Arc<SampleQueue>>,
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
    /// Creates an idle playback state with no prepared queue.
    pub(super) fn new() -> Self {
        Self {
            queue: OnceLock::new(),
            failure: OnceLock::new(),
            producer_done: AtomicBool::new(true),
            playback_done: AtomicBool::new(true),
            stop_requested: AtomicBool::new(false),
            expects_audio: AtomicBool::new(true),
            received_audio: AtomicBool::new(false),
            underruns: AtomicU64::new(0),
            missing_frames: AtomicU64::new(0),
            created_at: std::time::Instant::now(),
            first_underrun_ns: AtomicU64::new(u64::MAX),
            last_underrun_ns: AtomicU64::new(0),
            ready: AtomicBool::new(true),
        }
    }

    /// Installs this stream's queue before callbacks or producers can start.
    /// Reusing state for another stream is rejected so failures cannot be cleared.
    pub(super) fn prepare_queue(&self, capacity: usize) -> SoundResult<Arc<SampleQueue>> {
        self.check_error()?;
        if self.queue.get().is_some() || self.is_stop_requested() {
            return Err(SoundError::invalid_state(
                "Playback state cannot be restarted",
            ));
        }
        let queue = Arc::new(SampleQueue::new(capacity)?);
        self.queue
            .set(queue.clone())
            .map_err(|_| SoundError::invalid_state("Playback queue is already prepared"))?;
        self.producer_done.store(false, Ordering::Release);
        self.playback_done.store(false, Ordering::Release);
        self.stop_requested.store(false, Ordering::Release);
        Ok(queue)
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

    /// Returns the completion flag observed by queue producers and the callback.
    pub(super) fn producer_done(&self) -> &AtomicBool {
        &self.producer_done
    }

    /// Gates callback consumption until enough initial blocks have been queued.
    pub(super) fn begin_prebuffering(&self) {
        self.ready.store(false, Ordering::Release);
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

    /// Returns the stop flag observed by producers while they wait for capacity.
    pub(super) fn stop_requested(&self) -> &AtomicBool {
        &self.stop_requested
    }

    /// Marks the producer as finished and wakes any blocked queue users.
    pub(super) fn mark_producer_done(&self) -> SoundResult<()> {
        self.producer_done.store(true, Ordering::Release);
        // Short sounds may finish before reaching the prefill threshold.
        self.release_prebuffer();
        // Taking the queue mutex also finishes any enqueue that had already
        // accepted its block, before this method returns to the producer.
        self.notify_all();
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
        self.notify_all();
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

    /// Notifies the active queue, if any, so waiting producers can re-check flags.
    fn notify_all(&self) {
        if let Some(queue) = self.queue.get() {
            queue.notify_available_space();
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/playback/state.rs"]
mod tests;
