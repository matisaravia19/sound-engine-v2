use crate::core::error::{SoundError, SoundResult};
use crate::playback::queue::SampleQueue;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Shared playback lifecycle flags used by producers and the device callback.
///
/// The state is shared through `Arc<PlaybackState>`. Atomic flags are used for
/// callback-safe status checks, while the optional queue handle is only used to
/// wake blocked producers during finish/stop transitions.
pub(super) struct PlaybackState {
    /// Queue currently associated with this playback run, if one was prepared.
    queue: Mutex<Option<Arc<SampleQueue>>>,
    /// Set by the producer once no more blocks will be pushed.
    producer_done: AtomicBool,
    /// Set once the callback has drained final samples or stop was requested.
    playback_done: AtomicBool,
    /// Set when producers and the callback should stop as soon as possible.
    stop_requested: AtomicBool,
}

impl PlaybackState {
    /// Creates an idle playback state with no prepared queue.
    pub(super) fn new() -> Self {
        Self {
            queue: Mutex::new(None),
            producer_done: AtomicBool::new(true),
            playback_done: AtomicBool::new(true),
            stop_requested: AtomicBool::new(false),
        }
    }

    /// Resets lifecycle flags and installs a fresh queue for a playback run.
    pub(super) fn prepare_queue(&self, capacity: usize) -> SoundResult<Arc<SampleQueue>> {
        self.producer_done.store(false, Ordering::Release);
        self.playback_done.store(false, Ordering::Release);
        self.stop_requested.store(false, Ordering::Release);

        let queue = Arc::new(SampleQueue::new(capacity));
        *self
            .queue
            .lock()
            .map_err(|_| SoundError::poisoned_lock("playback state lock poisoned"))? = Some(queue.clone());
        Ok(queue)
    }

    /// Returns the producer-completion flag observed by the callback.
    pub(super) fn producer_done(&self) -> &AtomicBool {
        &self.producer_done
    }

    /// Returns the stop flag observed by producers while they wait for capacity.
    pub(super) fn stop_requested(&self) -> &AtomicBool {
        &self.stop_requested
    }

    /// Marks the producer as finished and wakes any blocked queue users.
    pub(super) fn mark_producer_done(&self) -> SoundResult<()> {
        self.producer_done.store(true, Ordering::Release);
        self.notify_all()
    }

    /// Marks playback as drained from the device callback side.
    pub(super) fn mark_playback_done(&self) {
        self.playback_done.store(true, Ordering::Release);
    }

    /// Requests immediate playback stop and wakes blocked producers.
    pub(super) fn request_stop(&self) -> SoundResult<()> {
        self.stop_requested.store(true, Ordering::Release);
        self.playback_done.store(true, Ordering::Release);
        self.notify_all()
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
    fn notify_all(&self) -> SoundResult<()> {
        if let Some(queue) = self
            .queue
            .lock()
            .map_err(|_| SoundError::poisoned_lock("playback state lock poisoned"))?
            .as_ref()
        {
            queue.notify_available_space();
        }
        Ok(())
    }
}
