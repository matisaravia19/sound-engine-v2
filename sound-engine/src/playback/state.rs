use crate::core::error::{SoundError, SoundResult};
use crate::playback::queue::SampleQueue;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

pub(super) struct PlaybackState {
    queue: Mutex<Option<Arc<SampleQueue>>>,
    producer_done: AtomicBool,
    playback_done: AtomicBool,
    stop_requested: AtomicBool,
}

impl PlaybackState {
    pub(super) fn new() -> Self {
        Self {
            queue: Mutex::new(None),
            producer_done: AtomicBool::new(true),
            playback_done: AtomicBool::new(true),
            stop_requested: AtomicBool::new(false),
        }
    }

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

    pub(super) fn producer_done(&self) -> &AtomicBool {
        &self.producer_done
    }

    pub(super) fn stop_requested(&self) -> &AtomicBool {
        &self.stop_requested
    }

    pub(super) fn mark_producer_done(&self) -> SoundResult<()> {
        self.producer_done.store(true, Ordering::Release);
        self.notify_all()
    }

    pub(super) fn mark_playback_done(&self) {
        self.playback_done.store(true, Ordering::Release);
    }

    pub(super) fn request_stop(&self) -> SoundResult<()> {
        self.stop_requested.store(true, Ordering::Release);
        self.playback_done.store(true, Ordering::Release);
        self.notify_all()
    }

    pub(super) fn is_playback_done(&self) -> bool {
        self.playback_done.load(Ordering::Acquire)
    }

    pub(super) fn is_stop_requested(&self) -> bool {
        self.stop_requested.load(Ordering::Acquire)
    }

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
