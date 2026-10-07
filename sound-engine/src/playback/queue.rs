use crate::core::error::{ErrorCode, SoundError, SoundResult};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex, TryLockError};

/// Bounded FIFO of interleaved `f32` playback samples.
///
/// Owned by one playback state. Producers push complete engine blocks; the
/// audio callback drains batches without blocking. Producers
/// may block on the condition variable until the callback frees capacity.
pub(super) struct SampleQueue {
    /// Interleaved engine samples protected against producer/callback access.
    samples: Mutex<VecDeque<f32>>,
    /// Wakes blocked producers when the callback drains queued samples.
    available: Condvar,
    /// Maximum number of samples retained ahead of the device callback.
    capacity: usize,
}

impl SampleQueue {
    /// Creates an empty queue with space for `capacity` interleaved samples.
    /// Capacity overflow and allocator failure are returned before playback starts.
    pub(super) fn new(capacity: usize) -> SoundResult<Self> {
        let mut samples = VecDeque::new();
        samples.try_reserve_exact(capacity).map_err(|error| {
            SoundError::with_source(
                ErrorCode::ResourceAllocationFailed,
                format!("Cannot allocate playback queue for {capacity} samples: {error}"),
                error,
            )
        })?;
        Ok(Self {
            samples: Mutex::new(samples),
            available: Condvar::new(),
            capacity,
        })
    }

    /// Returns queued engine samples while the producer is filling its prebuffer.
    pub(super) fn queued_samples(&self) -> SoundResult<usize> {
        Ok(self
            .samples
            .lock()
            .map_err(|_| SoundError::poisoned_lock("sample queue lock poisoned"))?
            .len())
    }

    /// Pushes one complete rendered block, blocking until enough capacity exists.
    ///
    /// Returns `false` on cancellation and an invalid-state error after finish.
    /// Both flags are checked under the queue lock; their publishers must call
    /// `notify_available_space` so a waiting producer cannot miss a transition.
    pub(super) fn push_block(
        &self,
        block: &[f32],
        stop_requested: &AtomicBool,
        producer_done: &AtomicBool,
    ) -> SoundResult<bool> {
        crate::validate!(
            block.len() <= self.capacity,
            ErrorCode::InvalidArgument,
            "sample block length {} exceeds queue capacity {}",
            block.len(),
            self.capacity
        );

        let mut samples = self
            .samples
            .lock()
            .map_err(|_| SoundError::poisoned_lock("sample queue lock poisoned"))?;
        loop {
            if stop_requested.load(Ordering::Acquire) {
                return Ok(false);
            }
            if producer_done.load(Ordering::Acquire) {
                return Err(SoundError::invalid_state(
                    "Cannot push audio after playback was finished",
                ));
            }
            if self.capacity.saturating_sub(samples.len()) >= block.len() {
                break;
            }

            samples = self
                .available
                .wait(samples)
                .map_err(|_| SoundError::poisoned_lock("sample queue lock poisoned"))?;
        }

        samples.extend(block);
        Ok(true)
    }

    /// Drains up to `limit` samples for one device callback without waiting.
    ///
    /// `None` means a producer holds the queue; it does not imply exhaustion.
    /// The consumer runs under the queue lock and must not block or allocate.
    pub(super) fn drain_into(
        &self,
        limit: usize,
        mut consume: impl FnMut(usize, f32),
    ) -> SoundResult<Option<usize>> {
        let mut samples = match self.samples.try_lock() {
            Ok(samples) => samples,
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(TryLockError::Poisoned(_)) => {
                return Err(SoundError::poisoned_lock("sample queue lock poisoned"));
            }
        };

        let drained = limit.min(samples.len());
        for index in 0..drained {
            if let Some(next_sample) = samples.pop_front() {
                consume(index, next_sample);
            }
        }

        if drained > 0 {
            self.available.notify_all();
        }

        Ok(Some(drained))
    }

    /// Wakes producers after queue capacity or lifecycle flags change.
    pub(super) fn notify_available_space(&self) {
        // Pair lifecycle notifications with the predicate mutex so a producer
        // cannot miss completion/cancellation between checking flags and waiting.
        let _samples = self
            .samples
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        self.available.notify_all();
    }
}

#[cfg(test)]
#[path = "../../tests/unit/playback/queue.rs"]
mod tests;
