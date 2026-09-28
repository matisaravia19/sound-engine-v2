use crate::core::error::ErrorCode;
use crate::core::error::{SoundError, SoundResult};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex, TryLockError};

/// Bounded FIFO of interleaved `f32` playback samples.
///
/// Producers hold shared ownership through `Arc<SampleQueue>` and push complete
/// engine blocks. The audio callback drains batches without blocking; producers
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
    /// Returns `false` if playback is stopped before the block is enqueued. The
    /// `stop_requested` flag lets blocked producers wake and exit cleanly.
    pub(super) fn push_block(
        &self,
        block: &[f32],
        stop_requested: &AtomicBool,
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
        while self.capacity.saturating_sub(samples.len()) < block.len() {
            if stop_requested.load(Ordering::Acquire) {
                return Ok(false);
            }

            samples = self
                .available
                .wait(samples)
                .map_err(|_| SoundError::poisoned_lock("sample queue lock poisoned"))?;
        }

        if stop_requested.load(Ordering::Acquire) {
            return Ok(false);
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

    /// Wakes producers waiting for queue capacity.
    pub(super) fn notify_available_space(&self) {
        // Pair external stop notifications with the predicate mutex so a
        // producer cannot miss shutdown between checking the flag and waiting.
        let _samples = self
            .samples
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        self.available.notify_all();
    }

    #[cfg(test)]
    /// Returns the current queued sample count for tests.
    fn len(&self) -> SoundResult<usize> {
        let samples = self
            .samples
            .lock()
            .map_err(|_| SoundError::poisoned_lock("sample queue lock poisoned"))?;
        Ok(samples.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oversized_blocks_are_rejected_even_in_release_builds() {
        let queue = SampleQueue::new(1).unwrap();
        assert!(
            queue
                .push_block(&[0.0, 0.0], &AtomicBool::new(false))
                .is_err()
        );
    }

    #[test]
    fn stop_wakes_a_producer_waiting_on_a_full_queue() {
        use std::sync::{Arc, mpsc};
        let queue = Arc::new(SampleQueue::new(1).unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        queue.push_block(&[0.0], &stop).unwrap();
        let (sender, receiver) = mpsc::channel();
        let worker_queue = queue.clone();
        let worker_stop = stop.clone();
        let worker = std::thread::spawn(move || {
            sender
                .send(worker_queue.push_block(&[1.0], &worker_stop).unwrap())
                .unwrap();
        });
        stop.store(true, Ordering::Release);
        queue.notify_available_space();
        assert!(
            !receiver
                .recv_timeout(std::time::Duration::from_secs(2))
                .unwrap()
        );
        worker.join().unwrap();
    }

    #[test]
    fn queue_respects_capacity_and_stop() -> SoundResult<()> {
        let queue = SampleQueue::new(2)?;
        let stop_requested = AtomicBool::new(false);

        assert!(queue.push_block(&[0.1, 0.2], &stop_requested)?);
        assert_eq!(queue.len()?, 2);

        stop_requested.store(true, Ordering::Release);
        assert!(!queue.push_block(&[0.3], &stop_requested)?);
        assert_eq!(queue.len()?, 2);
        Ok(())
    }

    #[test]
    fn queue_drains_batch_with_one_call() -> SoundResult<()> {
        let queue = SampleQueue::new(4)?;
        let stop_requested = AtomicBool::new(false);
        assert!(queue.push_block(&[0.1, 0.2, 0.3], &stop_requested)?);

        let mut output = [0.0; 4];
        let drained = queue.drain_into(output.len(), |index, sample| output[index] = sample)?;

        assert_eq!(drained, Some(3));
        assert_eq!(output, [0.1, 0.2, 0.3, 0.0]);
        assert_eq!(queue.len()?, 0);
        Ok(())
    }

    #[test]
    fn busy_queue_preserves_samples_without_waiting() -> SoundResult<()> {
        let queue = SampleQueue::new(2)?;
        queue.push_block(&[0.25, -0.25], &AtomicBool::new(false))?;
        let guard = queue.samples.lock().unwrap();
        assert_eq!(
            queue.drain_into(2, |_, _| panic!("busy queue drained"))?,
            None
        );
        drop(guard);
        let mut output = [0.0; 2];
        assert_eq!(
            queue.drain_into(2, |index, sample| output[index] = sample)?,
            Some(2)
        );
        assert_eq!(output, [0.25, -0.25]);
        Ok(())
    }
}
