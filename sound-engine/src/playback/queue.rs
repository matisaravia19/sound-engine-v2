use crate::core::error::{ErrorCode, SoundError, SoundResult};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};

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
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            samples: Mutex::new(VecDeque::with_capacity(capacity)),
            available: Condvar::new(),
            capacity,
        }
    }

    /// Pushes one complete rendered block, blocking until enough capacity exists.
    ///
    /// Returns `false` if playback is stopped before the block is enqueued. The
    /// `stop_requested` flag lets blocked producers wake and exit cleanly.
    pub(super) fn push_block(&self, block: &[f32], stop_requested: &AtomicBool) -> SoundResult<bool> {
        crate::debug_validate!(
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

    /// Drains up to `output.len()` samples for one device callback.
    ///
    /// The callback never waits for producers. Missing samples are reported by
    /// returning a count smaller than `output.len()`.
    pub(super) fn pop_into(&self, output: &mut [f32]) -> SoundResult<usize> {
        let mut samples = self
            .samples
            .lock()
            .map_err(|_| SoundError::poisoned_lock("sample queue lock poisoned"))?;

        let drained = output.len().min(samples.len());
        for sample in output.iter_mut().take(drained) {
            if let Some(next_sample) = samples.pop_front() {
                *sample = next_sample;
            }
        }

        if drained > 0 {
            self.notify_available_space();
        }

        Ok(drained)
    }

    /// Wakes producers waiting for queue capacity.
    pub(super) fn notify_available_space(&self) {
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
    fn queue_respects_capacity_and_stop() -> SoundResult<()> {
        let queue = SampleQueue::new(2);
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
        let queue = SampleQueue::new(4);
        let stop_requested = AtomicBool::new(false);
        assert!(queue.push_block(&[0.1, 0.2, 0.3], &stop_requested)?);

        let mut output = [0.0; 4];
        let drained = queue.pop_into(&mut output)?;

        assert_eq!(drained, 3);
        assert_eq!(output, [0.1, 0.2, 0.3, 0.0]);
        assert_eq!(queue.len()?, 0);
        Ok(())
    }
}
