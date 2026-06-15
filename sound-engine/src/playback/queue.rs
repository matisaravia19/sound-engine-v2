use crate::core::error::{SoundError, SoundResult};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};

pub(super) struct SampleQueue {
    samples: Mutex<VecDeque<f32>>,
    available: Condvar,
    capacity: usize,
}

impl SampleQueue {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            samples: Mutex::new(VecDeque::with_capacity(capacity)),
            available: Condvar::new(),
            capacity,
        }
    }

    pub(super) fn push_block(&self, block: &[f32], stop_requested: &AtomicBool) -> SoundResult<bool> {
        if block.len() > self.capacity {
            return Err(SoundError::invalid_argument(format!(
                "sample block length {} exceeds queue capacity {}",
                block.len(),
                self.capacity
            )));
        }

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

    pub(super) fn notify_available_space(&self) {
        self.available.notify_all();
    }

    #[cfg(test)]
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
