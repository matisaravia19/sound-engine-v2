use super::*;

#[test]
fn oversized_blocks_are_rejected_even_in_release_builds() {
    let queue = SampleQueue::new(1).unwrap();
    assert!(
        queue
            .push_block(
                &[0.0, 0.0],
                &AtomicBool::new(false),
                &AtomicBool::new(false)
            )
            .is_err()
    );
}

#[test]
fn stop_wakes_a_producer_waiting_on_a_full_queue() {
    use std::sync::{Arc, mpsc};
    let queue = Arc::new(SampleQueue::new(1).unwrap());
    let stop = Arc::new(AtomicBool::new(false));
    queue
        .push_block(&[0.0], &stop, &AtomicBool::new(false))
        .unwrap();
    let (sender, receiver) = mpsc::channel();
    let worker_queue = queue.clone();
    let worker_stop = stop.clone();
    let worker = std::thread::spawn(move || {
        sender
            .send(
                worker_queue
                    .push_block(&[1.0], &worker_stop, &AtomicBool::new(false))
                    .unwrap(),
            )
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

    assert!(queue.push_block(&[0.1, 0.2], &stop_requested, &AtomicBool::new(false))?);
    assert_eq!(queue.len()?, 2);

    stop_requested.store(true, Ordering::Release);
    assert!(!queue.push_block(&[0.3], &stop_requested, &AtomicBool::new(false))?);
    assert_eq!(queue.len()?, 2);
    Ok(())
}

#[test]
fn queue_drains_batch_with_one_call() -> SoundResult<()> {
    let queue = SampleQueue::new(4)?;
    let stop_requested = AtomicBool::new(false);
    assert!(queue.push_block(&[0.1, 0.2, 0.3], &stop_requested, &AtomicBool::new(false))?);

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
    queue.push_block(
        &[0.25, -0.25],
        &AtomicBool::new(false),
        &AtomicBool::new(false),
    )?;
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

impl SampleQueue {
    /// Returns the current queued sample count for tests.
    fn len(&self) -> SoundResult<usize> {
        let samples = self
            .samples
            .lock()
            .map_err(|_| SoundError::poisoned_lock("sample queue lock poisoned"))?;
        Ok(samples.len())
    }
}
