use super::*;
use std::sync::atomic::{AtomicBool, Ordering};

#[test]
fn idle_receiver_observes_stop_without_a_command() {
    let (_sender, receiver) = mpsc::channel::<()>();
    let running = Arc::new(AtomicBool::new(true));
    let worker_running = running.clone();
    let (result_tx, result_rx) = mpsc::channel();
    let worker = thread::spawn(move || {
        let result = receive_while_running(&receiver, || worker_running.load(Ordering::Acquire));
        result_tx.send(result).unwrap();
    });
    running.store(false, Ordering::Release);
    assert_eq!(
        result_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap(),
        None
    );
    worker.join().unwrap();
}
