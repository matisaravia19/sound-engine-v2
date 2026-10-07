use super::*;
use std::rc::Rc;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct Owner(Arc<AtomicUsize>);
impl Drop for Owner {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn failed_or_panicking_initialization_returns_the_live_owner() {
    for panic in [false, true] {
        let drops = Arc::new(AtomicUsize::new(0));
        let result = start(
            "startup-failure-test",
            Owner(drops.clone()),
            move || -> SoundResult<((), ())> {
                if panic {
                    panic!("simulated initialization panic");
                }
                Err(SoundError::backend_unavailable(
                    "simulated initialization failure",
                ))
            },
            |_, _| panic!("owner must not be transferred"),
        );
        let (owner, error) = result.err().unwrap();
        assert!(error.to_string().contains(if panic {
            "panicked"
        } else {
            "simulated initialization failure"
        }));
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(owner);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn non_send_state_stays_on_worker_and_owner_returns_after_run() {
    let drops = Arc::new(AtomicUsize::new(0));
    let (handle, worker) = start(
        "startup-success-test",
        Owner(drops.clone()),
        || Ok((Rc::new(7), 42)),
        |owner, state| {
            assert_eq!(*state, 7);
            (owner, Ok(()))
        },
    )
    .unwrap_or_else(|(_, error)| panic!("{error}"));
    assert_eq!(handle, 42);
    let (owner, result) = worker.join().unwrap().unwrap();
    result.unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(owner);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}
