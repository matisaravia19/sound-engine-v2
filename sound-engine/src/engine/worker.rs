//! Startup handoff that keeps the caller's owner recoverable until initialization succeeds.

use crate::core::error::{SoundError, SoundResult};
use std::sync::mpsc;
use std::thread::{self, JoinHandle};

/// Worker result; initialization can fail before an owner is transferred.
pub(super) type OwnerThread<T> = JoinHandle<Option<(T, SoundResult<()>)>>;

/// Initializes thread-local state before transferring `owner` to the worker.
/// Every startup failure joins the worker and returns the original owner.
pub(super) fn start<T, State, Handle>(
    name: &str,
    owner: T,
    initialize: impl FnOnce() -> SoundResult<(State, Handle)> + Send + 'static,
    run: impl FnOnce(T, State) -> (T, SoundResult<()>) + Send + 'static,
) -> Result<(Handle, OwnerThread<T>), (T, SoundError)>
where
    T: Send + 'static,
    State: 'static,
    Handle: Send + 'static,
{
    let (owner_tx, owner_rx) = mpsc::sync_channel(1);
    let (ready_tx, ready_rx) = mpsc::sync_channel(1);
    let worker = match thread::Builder::new().name(name.into()).spawn(move || {
        let (state, handle) = match initialize() {
            Ok(initialized) => initialized,
            Err(error) => {
                let _ = ready_tx.send(Err(error));
                return None;
            }
        };
        if ready_tx.send(Ok(handle)).is_err() {
            return None;
        }
        drop(ready_tx);
        let owner = owner_rx.recv().ok()?;
        Some(run(owner, state))
    }) {
        Ok(worker) => worker,
        Err(error) => {
            return Err((
                owner,
                SoundError::external(format!("Cannot start {name}: {error}")),
            ));
        }
    };
    let handle = match ready_rx.recv() {
        Ok(Ok(handle)) => handle,
        result => {
            // Close the handoff before joining, so an initialized worker cannot
            // remain asleep waiting for ownership during startup rollback.
            drop(owner_tx);
            let joined = worker.join();
            let error = match result {
                Ok(Err(error)) => error,
                _ if joined.is_err() => {
                    SoundError::external("Playback worker panicked during initialization")
                }
                _ => SoundError::external("Playback worker stopped during initialization"),
            };
            return Err((owner, error));
        }
    };
    if let Err(error) = owner_tx.send(owner) {
        let _ = worker.join();
        return Err((
            error.0,
            SoundError::external("Playback worker stopped before ownership transfer"),
        ));
    }
    Ok((handle, worker))
}

#[cfg(test)]
#[path = "../../tests/unit/engine/worker.rs"]
mod tests;
