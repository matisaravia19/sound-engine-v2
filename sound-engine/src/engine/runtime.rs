//! Background playback runtime for the public engine facade.

use super::diagnostics::RuntimeDiagnostics;
use super::{EngineCore, ListenerPose};
use crate::core::error::{SoundError, SoundResult};
use crate::playback::{PlaybackConfig, PlaybackStopHandle, SoundPlayer};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;

/// Operation and reply performed on the core's owning thread between audio blocks.
type CoreRequest = Box<dyn FnOnce(&mut EngineCore, &RuntimeStatus) + Send>;

enum EngineCommand {
    Run(CoreRequest),
    Shutdown,
}

impl EngineCommand {
    fn request<T: Send + 'static>(
        operation: impl FnOnce(&mut EngineCore) -> SoundResult<T> + Send + 'static,
    ) -> (Self, Receiver<SoundResult<T>>) {
        let (reply, response) = mpsc::sync_channel(1);
        let command = Self::Run(Box::new(move |core, status| {
            let reply = reply;
            // Publish a panic before dropping the reply channel during unwinding.
            let _panic_status = PanicStatusGuard(status);
            let result = operation(core);
            // An acknowledged operation must expose its updated voice count.
            status.publish(core);
            let _ = reply.send(result);
        }));
        (command, response)
    }
}

/// Handle for the background thread that owns [`EngineCore`] while playback is running.
/// Dropping it wakes blocked playback and joins the worker before releasing the core.
pub(super) struct EngineRuntime {
    /// Command sender used by the public facade to request worker-thread work.
    commands: Sender<EngineCommand>,
    /// Join handle for recovering direct engine ownership when playback stops.
    worker: Option<super::worker::OwnerThread<Box<EngineCore>>>,
    /// Capacity-one pose mailbox; stale game frames are replaced before rendering.
    pending_listener: Arc<Mutex<Option<ListenerPose>>>,
    /// Wakes a full-queue producer before shutdown waits for its command loop.
    stop_handle: PlaybackStopHandle,
    /// Worker-published voice count and first asynchronous engine failure.
    status: Arc<RuntimeStatus>,
}

/// Startup failure before ownership transfer; the caller retains a usable core.
pub(super) struct RuntimeStartError {
    /// Engine that was never handed to a running playback loop.
    pub(super) core: Box<EngineCore>,
    /// Initialization or thread-creation failure.
    pub(super) error: SoundError,
}

struct RuntimeStatus {
    active_voices: AtomicUsize,
    failure: OnceLock<SoundError>,
}

impl RuntimeStatus {
    fn new(active_voices: usize) -> Self {
        Self {
            active_voices: AtomicUsize::new(active_voices),
            failure: OnceLock::new(),
        }
    }

    fn publish(&self, core: &EngineCore) {
        self.active_voices
            .store(core.active_voice_count(), Ordering::Release);
    }

    fn check_error(&self) -> SoundResult<()> {
        self.failure.get().cloned().map_or(Ok(()), Err)
    }
}

struct PanicStatusGuard<'a>(&'a RuntimeStatus);

impl Drop for PanicStatusGuard<'_> {
    fn drop(&mut self) {
        if thread::panicking() {
            // A panicking worker destroys its owner instead of returning it.
            self.0.active_voices.store(0, Ordering::Release);
            let _ = self
                .0
                .failure
                .set(SoundError::external("Playback worker panicked"));
        }
    }
}

impl EngineRuntime {
    /// Starts the playback worker and moves direct engine ownership into it.
    pub(super) fn start(
        core: Box<EngineCore>,
        playback_config: PlaybackConfig,
    ) -> Result<Self, RuntimeStartError> {
        let sample_rate = core.sample_rate();
        let block_size = core.block_size();
        let output_channels = core.output_channels();
        Self::start_with_player(core, move || {
            SoundPlayer::new(sample_rate, output_channels, block_size, playback_config)
        })
    }

    fn start_with_player(
        core: Box<EngineCore>,
        initialize: impl FnOnce() -> SoundResult<SoundPlayer> + Send + 'static,
    ) -> Result<Self, RuntimeStartError> {
        let (sender, receiver) = mpsc::channel();
        let pending_listener = Arc::new(Mutex::new(None));
        let worker_listener = pending_listener.clone();
        let status = Arc::new(RuntimeStatus::new(core.active_voice_count()));
        let worker_status = status.clone();
        let (stop_handle, worker) = super::worker::start(
            "sound-engine-playback",
            core,
            move || {
                let player = initialize()?;
                let stop_handle = player.stop_handle();
                Ok((player, stop_handle))
            },
            move |mut core, player| {
                let _panic_status = PanicStatusGuard(&worker_status);
                let result = runtime_loop(
                    &mut core,
                    &player,
                    &receiver,
                    &worker_listener,
                    &worker_status,
                );
                worker_status.publish(&core);
                if let Err(error) = &result {
                    let _ = worker_status.failure.set(error.clone());
                }
                // Publish the cause before disconnecting pending command replies.
                // Otherwise a waiting caller can observe only a generic channel error.
                drop(receiver);
                (core, result)
            },
        )
        .map_err(|(core, error)| RuntimeStartError {
            core,
            error,
        })?;

        Ok(Self {
            commands: sender,
            worker: Some(worker),
            pending_listener,
            stop_handle,
            status,
        })
    }

    /// Clears older coalesced poses and waits for this pose to reach the worker.
    pub(super) fn set_listener_pose(&self, listener: ListenerPose) -> SoundResult<()> {
        self.pending_listener
            .lock()
            .map_err(|_| SoundError::poisoned_lock("listener mailbox"))?
            .take();
        self.request(move |core| core.set_listener_pose(listener))
    }

    /// Replaces an unapplied listener pose without waiting for the audio worker.
    pub(super) fn update_listener_pose(&self, listener: ListenerPose) -> SoundResult<()> {
        self.check_error()?;
        if self
            .worker
            .as_ref()
            .is_none_or(|worker| worker.is_finished())
        {
            return Err(SoundError::invalid_state(
                "sound player worker is not running",
            ));
        }
        let listener = crate::acoustics::canonicalize_listener(listener)?;
        *self
            .pending_listener
            .lock()
            .map_err(|_| SoundError::poisoned_lock("listener mailbox"))? = Some(listener);
        Ok(())
    }

    /// Joins the worker, returning the recovered core and its playback result.
    /// The outer error means the worker could not return ownership.
    pub(super) fn stop(mut self) -> SoundResult<(Box<EngineCore>, SoundResult<()>)> {
        // A command cannot wake a worker blocked in push_block. Wake its queue
        // first; the command separately wakes an idle receiver.
        let stopped = self.stop_handle.request_stop();
        let _ = self.commands.send(EngineCommand::Shutdown);
        let worker = self
            .worker
            .take()
            .ok_or_else(|| SoundError::invalid_state("sound player worker was already stopped"))?;
        match worker.join() {
            Ok(Some((core, result))) => Ok((
                core,
                result
                    .and(stopped)
                    .and_then(|()| self.stop_handle.check_error()),
            )),
            Ok(None) => Err(SoundError::external(
                "sound player worker stopped before ownership transfer",
            )),
            Err(_) => Err(self.stopped_error("sound player worker thread panicked")),
        }
    }

    /// Reports worker liveness even when an asynchronous device failure stopped it.
    pub(super) fn is_running(&self) -> bool {
        self.worker
            .as_ref()
            .is_some_and(|worker| !worker.is_finished())
    }

    /// Returns the latest registry count, published before command replies and queue waits.
    pub(super) fn active_voice_count(&self) -> usize {
        self.status.active_voices.load(Ordering::Acquire)
    }

    fn check_error(&self) -> SoundResult<()> {
        self.status.check_error()?;
        self.stop_handle.check_error()
    }

    /// Runs an owned operation on the worker, publishing status before returning its result.
    /// Inputs and results cross the thread boundary; the engine core remains worker-owned.
    pub(super) fn request<T: Send + 'static>(
        &self,
        operation: impl FnOnce(&mut EngineCore) -> SoundResult<T> + Send + 'static,
    ) -> SoundResult<T> {
        self.check_error()?;
        let (command, response) = EngineCommand::request(operation);
        self.commands
            .send(command)
            .map_err(|_| self.stopped_error("sound player worker is not running"))?;
        response
            .recv()
            .map_err(|_| self.stopped_error("sound player worker stopped before replying"))?
    }

    fn stopped_error(&self, message: &str) -> SoundError {
        self.check_error()
            .err()
            .unwrap_or_else(|| SoundError::invalid_state(message))
    }
}

impl Drop for EngineRuntime {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.take() {
            let _ = self.stop_handle.request_stop();
            let _ = self.commands.send(EngineCommand::Shutdown);
            let _ = worker.join();
        }
    }
}

/// Runs the worker loop that owns direct engine state during background playback.
///
/// Commands are handled between render blocks so public API calls can mutate the
/// core without sharing it across threads. Rendered blocks are pushed into the
/// device player until the voices finish or shutdown is requested.
fn runtime_loop(
    core: &mut EngineCore,
    player: &SoundPlayer,
    receiver: &Receiver<EngineCommand>,
    pending_listener: &Mutex<Option<ListenerPose>>,
    status: &RuntimeStatus,
) -> SoundResult<()> {
    let mut output_block = vec![0.0; core.block_size() * core.output_channels()];
    let block_duration =
        std::time::Duration::from_secs_f64(core.block_size() as f64 / core.sample_rate() as f64);
    // Repay expensive spatial work by producing audio before the next phase.
    // This preserves the configured queue capacity and the complete ray budget.
    let mut render_debt = std::time::Duration::ZERO;
    let mut diagnostics = RuntimeDiagnostics::new(
        player,
        core.sample_rate(),
        core.output_channels(),
        output_block.len(),
    );

    loop {
        player.check_error()?;
        player.set_expects_audio(core.has_active_voices());
        // If there are no active voices, block and wait for a command.
        if !core.has_active_voices() {
            render_debt = std::time::Duration::ZERO;
            match receive_while_running(receiver, || !player.is_stop_requested()) {
                Some(EngineCommand::Shutdown) => {
                    return player.finish();
                }
                Some(command) => {
                    if handle_command(command, core, status) {
                        return player.stop();
                    }
                }
                None => {
                    return player.stop();
                }
            }
        }

        // If there are active voices, handle any pending commands without blocking.
        loop {
            match receiver.try_recv() {
                Ok(command) => {
                    if handle_command(command, core, status) {
                        return player.stop();
                    }
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => return player.stop(),
            }
        }

        if core.has_active_voices() {
            player.set_expects_audio(true);
            let started = std::time::Instant::now();
            let pose = pending_listener
                .lock()
                .map_err(|_| SoundError::poisoned_lock("listener mailbox"))?
                .take();
            if let Some(pose) = pose {
                core.set_listener_pose(pose)?;
            }
            output_block.fill(0.0);
            let spatial_deferred = !render_debt.is_zero();
            let rendered = core.render_realtime_block(&mut output_block, !spatial_deferred);
            status.publish(core);
            let metrics = match rendered {
                Ok(metrics) => metrics,
                Err(error) => {
                    let _ = player.stop();
                    return Err(error);
                }
            };
            render_debt = (render_debt + started.elapsed()).saturating_sub(block_duration);
            diagnostics.record(&output_block, started, metrics);

            match player.push_block(&output_block) {
                Ok(true) => {}
                Ok(false) => return player.check_error(),
                Err(error) => return Err(error),
            }
            if !core.has_active_voices() {
                player.release_prebuffer();
            }
        } else if player.is_stop_requested() {
            // With no active voices, no push_block call remains to observe the
            // stop flag, so the idle loop must exit explicitly.
            return player.stop();
        }
    }
}

// Device failure can request stop without sending an engine command. Bound idle
// waits so the worker observes that flag even when no voice is producing blocks.
fn receive_while_running<T>(receiver: &Receiver<T>, running: impl Fn() -> bool) -> Option<T> {
    while running() {
        match receiver.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(command) => return Some(command),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => return None,
        }
    }
    None
}

/// Applies one runtime command and returns `true` when the worker should stop.
fn handle_command(command: EngineCommand, core: &mut EngineCore, status: &RuntimeStatus) -> bool {
    match command {
        EngineCommand::Run(operation) => operation(core, status),
        EngineCommand::Shutdown => return true,
    }
    false
}

#[cfg(test)]
#[path = "../../tests/unit/engine/runtime.rs"]
mod tests;

#[cfg(test)]
#[path = "../../tests/unit/engine/runtime_lifecycle.rs"]
mod lifecycle_tests;
