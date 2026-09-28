//! Background playback runtime for the public engine facade.

use super::diagnostics::RuntimeDiagnostics;
use super::{EngineCore, ListenerPose, PlaySpatialSoundRequest, PointSource, SoundSourceId};
use crate::acoustics::IrSnapshot;
use crate::auralization::{SoundAsset, SoundId, VoiceId};
use crate::core::error::{SoundError, SoundResult};
use crate::playback::{PlaybackConfig, PlaybackStopHandle, SoundPlayer};
use crate::scene::{SceneDescription, SceneUpdates, SceneVersion};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, SyncSender};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;

#[cfg(test)]
#[path = "runtime_tests.rs"]
mod tests;

/// Commands sent from the public handle to the runtime-owned engine core.
enum EngineCommand {
    /// Replace the scene and reply with the new scene version.
    LoadScene(SceneDescription, SyncSender<SoundResult<SceneVersion>>),
    /// Apply scene edits and reply with the resulting scene version.
    ApplySceneUpdates(SceneUpdates, SyncSender<SoundResult<SceneVersion>>),
    /// Update the listener pose used by later spatial play requests.
    SetListenerPose(ListenerPose, SyncSender<SoundResult<()>>),
    /// Load a WAV file on the worker thread and reply with its sound id.
    LoadWavSound(PathBuf, SyncSender<SoundResult<SoundId>>),
    /// Insert an already decoded asset on the worker thread.
    InsertSound(SoundAsset, SyncSender<SoundResult<SoundId>>),
    /// Create a tracked movable sound source.
    CreateSoundSource(PointSource, SyncSender<SoundResult<SoundSourceId>>),
    /// Update an existing tracked sound source.
    UpdateSoundSource(SoundSourceId, PointSource, SyncSender<SoundResult<()>>),
    /// Remove a tracked sound source.
    RemoveSoundSource(SoundSourceId, SyncSender<SoundResult<()>>),
    /// Start a spatial voice and reply with its voice id.
    PlaySound(PlaySpatialSoundRequest, SyncSender<SoundResult<VoiceId>>),
    /// Start a spatial voice from a tracked sound source.
    PlaySoundFromSource(
        SoundId,
        SoundSourceId,
        f32,
        SyncSender<SoundResult<VoiceId>>,
    ),
    /// Build or reuse an impulse response and reply with the raw snapshot.
    BuildImpulseResponse(
        PointSource,
        ListenerPose,
        SyncSender<SoundResult<IrSnapshot>>,
    ),
    /// Stop an active voice if it exists.
    StopVoice(VoiceId, SyncSender<SoundResult<()>>),
    /// Stop the worker loop and return the owned engine core.
    Shutdown,
}

/// Handle for the background thread that owns [`EngineCore`] while playback is running.
pub(super) struct EngineRuntime {
    /// Command sender used by the public facade to request worker-thread work.
    commands: Sender<EngineCommand>,
    /// Join handle for recovering direct engine ownership when playback stops.
    worker: Option<super::worker::OwnerThread<EngineCore>>,
    /// Sample rate copied from the core before it moved to the worker thread.
    sample_rate: u32,
    /// Render block size copied from the core before it moved to the worker thread.
    block_size: usize,
    /// Output channel count copied from the core before it moved to the worker thread.
    output_channels: usize,
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
    pub(super) core: EngineCore,
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

struct PanicStatusGuard(Arc<RuntimeStatus>);

impl Drop for PanicStatusGuard {
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
        core: EngineCore,
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
        core: EngineCore,
        initialize: impl FnOnce() -> SoundResult<SoundPlayer> + Send + 'static,
    ) -> Result<Self, RuntimeStartError> {
        let sample_rate = core.sample_rate();
        let block_size = core.block_size();
        let output_channels = core.output_channels();
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
                let _panic_status = PanicStatusGuard(worker_status.clone());
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
            sample_rate,
            block_size,
            output_channels,
            pending_listener,
            stop_handle,
            status,
        })
    }

    /// Sends a scene replacement command to the runtime.
    pub(super) fn load_scene(&self, scene: SceneDescription) -> SoundResult<SceneVersion> {
        self.request(|reply| EngineCommand::LoadScene(scene, reply))
    }

    /// Sends a scene update command to the runtime.
    pub(super) fn apply_scene_updates(&self, updates: SceneUpdates) -> SoundResult<SceneVersion> {
        self.request(|reply| EngineCommand::ApplySceneUpdates(updates, reply))
    }

    /// Sends a listener pose update command to the runtime.
    pub(super) fn set_listener_pose(&self, listener: ListenerPose) -> SoundResult<()> {
        self.pending_listener
            .lock()
            .map_err(|_| SoundError::poisoned_lock("listener mailbox"))?
            .take();
        self.request(|reply| EngineCommand::SetListenerPose(listener, reply))
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

    /// Sends a WAV loading command to the runtime.
    pub(super) fn load_wav_sound(&self, path: PathBuf) -> SoundResult<SoundId> {
        self.request(|reply| EngineCommand::LoadWavSound(path, reply))
    }

    /// Sends an already decoded sound asset to the runtime.
    pub(super) fn insert_sound(&self, asset: SoundAsset) -> SoundResult<SoundId> {
        self.request(|reply| EngineCommand::InsertSound(asset, reply))
    }

    /// Sends a tracked source creation command to the runtime.
    pub(super) fn create_sound_source(&self, source: PointSource) -> SoundResult<SoundSourceId> {
        self.request(|reply| EngineCommand::CreateSoundSource(source, reply))
    }

    /// Sends a tracked source update command to the runtime.
    pub(super) fn update_sound_source(
        &self,
        id: SoundSourceId,
        source: PointSource,
    ) -> SoundResult<()> {
        self.request(|reply| EngineCommand::UpdateSoundSource(id, source, reply))
    }

    /// Sends a tracked source removal command to the runtime.
    pub(super) fn remove_sound_source(&self, id: SoundSourceId) -> SoundResult<()> {
        self.request(|reply| EngineCommand::RemoveSoundSource(id, reply))
    }

    /// Sends a spatial play command to the runtime and waits for the voice id.
    pub(super) fn play_sound(&self, request: PlaySpatialSoundRequest) -> SoundResult<VoiceId> {
        self.request(|reply| EngineCommand::PlaySound(request, reply))
    }

    /// Sends a tracked-source play command to the runtime and waits for the voice id.
    pub(super) fn play_sound_from_source(
        &self,
        sound_id: SoundId,
        source_id: SoundSourceId,
        volume: f32,
    ) -> SoundResult<VoiceId> {
        self.request(|reply| EngineCommand::PlaySoundFromSource(sound_id, source_id, volume, reply))
    }

    /// Sends an acoustic IR query to the runtime and waits for the raw snapshot.
    pub(super) fn build_impulse_response(
        &self,
        source: PointSource,
        listener: ListenerPose,
    ) -> SoundResult<IrSnapshot> {
        self.request(|reply| EngineCommand::BuildImpulseResponse(source, listener, reply))
    }

    /// Sends a voice stop command to the runtime.
    pub(super) fn stop_voice(&self, voice_id: VoiceId) -> SoundResult<()> {
        self.request(|reply| EngineCommand::StopVoice(voice_id, reply))
    }

    /// Stops the runtime thread and returns direct engine ownership.
    pub(super) fn stop(mut self) -> (Option<EngineCore>, SoundResult<()>) {
        // A command cannot wake a worker blocked in push_block. Wake its queue
        // first; the command separately wakes an idle receiver.
        let stopped = self.stop_handle.request_stop();
        let _ = self.commands.send(EngineCommand::Shutdown);
        let Some(worker) = self.worker.take() else {
            return (
                None,
                Err(SoundError::invalid_state(
                    "sound player worker was already stopped",
                )),
            );
        };
        match worker.join() {
            Ok(Some((core, result))) => (
                Some(core),
                result
                    .and(stopped)
                    .and_then(|()| self.stop_handle.check_error()),
            ),
            Ok(None) => (
                None,
                Err(SoundError::external(
                    "sound player worker stopped before ownership transfer",
                )),
            ),
            Err(_) => (
                None,
                Err(SoundError::external("sound player worker thread panicked")),
            ),
        }
    }

    /// Returns the engine sample rate used by the runtime thread.
    pub(super) fn sample_rate(&self) -> u32 {
        self.sample_rate
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

    /// Returns the render block size used by the runtime thread.
    pub(super) fn block_size(&self) -> usize {
        self.block_size
    }

    /// Returns the output channel count used by the runtime thread.
    pub(super) fn output_channels(&self) -> usize {
        self.output_channels
    }

    /// Sends a command with a one-shot reply channel and waits for the result.
    fn request<T>(
        &self,
        build: impl FnOnce(SyncSender<SoundResult<T>>) -> EngineCommand,
    ) -> SoundResult<T> {
        self.check_error()?;
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        self.commands
            .send(build(reply_tx))
            .map_err(|_| self.stopped_error("sound player worker is not running"))?;
        reply_rx
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
            let stages = match rendered {
                Ok(stages) => stages,
                Err(error) => {
                    let _ = player.stop();
                    return Err(error);
                }
            };
            render_debt = (render_debt + started.elapsed()).saturating_sub(block_duration);
            diagnostics.record(
                &output_block,
                started,
                stages,
                core.spatial_update_status(),
                spatial_deferred,
            );

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
        EngineCommand::LoadScene(scene, reply) => {
            let result = core.load_scene(scene);
            reply_with_status(core, status, reply, result);
        }
        EngineCommand::ApplySceneUpdates(updates, reply) => {
            let result = core.apply_scene_updates(updates);
            reply_with_status(core, status, reply, result);
        }
        EngineCommand::SetListenerPose(listener, reply) => {
            let result = core.set_listener_pose(listener);
            reply_with_status(core, status, reply, result);
        }
        EngineCommand::LoadWavSound(path, reply) => {
            let result = core.load_wav_sound(path);
            reply_with_status(core, status, reply, result);
        }
        EngineCommand::InsertSound(asset, reply) => {
            let result = core.insert_sound(asset);
            reply_with_status(core, status, reply, result);
        }
        EngineCommand::CreateSoundSource(source, reply) => {
            let result = core.create_sound_source(source);
            reply_with_status(core, status, reply, result);
        }
        EngineCommand::UpdateSoundSource(id, source, reply) => {
            let result = core.update_sound_source(id, source);
            reply_with_status(core, status, reply, result);
        }
        EngineCommand::RemoveSoundSource(id, reply) => {
            let result = core.remove_sound_source(id);
            reply_with_status(core, status, reply, result);
        }
        EngineCommand::PlaySound(request, reply) => {
            let result = core.play_sound(request);
            reply_with_status(core, status, reply, result);
        }
        EngineCommand::PlaySoundFromSource(sound_id, source_id, volume, reply) => {
            let result = core.play_sound_from_source(sound_id, source_id, volume);
            reply_with_status(core, status, reply, result);
        }
        EngineCommand::BuildImpulseResponse(source, listener, reply) => {
            let result = core.build_impulse_response(source, listener);
            reply_with_status(core, status, reply, result);
        }
        EngineCommand::StopVoice(voice_id, reply) => {
            core.stop_voice(voice_id);
            reply_with_status(core, status, reply, Ok(()));
        }
        EngineCommand::Shutdown => return true,
    }
    false
}

fn reply_with_status<T>(
    core: &EngineCore,
    status: &RuntimeStatus,
    reply: SyncSender<SoundResult<T>>,
    result: SoundResult<T>,
) {
    // An acknowledged command must not leave the caller seeing the old count.
    status.publish(core);
    let _ = reply.send(result);
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn idle_receiver_observes_stop_without_a_command() {
        let (_sender, receiver) = mpsc::channel::<()>();
        let running = Arc::new(AtomicBool::new(true));
        let worker_running = running.clone();
        let (result_tx, result_rx) = mpsc::channel();
        let worker = thread::spawn(move || {
            let result =
                receive_while_running(&receiver, || worker_running.load(Ordering::Acquire));
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
}
