//! Background playback runtime for the public engine facade.

use super::{EngineCore, ListenerPose, PlaySpatialSoundRequest, PointSource};
use crate::acoustics::IrSnapshot;
use crate::auralization::{SoundAsset, SoundId, VoiceId};
use crate::core::error::{SoundError, SoundResult};
use crate::playback::{PlaybackConfig, SoundPlayer};
use crate::scene::{SceneDescription, SceneUpdates, SceneVersion};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender, SyncSender};
use std::thread::{self, JoinHandle};

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
    /// Start a spatial voice and reply with its voice id.
    PlaySound(PlaySpatialSoundRequest, SyncSender<SoundResult<VoiceId>>),
    /// Build or reuse an impulse response and reply with the raw snapshot.
    BuildImpulseResponse(PointSource, ListenerPose, SyncSender<SoundResult<IrSnapshot>>),
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
    worker: Option<JoinHandle<(EngineCore, SoundResult<()>)>>,
    /// Sample rate copied from the core before it moved to the worker thread.
    sample_rate: u32,
    /// Render block size copied from the core before it moved to the worker thread.
    block_size: usize,
    /// Output channel count copied from the core before it moved to the worker thread.
    output_channels: usize,
}

/// Startup failure that preserves [`EngineCore`] when the worker can return it safely.
pub(super) enum RuntimeStartError {
    /// The player failed to initialize, but the engine core was recovered.
    Recoverable(EngineCore, SoundError),
    /// The worker failed in a way that prevented core recovery.
    Fatal(SoundError),
}

impl EngineRuntime {
    /// Starts the playback worker and moves direct engine ownership into it.
    pub(super) fn start(core: EngineCore, playback_config: PlaybackConfig) -> Result<Self, RuntimeStartError> {
        let sample_rate = core.sample_rate();
        let block_size = core.block_size();
        let output_channels = core.output_channels();
        let (sender, receiver) = mpsc::channel();
        let (init_tx, init_rx) = mpsc::sync_channel(1);

        let worker = thread::spawn(move || {
            let player = match SoundPlayer::new(sample_rate, output_channels, block_size, playback_config) {
                Ok(player) => {
                    let _ = init_tx.send(Ok(()));
                    player
                }
                Err(error) => {
                    let _ = init_tx.send(Err(error));
                    return (core, Ok(()));
                }
            };
            runtime_loop(core, player, receiver)
        });

        match init_rx
            .recv()
            .map_err(|_| RuntimeStartError::Fatal(SoundError::external("sound player worker stopped during startup")))?
        {
            Ok(()) => {}
            Err(error) => {
                let _ = sender.send(EngineCommand::Shutdown);
                return match worker.join() {
                    Ok((core, _)) => Err(RuntimeStartError::Recoverable(core, error)),
                    Err(_) => Err(RuntimeStartError::Fatal(SoundError::external(
                        "sound player worker thread panicked during startup",
                    ))),
                };
            }
        }

        Ok(Self {
            commands: sender,
            worker: Some(worker),
            sample_rate,
            block_size,
            output_channels,
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
        self.request(|reply| EngineCommand::SetListenerPose(listener, reply))
    }

    /// Sends a WAV loading command to the runtime.
    pub(super) fn load_wav_sound(&self, path: PathBuf) -> SoundResult<SoundId> {
        self.request(|reply| EngineCommand::LoadWavSound(path, reply))
    }

    /// Sends an already decoded sound asset to the runtime.
    pub(super) fn insert_sound(&self, asset: SoundAsset) -> SoundResult<SoundId> {
        self.request(|reply| EngineCommand::InsertSound(asset, reply))
    }

    /// Sends a spatial play command to the runtime and waits for the voice id.
    pub(super) fn play_sound(&self, request: PlaySpatialSoundRequest) -> SoundResult<VoiceId> {
        self.request(|reply| EngineCommand::PlaySound(request, reply))
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
        let _ = self.commands.send(EngineCommand::Shutdown);
        let Some(worker) = self.worker.take() else {
            return (
                None,
                Err(SoundError::invalid_state("sound player worker was already stopped")),
            );
        };
        match worker.join() {
            Ok((core, result)) => (Some(core), result),
            Err(_) => (None, Err(SoundError::external("sound player worker thread panicked"))),
        }
    }

    /// Returns the engine sample rate used by the runtime thread.
    pub(super) fn sample_rate(&self) -> u32 {
        self.sample_rate
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
    fn request<T>(&self, build: impl FnOnce(SyncSender<SoundResult<T>>) -> EngineCommand) -> SoundResult<T> {
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        self.commands
            .send(build(reply_tx))
            .map_err(|_| SoundError::invalid_state("sound player worker is not running"))?;
        reply_rx
            .recv()
            .map_err(|_| SoundError::invalid_state("sound player worker stopped before replying"))?
    }
}

/// Runs the worker loop that owns direct engine state during background playback.
///
/// Commands are handled between render blocks so public API calls can mutate the
/// core without sharing it across threads. Rendered blocks are pushed into the
/// device player until the voices finish or shutdown is requested.
fn runtime_loop(
    mut core: EngineCore,
    player: SoundPlayer,
    receiver: Receiver<EngineCommand>,
) -> (EngineCore, SoundResult<()>) {
    let mut output_block = vec![0.0; core.block_size() * core.output_channels()];

    loop {
        // If there are no active voices, block and wait for a command.
        if !core.has_active_voices() {
            match receiver.recv() {
                Ok(EngineCommand::Shutdown) => {
                    let result = player.finish();
                    return (core, result);
                }
                Ok(command) => {
                    if handle_command(command, &mut core) {
                        let result = player.stop();
                        return (core, result);
                    }
                }
                Err(_) => {
                    let result = player.stop();
                    return (core, result);
                }
            }
        }

        // If there are active voices, handle any pending commands without blocking.
        while let Ok(command) = receiver.try_recv() {
            if handle_command(command, &mut core) {
                let result = player.stop();
                return (core, result);
            }
        }

        if core.has_active_voices() {
            output_block.fill(0.0);
            if let Err(error) = core.render_block(&mut output_block) {
                let _ = player.stop();
                return (core, Err(error));
            }

            match player.push_block(&output_block) {
                Ok(true) => {}
                Ok(false) => return (core, Ok(())),
                Err(error) => return (core, Err(error)),
            }
        } else if player.is_stop_requested() {
            // With no active voices, no push_block call remains to observe the
            // stop flag, so the idle loop must exit explicitly.
            let result = player.stop();
            return (core, result);
        }
    }
}

/// Applies one runtime command and returns `true` when the worker should stop.
fn handle_command(command: EngineCommand, core: &mut EngineCore) -> bool {
    match command {
        EngineCommand::LoadScene(scene, reply) => {
            let _ = reply.send(core.load_scene(scene));
            false
        }
        EngineCommand::ApplySceneUpdates(updates, reply) => {
            let _ = reply.send(core.apply_scene_updates(updates));
            false
        }
        EngineCommand::SetListenerPose(listener, reply) => {
            core.set_listener_pose(listener);
            let _ = reply.send(Ok(()));
            false
        }
        EngineCommand::LoadWavSound(path, reply) => {
            let _ = reply.send(core.load_wav_sound(path));
            false
        }
        EngineCommand::InsertSound(asset, reply) => {
            let _ = reply.send(core.insert_sound(asset));
            false
        }
        EngineCommand::PlaySound(request, reply) => {
            let _ = reply.send(core.play_sound(request));
            false
        }
        EngineCommand::BuildImpulseResponse(source, listener, reply) => {
            let _ = reply.send(core.build_impulse_response(source, listener));
            false
        }
        EngineCommand::StopVoice(voice_id, reply) => {
            core.stop_voice(voice_id);
            let _ = reply.send(Ok(()));
            false
        }
        EngineCommand::Shutdown => true,
    }
}
