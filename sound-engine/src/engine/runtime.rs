//! Background playback runtime for the public engine facade.

use super::{EngineCore, ListenerPose, PlaySpatialSoundRequest, PointSource, SoundSourceId};
use crate::acoustics::IrSnapshot;
use crate::auralization::{SoundAsset, SoundId, VoiceId};
use crate::core::error::{SoundError, SoundResult};
use crate::playback::{PlaybackConfig, SoundPlayer};
use crate::scene::{SceneDescription, SceneUpdates, SceneVersion};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender, SyncSender};
use std::sync::{Arc, Mutex};
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
    /// Create a tracked movable sound source.
    CreateSoundSource(PointSource, SyncSender<SoundResult<SoundSourceId>>),
    /// Update an existing tracked sound source.
    UpdateSoundSource(SoundSourceId, PointSource, SyncSender<SoundResult<()>>),
    /// Remove a tracked sound source.
    RemoveSoundSource(SoundSourceId, SyncSender<SoundResult<()>>),
    /// Start a spatial voice and reply with its voice id.
    PlaySound(PlaySpatialSoundRequest, SyncSender<SoundResult<VoiceId>>),
    /// Start a spatial voice from a tracked sound source.
    PlaySoundFromSource(SoundId, SoundSourceId, f32, SyncSender<SoundResult<VoiceId>>),
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
    /// Capacity-one pose mailbox; stale game frames are replaced before rendering.
    pending_listener: Arc<Mutex<Option<ListenerPose>>>,
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
        let pending_listener = Arc::new(Mutex::new(None));
        let worker_listener = pending_listener.clone();

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
            runtime_loop(core, player, receiver, worker_listener)
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
            pending_listener,
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
        if self.worker.as_ref().is_none_or(|worker| worker.is_finished()) {
            return Err(SoundError::invalid_state("sound player worker is not running"));
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
    pub(super) fn update_sound_source(&self, id: SoundSourceId, source: PointSource) -> SoundResult<()> {
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
    pending_listener: Arc<Mutex<Option<ListenerPose>>>,
) -> (EngineCore, SoundResult<()>) {
    let mut output_block = vec![0.0; core.block_size() * core.output_channels()];
    let block_duration = std::time::Duration::from_secs_f64(core.block_size() as f64 / core.sample_rate() as f64);
    // Repay expensive spatial work by producing audio before the next phase.
    // This preserves the configured queue capacity and the complete ray budget.
    let mut render_debt = std::time::Duration::ZERO;
    let mut measurements = RuntimeMeasurements {
        player: &player,
        path: std::env::var_os("SOUND_ENGINE_TIMING_CSV").map(PathBuf::from),
        blocks: Vec::new(),
        started: std::time::Instant::now(),
        capture_path: std::env::var_os("SOUND_ENGINE_CAPTURE_WAV").map(PathBuf::from),
        capture: Vec::new(),
        sample_rate: core.sample_rate(),
        channels: core.output_channels(),
    };

    loop {
        player.set_expects_audio(core.has_active_voices());
        // If there are no active voices, block and wait for a command.
        if !core.has_active_voices() {
            render_debt = std::time::Duration::ZERO;
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
            player.set_expects_audio(true);
            let started = std::time::Instant::now();
            let pose = match pending_listener.lock() {
                Ok(mut pending) => pending.take(),
                Err(_) => return (core, Err(SoundError::poisoned_lock("listener mailbox"))),
            };
            if let Some(pose) = pose {
                if let Err(error) = core.set_listener_pose(pose) {
                    return (core, Err(error));
                }
            }
            output_block.fill(0.0);
            let spatial_deferred = !render_debt.is_zero();
            let rendered = core.render_realtime_block(&mut output_block, !spatial_deferred);
            let stages = match rendered {
                Ok(stages) => stages,
                Err(error) => {
                    let _ = player.stop();
                    return (core, Err(error));
                }
            };
            render_debt = (render_debt + started.elapsed()).saturating_sub(block_duration);
            if measurements.path.is_some() {
                let time = measurements.started.elapsed().as_secs_f64();
                let ms = started.elapsed().as_secs_f64() * 1000.0;
                let mut squares = [0.0_f64; 2];
                for frame in output_block.chunks_exact(core.output_channels()) {
                    for (channel, &sample) in frame.iter().enumerate() {
                        squares[channel] += (sample as f64).powi(2);
                    }
                }
                let (spatial_updates, spatial_update_age_ms) = core.spatial_update_status();
                measurements.blocks.push(RuntimeBlockMeasurement {
                    time,
                    ms,
                    left: (squares[0] / core.block_size() as f64).sqrt(),
                    right: (squares[1] / core.block_size() as f64).sqrt(),
                    stages,
                    spatial_updates,
                    spatial_update_age_ms,
                    spatial_deferred,
                });
            }
            if measurements.capture_path.is_some() {
                measurements.capture.extend_from_slice(&output_block);
            }

            match player.push_block(&output_block) {
                Ok(true) => {}
                Ok(false) => return (core, Ok(())),
                Err(error) => return (core, Err(error)),
            }
            if !core.has_active_voices() {
                player.release_prebuffer();
            }
        } else if player.is_stop_requested() {
            // With no active voices, no push_block call remains to observe the
            // stop flag, so the idle loop must exit explicitly.
            let result = player.stop();
            return (core, result);
        }
    }
}

/// One producer block's timing and coherent IR update diagnostics.
struct RuntimeBlockMeasurement {
    /// Seconds since worker startup, including initialization.
    time: f64,
    /// Producer wall time, excluding waiting for queue capacity.
    ms: f64,
    /// Left channel RMS before queuing.
    left: f64,
    /// Right channel RMS before queuing.
    right: f64,
    /// Ray/synthesis, IR registration and convolution wall times in milliseconds.
    stages: [f64; 3],
    /// Cumulative count of replaced IRs; cache reuse does not increment it.
    spatial_updates: u64,
    /// Last replacement's request-to-registration age, excluding device queuing.
    spatial_update_age_ms: f64,
    /// Whether this block repaid prior render work without advancing the IR.
    spatial_deferred: bool,
}

/// Optional in-memory measurements, flushed only after the playback loop stops.
struct RuntimeMeasurements<'a> {
    /// Borrowed player remains alive until diagnostics have been captured.
    player: &'a SoundPlayer,
    /// Explicit output destination; absence disables sample retention.
    path: Option<PathBuf>,
    /// Elapsed seconds, render time, ear RMS and trace/prepare/convolve times.
    blocks: Vec<RuntimeBlockMeasurement>,
    /// Monotonic origin for correlating updates and frame times.
    started: std::time::Instant,
    /// Optional capture of rendered blocks before the device queue (not loopback).
    capture_path: Option<PathBuf>,
    /// Interleaved rendered frames, retained only when capture is requested.
    capture: Vec<f32>,
    /// WAV sample rate copied before the core moves or shuts down.
    sample_rate: u32,
    /// Engine channel count stored in the capture header.
    channels: usize,
}

impl Drop for RuntimeMeasurements<'_> {
    fn drop(&mut self) {
        // Snapshot before diagnostic file I/O; callbacks may still arrive until
        // the device stream is dropped after this diagnostics owner.
        let (underruns, missing) = self.player.starvation_counts();
        let starvation_times = self.player.starvation_time_range_seconds();
        if let Some(path) = &self.capture_path {
            let result = (|| -> Result<(), hound::Error> {
                let spec = hound::WavSpec {
                    channels: self.channels as u16,
                    sample_rate: self.sample_rate,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                };
                let mut writer = hound::WavWriter::create(path, spec)?;
                for &sample in &self.capture {
                    writer.write_sample(sample)?;
                }
                writer.finalize()
            })();
            if let Err(error) = result {
                eprintln!("Could not save rendered audio: {error}");
            }
        }
        let Some(path) = &self.path else {
            return;
        };
        let result = (|| -> std::io::Result<()> {
            use std::io::Write;
            let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
            writeln!(
                file,
                "block,elapsed_seconds,render_ms,left_rms,right_rms,trace_ms,ir_prepare_ms,convolution_ms,spatial_updates,spatial_update_age_ms,spatial_deferred"
            )?;
            for (index, block) in self.blocks.iter().enumerate() {
                writeln!(
                    file,
                    "{index},{},{},{},{},{},{},{},{},{},{}",
                    block.time,
                    block.ms,
                    block.left,
                    block.right,
                    block.stages[0],
                    block.stages[1],
                    block.stages[2],
                    block.spatial_updates,
                    block.spatial_update_age_ms,
                    u8::from(block.spatial_deferred)
                )?;
            }
            std::fs::write(
                path.with_extension("audio.json"),
                format!(
                    "{{\"underrun_callbacks\":{underruns},\"missing_frames\":{missing},\"first_underrun_seconds_since_player_creation\":{},\"last_underrun_seconds_since_player_creation\":{}}}\n",
                    starvation_times.map_or_else(|| "null".to_owned(), |(first, _)| first.to_string()),
                    starvation_times.map_or_else(|| "null".to_owned(), |(_, last)| last.to_string()),
                ),
            )?;
            Ok(())
        })();
        if let Err(error) = result {
            eprintln!("Could not save runtime measurements: {error}");
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
            let _ = reply.send(core.set_listener_pose(listener));
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
        EngineCommand::CreateSoundSource(source, reply) => {
            let _ = reply.send(core.create_sound_source(source));
            false
        }
        EngineCommand::UpdateSoundSource(id, source, reply) => {
            let _ = reply.send(core.update_sound_source(id, source));
            false
        }
        EngineCommand::RemoveSoundSource(id, reply) => {
            let _ = reply.send(core.remove_sound_source(id));
            false
        }
        EngineCommand::PlaySound(request, reply) => {
            let _ = reply.send(core.play_sound(request));
            false
        }
        EngineCommand::PlaySoundFromSource(sound_id, source_id, volume, reply) => {
            let _ = reply.send(core.play_sound_from_source(sound_id, source_id, volume));
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
