//! Public sound engine facade and shared orchestration types.

#[cfg(feature = "playback")]
mod diagnostics;
mod direct;
#[cfg(feature = "playback")]
mod runtime;
mod types;
#[cfg(feature = "playback")]
mod worker;

pub use types::{PlaySpatialSoundRequest, SoundSourceId};

use crate::acoustics::{IrSnapshot, ListenerPose, PointSource};
use crate::auralization::{SoundAsset, SoundId, VoiceId};
use crate::core::config::EngineConfig;
use crate::core::error::{SoundError, SoundResult};
use crate::scene::{SceneDescription, SceneUpdates, SceneVersion};
use direct::EngineCore;
use std::path::Path;

#[cfg(feature = "playback")]
use crate::playback::PlaybackConfig;
#[cfg(feature = "playback")]
use runtime::EngineRuntime;

/// Public handle for synchronous rendering or feature-gated background playback.
/// Dropping the handle stops and joins its worker before releasing the engine core.
/// If stopping cannot recover the core after a worker panic, subsequent operations
/// return that failure. Audio-format getters remain available.
pub struct SoundEngine {
    /// Current core owner, or the failure that prevented ownership recovery.
    mode: SoundResult<EngineMode>,
    /// Immutable sample rate, retained independently of worker ownership.
    sample_rate: u32,
    /// Immutable number of frames rendered per block.
    block_size: usize,
    /// Immutable number of interleaved output channels per frame.
    output_channels: usize,
}

/// Internal ownership mode for the engine core.
enum EngineMode {
    /// Synchronous mode where the public handle owns the core directly.
    Direct(Box<EngineCore>),
    #[cfg(feature = "playback")]
    /// Background playback mode where the runtime worker owns the core.
    Runtime(EngineRuntime),
}

impl SoundEngine {
    /// Creates the scalar acoustic engine with an empty scene and sound bank.
    /// Invalid settings are rejected before Vulkan initialization.
    pub fn new(config: EngineConfig) -> SoundResult<Self> {
        let core = Box::new(EngineCore::new(config)?);
        Ok(Self::from_core(core))
    }

    fn from_core(core: Box<EngineCore>) -> Self {
        Self {
            sample_rate: core.sample_rate(),
            block_size: core.block_size(),
            output_channels: core.output_channels(),
            mode: Ok(EngineMode::Direct(core)),
        }
    }

    /// Replaces the scene and invalidates cached impulse responses.
    pub fn load_scene(&mut self, scene: SceneDescription) -> SoundResult<SceneVersion> {
        self.with_core(move |core| core.load_scene(scene))
    }

    /// Applies scene updates and invalidates cached impulse responses when anything changed.
    pub fn apply_scene_updates(&mut self, updates: SceneUpdates) -> SoundResult<SceneVersion> {
        self.with_core(move |core| core.apply_scene_updates(updates))
    }

    /// Sets the listener pose used by subsequent spatial play requests.
    pub fn set_listener_pose(&mut self, listener: ListenerPose) -> SoundResult<()> {
        match self.mode_mut()? {
            EngineMode::Direct(core) => core.set_listener_pose(listener),
            #[cfg(feature = "playback")]
            EngineMode::Runtime(runtime) => runtime.set_listener_pose(listener),
        }
    }

    /// Loads a WAV asset into the engine sound bank at the configured sample rate.
    pub fn load_wav_sound(&mut self, path: impl AsRef<Path>) -> SoundResult<SoundId> {
        let path = path.as_ref().to_path_buf();
        self.with_core(move |core| core.load_wav_sound(path))
    }

    /// Inserts an already decoded mono sound asset into the engine sound bank.
    pub fn insert_sound(&mut self, asset: SoundAsset) -> SoundResult<SoundId> {
        self.with_core(move |core| core.insert_sound(asset))
    }

    /// Submits the latest listener pose without waiting for background rendering.
    /// During playback, poses are coalesced in a capacity-one mailbox and applied
    /// before an audio block. Dirty voices refresh one at a time and crossfade
    /// into their new IR. Direct mode applies immediately. Use
    /// `set_listener_pose` when subsequent synchronous requests must observe it.
    pub fn update_listener_pose(&mut self, listener: ListenerPose) -> SoundResult<()> {
        match self.mode_mut()? {
            EngineMode::Direct(core) => core.set_listener_pose(listener),
            #[cfg(feature = "playback")]
            EngineMode::Runtime(runtime) => runtime.update_listener_pose(listener),
        }
    }

    /// Creates a tracked movable source for future spatial playback.
    pub fn create_sound_source(&mut self, source: PointSource) -> SoundResult<SoundSourceId> {
        self.with_core(move |core| core.create_sound_source(source))
    }

    /// Updates the position and amplitude of an existing tracked source.
    pub fn update_sound_source(
        &mut self,
        id: SoundSourceId,
        source: PointSource,
    ) -> SoundResult<()> {
        self.with_core(move |core| core.update_sound_source(id, source))
    }

    /// Removes a tracked source that is not referenced by an active voice.
    pub fn remove_sound_source(&mut self, id: SoundSourceId) -> SoundResult<()> {
        self.with_core(move |core| core.remove_sound_source(id))
    }

    /// Plays a sound through an acoustic IR generated from the current scene and listener.
    pub fn play_sound(&mut self, request: PlaySpatialSoundRequest) -> SoundResult<VoiceId> {
        self.with_core(move |core| core.play_sound(request))
    }

    /// Plays a sound through an acoustic IR generated from a tracked source.
    pub fn play_sound_from_source(
        &mut self,
        sound_id: SoundId,
        source_id: SoundSourceId,
        volume: f32,
    ) -> SoundResult<VoiceId> {
        self.with_core(move |core| core.play_sound_from_source(sound_id, source_id, volume))
    }

    /// Builds or reuses the raw acoustic impulse response for a source/listener pair.
    pub fn build_impulse_response(
        &mut self,
        source: PointSource,
        listener: ListenerPose,
    ) -> SoundResult<IrSnapshot> {
        self.with_core(move |core| core.build_impulse_response(source, listener))
    }

    /// Stops an active voice.
    pub fn stop_voice(&mut self, voice_id: VoiceId) -> SoundResult<()> {
        self.with_core(move |core| {
            core.stop_voice(voice_id);
            Ok(())
        })
    }

    /// Renders and mixes one interleaved output block from all active voices.
    pub fn render_block(&mut self, output_block: &mut [f32]) -> SoundResult<()> {
        match self.mode_mut()? {
            EngineMode::Direct(core) => core.render_block(output_block),
            #[cfg(feature = "playback")]
            EngineMode::Runtime(_) => Err(SoundError::invalid_state(
                "render_block cannot be used while the sound player is running",
            )),
        }
    }

    /// Returns whether the engine currently has active or tail-flushing voices.
    /// In runtime mode this reads the latest published registry count; queued
    /// device audio can remain after all voices finish rendering.
    pub fn has_active_voices(&self) -> bool {
        match self.mode.as_ref() {
            Ok(EngineMode::Direct(core)) => core.has_active_voices(),
            #[cfg(feature = "playback")]
            Ok(EngineMode::Runtime(runtime)) => runtime.active_voice_count() != 0,
            Err(_) => false,
        }
    }

    /// Returns active or tail-flushing voices, including the latest worker snapshot.
    /// A stopped worker retains voices until direct ownership is recovered,
    /// unless a panic destroyed the core.
    pub fn active_voice_count(&self) -> usize {
        match self.mode.as_ref() {
            Ok(EngineMode::Direct(core)) => core.active_voice_count(),
            #[cfg(feature = "playback")]
            Ok(EngineMode::Runtime(runtime)) => runtime.active_voice_count(),
            Err(_) => 0,
        }
    }

    /// Returns the configured engine sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Returns the number of frames rendered per block.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Returns the number of interleaved output channels per frame.
    pub fn output_channels(&self) -> usize {
        self.output_channels
    }

    /// Starts a feature-gated background sound player.
    #[cfg(feature = "playback")]
    pub fn start_sound_player(&mut self, config: PlaybackConfig) -> SoundResult<()> {
        match self.take_mode()? {
            EngineMode::Direct(core) => match EngineRuntime::start(core, config) {
                Ok(runtime) => {
                    self.mode = Ok(EngineMode::Runtime(runtime));
                    Ok(())
                }
                Err(error) => {
                    self.mode = Ok(EngineMode::Direct(error.core));
                    Err(error.error)
                }
            },
            EngineMode::Runtime(runtime) => {
                self.mode = Ok(EngineMode::Runtime(runtime));
                Err(SoundError::invalid_state("sound player is already running"))
            }
        }
    }

    /// Stops the feature-gated background sound player and returns to direct mode.
    /// Device and rendering errors return with direct ownership restored. A worker
    /// panic destroys the core, leaving the handle permanently failed.
    #[cfg(feature = "playback")]
    pub fn stop_sound_player(&mut self) -> SoundResult<()> {
        match self.take_mode()? {
            EngineMode::Direct(core) => {
                self.mode = Ok(EngineMode::Direct(core));
                Ok(())
            }
            EngineMode::Runtime(runtime) => match runtime.stop() {
                Ok((core, result)) => {
                    self.mode = Ok(EngineMode::Direct(core));
                    result
                }
                Err(error) => {
                    self.mode = Err(error.clone());
                    Err(error)
                }
            },
        }
    }

    /// Returns whether the feature-gated background sound player is running.
    /// After asynchronous failure, call `stop_sound_player` to retrieve the error
    /// and recover direct ownership unless the worker panicked.
    #[cfg(feature = "playback")]
    pub fn is_sound_player_running(&self) -> bool {
        matches!(&self.mode, Ok(EngineMode::Runtime(runtime)) if runtime.is_running())
    }

    /// Applies an operation directly or waits for it on the core's owning worker.
    fn with_core<T: Send + 'static>(
        &mut self,
        operation: impl FnOnce(&mut EngineCore) -> SoundResult<T> + Send + 'static,
    ) -> SoundResult<T> {
        match self.mode_mut()? {
            EngineMode::Direct(core) => operation(core),
            #[cfg(feature = "playback")]
            EngineMode::Runtime(runtime) => runtime.request(operation),
        }
    }

    /// Borrows the current owner or reports the retained ownership failure.
    fn mode_mut(&mut self) -> SoundResult<&mut EngineMode> {
        self.mode.as_mut().map_err(|error| error.clone())
    }

    /// Moves ownership out while leaving a failure if the transfer unwinds.
    #[cfg(feature = "playback")]
    fn take_mode(&mut self) -> SoundResult<EngineMode> {
        self.mode_mut()?;
        std::mem::replace(
            &mut self.mode,
            Err(SoundError::invalid_state(
                "sound engine ownership transfer was interrupted",
            )),
        )
    }
}

/// Validates a spatial play request before allocating acoustic or voice resources.
fn validate_play_request(request: PlaySpatialSoundRequest) -> SoundResult<()> {
    if !request.volume.is_finite() {
        return Err(SoundError::invalid_argument("sound volume must be finite"));
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/unit/engine/mod.rs"]
mod tests;
