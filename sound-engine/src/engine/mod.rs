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
pub struct SoundEngine {
    /// Current owner mode; temporarily `None` while ownership moves between modes.
    mode: Option<EngineMode>,
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
    pub fn new(config: EngineConfig) -> SoundResult<Self> {
        Ok(Self {
            mode: Some(EngineMode::Direct(Box::new(EngineCore::new(config)?))),
        })
    }

    /// Replaces the scene and invalidates cached impulse responses.
    pub fn load_scene(&mut self, scene: SceneDescription) -> SoundResult<SceneVersion> {
        match self.mode_mut()? {
            EngineMode::Direct(core) => core.load_scene(scene),
            #[cfg(feature = "playback")]
            EngineMode::Runtime(runtime) => runtime.load_scene(scene),
        }
    }

    /// Applies scene updates and invalidates cached impulse responses when anything changed.
    pub fn apply_scene_updates(&mut self, updates: SceneUpdates) -> SoundResult<SceneVersion> {
        match self.mode_mut()? {
            EngineMode::Direct(core) => core.apply_scene_updates(updates),
            #[cfg(feature = "playback")]
            EngineMode::Runtime(runtime) => runtime.apply_scene_updates(updates),
        }
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
        let path = path.as_ref();
        match self.mode_mut()? {
            EngineMode::Direct(core) => core.load_wav_sound(path),
            #[cfg(feature = "playback")]
            EngineMode::Runtime(runtime) => runtime.load_wav_sound(path.to_path_buf()),
        }
    }

    /// Inserts an already decoded mono sound asset into the engine sound bank.
    pub fn insert_sound(&mut self, asset: SoundAsset) -> SoundResult<SoundId> {
        match self.mode_mut()? {
            EngineMode::Direct(core) => core.insert_sound(asset),
            #[cfg(feature = "playback")]
            EngineMode::Runtime(runtime) => runtime.insert_sound(asset),
        }
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
        match self.mode_mut()? {
            EngineMode::Direct(core) => core.create_sound_source(source),
            #[cfg(feature = "playback")]
            EngineMode::Runtime(runtime) => runtime.create_sound_source(source),
        }
    }

    /// Updates the position and amplitude of an existing tracked source.
    pub fn update_sound_source(
        &mut self,
        id: SoundSourceId,
        source: PointSource,
    ) -> SoundResult<()> {
        match self.mode_mut()? {
            EngineMode::Direct(core) => core.update_sound_source(id, source),
            #[cfg(feature = "playback")]
            EngineMode::Runtime(runtime) => runtime.update_sound_source(id, source),
        }
    }

    /// Removes a tracked source that is not referenced by an active voice.
    pub fn remove_sound_source(&mut self, id: SoundSourceId) -> SoundResult<()> {
        match self.mode_mut()? {
            EngineMode::Direct(core) => core.remove_sound_source(id),
            #[cfg(feature = "playback")]
            EngineMode::Runtime(runtime) => runtime.remove_sound_source(id),
        }
    }

    /// Plays a sound through an acoustic IR generated from the current scene and listener.
    pub fn play_sound(&mut self, request: PlaySpatialSoundRequest) -> SoundResult<VoiceId> {
        match self.mode_mut()? {
            EngineMode::Direct(core) => core.play_sound(request),
            #[cfg(feature = "playback")]
            EngineMode::Runtime(runtime) => runtime.play_sound(request),
        }
    }

    /// Plays a sound through an acoustic IR generated from a tracked source.
    pub fn play_sound_from_source(
        &mut self,
        sound_id: SoundId,
        source_id: SoundSourceId,
        volume: f32,
    ) -> SoundResult<VoiceId> {
        match self.mode_mut()? {
            EngineMode::Direct(core) => core.play_sound_from_source(sound_id, source_id, volume),
            #[cfg(feature = "playback")]
            EngineMode::Runtime(runtime) => {
                runtime.play_sound_from_source(sound_id, source_id, volume)
            }
        }
    }

    /// Builds or reuses the raw acoustic impulse response for a source/listener pair.
    pub fn build_impulse_response(
        &mut self,
        source: PointSource,
        listener: ListenerPose,
    ) -> SoundResult<IrSnapshot> {
        match self.mode_mut()? {
            EngineMode::Direct(core) => core.build_impulse_response(source, listener),
            #[cfg(feature = "playback")]
            EngineMode::Runtime(runtime) => runtime.build_impulse_response(source, listener),
        }
    }

    /// Stops an active voice.
    pub fn stop_voice(&mut self, voice_id: VoiceId) -> SoundResult<()> {
        match self.mode_mut()? {
            EngineMode::Direct(core) => {
                core.stop_voice(voice_id);
                Ok(())
            }
            #[cfg(feature = "playback")]
            EngineMode::Runtime(runtime) => runtime.stop_voice(voice_id),
        }
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
            Some(EngineMode::Direct(core)) => core.has_active_voices(),
            #[cfg(feature = "playback")]
            Some(EngineMode::Runtime(runtime)) => runtime.active_voice_count() != 0,
            None => false,
        }
    }

    /// Returns active or tail-flushing voices, including the latest worker snapshot.
    /// A stopped worker retains voices until direct ownership is recovered.
    pub fn active_voice_count(&self) -> usize {
        match self.mode.as_ref() {
            Some(EngineMode::Direct(core)) => core.active_voice_count(),
            #[cfg(feature = "playback")]
            Some(EngineMode::Runtime(runtime)) => runtime.active_voice_count(),
            None => 0,
        }
    }

    /// Returns the configured engine sample rate.
    pub fn sample_rate(&self) -> u32 {
        match self.mode.as_ref() {
            Some(EngineMode::Direct(core)) => core.sample_rate(),
            #[cfg(feature = "playback")]
            Some(EngineMode::Runtime(runtime)) => runtime.sample_rate(),
            None => 0,
        }
    }

    /// Returns the number of frames rendered per block.
    pub fn block_size(&self) -> usize {
        match self.mode.as_ref() {
            Some(EngineMode::Direct(core)) => core.block_size(),
            #[cfg(feature = "playback")]
            Some(EngineMode::Runtime(runtime)) => runtime.block_size(),
            None => 0,
        }
    }

    /// Returns the number of interleaved output channels per frame.
    pub fn output_channels(&self) -> usize {
        match self.mode.as_ref() {
            Some(EngineMode::Direct(core)) => core.output_channels(),
            #[cfg(feature = "playback")]
            Some(EngineMode::Runtime(runtime)) => runtime.output_channels(),
            None => 0,
        }
    }

    /// Starts a feature-gated background sound player.
    #[cfg(feature = "playback")]
    pub fn start_sound_player(&mut self, config: PlaybackConfig) -> SoundResult<()> {
        let Some(mode) = self.mode.take() else {
            return Err(SoundError::invalid_state(
                "sound engine is already transitioning modes",
            ));
        };

        match mode {
            EngineMode::Direct(core) => match EngineRuntime::start(core, config) {
                Ok(runtime) => {
                    self.mode = Some(EngineMode::Runtime(runtime));
                    Ok(())
                }
                Err(error) => {
                    self.mode = Some(EngineMode::Direct(error.core));
                    Err(error.error)
                }
            },
            EngineMode::Runtime(runtime) => {
                self.mode = Some(EngineMode::Runtime(runtime));
                Err(SoundError::invalid_state("sound player is already running"))
            }
        }
    }

    /// Stops the feature-gated background sound player and returns to direct mode.
    #[cfg(feature = "playback")]
    pub fn stop_sound_player(&mut self) -> SoundResult<()> {
        let Some(mode) = self.mode.take() else {
            return Err(SoundError::invalid_state(
                "sound engine is already transitioning modes",
            ));
        };

        match mode {
            EngineMode::Direct(core) => {
                self.mode = Some(EngineMode::Direct(core));
                Ok(())
            }
            EngineMode::Runtime(runtime) => {
                let (core, result) = runtime.stop();
                if let Some(core) = core {
                    self.mode = Some(EngineMode::Direct(core));
                }
                result
            }
        }
    }

    /// Returns whether the feature-gated background sound player is running.
    /// After asynchronous failure, call `stop_sound_player` to retrieve the error
    /// and recover the engine's direct ownership.
    #[cfg(feature = "playback")]
    pub fn is_sound_player_running(&self) -> bool {
        matches!(&self.mode, Some(EngineMode::Runtime(runtime)) if runtime.is_running())
    }

    /// Borrows the current mode unless ownership is being transferred.
    fn mode_mut(&mut self) -> SoundResult<&mut EngineMode> {
        self.mode
            .as_mut()
            .ok_or_else(|| SoundError::invalid_state("sound engine is transitioning modes"))
    }
}

impl Drop for SoundEngine {
    fn drop(&mut self) {
        #[cfg(feature = "playback")]
        if matches!(self.mode, Some(EngineMode::Runtime(_))) {
            let _ = self.stop_sound_player();
        }
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
