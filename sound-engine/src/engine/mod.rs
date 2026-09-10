//! Public sound engine facade and shared orchestration types.

mod direct;
#[cfg(feature = "playback")]
mod runtime;

use crate::acoustics::IrSnapshot;
use crate::auralization::{SoundAsset, SoundId, VoiceId};
use crate::core::config::EngineConfig;
use crate::core::error::{SoundError, SoundResult};
use crate::scene::{SceneDescription, SceneUpdates, SceneVersion};
use direct::EngineCore;
use glam::Vec3;
use std::path::Path;

#[cfg(feature = "playback")]
use crate::playback::PlaybackConfig;
#[cfg(feature = "playback")]
use runtime::{EngineRuntime, RuntimeStartError};

/// Identifier for a tracked movable sound source.
pub type SoundSourceId = u64;

/// Listener pose used for spatial acoustic queries.
#[derive(Debug, Clone, Copy)]
pub struct ListenerPose {
    /// Listener position in world meters.
    pub position: Vec3,
    /// Listener forward direction in world space.
    pub forward: Vec3,
    /// Listener up direction in world space.
    ///
    /// Valid poses need not provide an exactly orthogonal unit vector; the
    /// engine orthogonalizes it against `forward` before acoustic queries.
    pub up: Vec3,
}

impl Default for ListenerPose {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            forward: -Vec3::Z,
            up: Vec3::Y,
        }
    }
}

impl ListenerPose {
    /// Creates a listener pose from a position and approximate forward/up axes.
    pub fn new(position: Vec3, forward: Vec3, up: Vec3) -> Self {
        Self {
            position,
            forward,
            up,
        }
    }

    /// Returns the unit right-ear axis derived from this pose.
    ///
    /// Invalid or collinear orientation vectors produce the zero vector; public
    /// engine entry points reject those poses before using them.
    pub fn right(self) -> Vec3 {
        self.forward
            .normalize_or_zero()
            .cross(self.up)
            .normalize_or_zero()
    }
}

/// Point sound source rendered through the current acoustic scene.
#[derive(Debug, Clone, Copy)]
pub struct PointSource {
    /// Source position in world meters.
    pub position: Vec3,
    /// Reference pressure amplitude used when generating an impulse response.
    pub amplitude: f32,
    /// Immutable source calibration and its local-to-world orientation.
    pub radiation: crate::acoustics::SourceRadiation,
}

impl PointSource {
    /// Creates a point source with unit reference pressure amplitude.
    pub fn new(position: Vec3) -> Self {
        Self {
            position,
            amplitude: 1.0,
            radiation: Default::default(),
        }
    }
}

/// Request used by [`SoundEngine`] to play a sound at a world-space position.
#[derive(Debug, Clone, Copy)]
pub struct PlaySpatialSoundRequest {
    /// Dry sound asset to render.
    pub sound_id: SoundId,
    /// World-space source used to build or reuse the acoustic IR.
    pub source: PointSource,
    /// Linear playback volume applied to the dry sound before convolution.
    pub volume: f32,
}

impl PlaySpatialSoundRequest {
    /// Creates a play request with unit source amplitude and unit volume.
    pub fn new(sound_id: SoundId, source_position: Vec3) -> Self {
        Self {
            sound_id,
            source: PointSource::new(source_position),
            volume: 1.0,
        }
    }
}

/// Public handle for synchronous rendering or feature-gated background playback.
pub struct SoundEngine {
    /// Current owner mode; temporarily `None` while ownership moves between modes.
    mode: Option<EngineMode>,
}

/// Internal ownership mode for the engine core.
enum EngineMode {
    /// Synchronous mode where the public handle owns the core directly.
    Direct(EngineCore),
    #[cfg(feature = "playback")]
    /// Background playback mode where the runtime worker owns the core.
    Runtime(EngineRuntime),
}

impl SoundEngine {
    /// Registers immutable directional pressure data for later ray queries.
    /// Call before starting playback: registration allocates/uploads GPU data.
    /// Use the returned engine-local id in `PointSource::radiation`; updating a
    /// tracked source's radiation refreshes its active voices through normal IR updates.
    pub fn register_source_directivity(
        &mut self,
        profile: &crate::acoustics::SourceDirectivity,
    ) -> SoundResult<crate::acoustics::SourceDirectivityId> {
        match self.mode_mut()? {
            EngineMode::Direct(core) => core.register_source_directivity(profile),
            #[cfg(feature = "playback")]
            EngineMode::Runtime(_) => Err(SoundError::invalid_state(
                "register source directivity before starting playback",
            )),
        }
    }
    /// Creates an engine with an empty scene and sound bank.
    pub fn new(config: EngineConfig) -> SoundResult<Self> {
        Self::with_acoustic_model(config, crate::acoustics::AcousticModel::Cosine)
    }

    /// Creates an engine with an explicitly selected ray-local diffraction model.
    /// The selected model supplies both manual IR queries and moving-voice playback.
    pub fn with_acoustic_model(
        config: EngineConfig,
        model: crate::acoustics::AcousticModel,
    ) -> SoundResult<Self> {
        Ok(Self {
            mode: Some(EngineMode::Direct(EngineCore::new(config, model)?)),
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
    /// before an audio block. Band-model IRs finish their coherent ray batches
    /// over later blocks before entering the crossfade. Direct mode applies immediately. Use
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

    /// Updates the position, amplitude and radiation of an existing tracked source.
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
    pub fn has_active_voices(&self) -> bool {
        match self.mode.as_ref() {
            Some(EngineMode::Direct(core)) => core.has_active_voices(),
            #[cfg(feature = "playback")]
            Some(EngineMode::Runtime(_)) => true,
            None => false,
        }
    }

    /// Returns the number of active or tail-flushing voices in direct mode.
    pub fn active_voice_count(&self) -> usize {
        match self.mode.as_ref() {
            Some(EngineMode::Direct(core)) => core.active_voice_count(),
            #[cfg(feature = "playback")]
            Some(EngineMode::Runtime(_)) => 0,
            None => 0,
        }
    }

    /// Returns the configured engine sample rate.
    pub fn sample_rate(&self) -> u32 {
        match self.mode.as_ref() {
            Some(EngineMode::Direct(core)) => Some(core.sample_rate()),
            #[cfg(feature = "playback")]
            Some(EngineMode::Runtime(runtime)) => Some(runtime.sample_rate()),
            None => None,
        }
        .unwrap_or(0)
    }

    /// Returns the number of frames rendered per block.
    pub fn block_size(&self) -> usize {
        match self.mode.as_ref() {
            Some(EngineMode::Direct(core)) => Some(core.block_size()),
            #[cfg(feature = "playback")]
            Some(EngineMode::Runtime(runtime)) => Some(runtime.block_size()),
            None => None,
        }
        .unwrap_or(0)
    }

    /// Returns the number of interleaved output channels per frame.
    pub fn output_channels(&self) -> usize {
        match self.mode.as_ref() {
            Some(EngineMode::Direct(core)) => Some(core.output_channels()),
            #[cfg(feature = "playback")]
            Some(EngineMode::Runtime(runtime)) => Some(runtime.output_channels()),
            None => None,
        }
        .unwrap_or(0)
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
                Err(RuntimeStartError::Recoverable(core, error)) => {
                    self.mode = Some(EngineMode::Direct(core));
                    Err(error)
                }
                Err(RuntimeStartError::Fatal(error)) => Err(error),
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
    #[cfg(feature = "playback")]
    pub fn is_sound_player_running(&self) -> bool {
        matches!(self.mode, Some(EngineMode::Runtime(_)))
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
mod tests {
    use super::*;

    #[test]
    fn play_request_defaults_to_unit_amplitude_and_volume() {
        let request = PlaySpatialSoundRequest::new(7, Vec3::new(1.0, 2.0, 3.0));

        assert_eq!(request.sound_id, 7);
        assert_eq!(request.source.position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(request.source.amplitude, 1.0);
        assert_eq!(request.volume, 1.0);
    }

    #[test]
    fn rejects_non_finite_playback_volume() {
        let mut request = PlaySpatialSoundRequest::new(1, Vec3::ZERO);
        request.volume = f32::NAN;
        assert!(validate_play_request(request).is_err());
    }
}
