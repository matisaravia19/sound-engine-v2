//! Public sound engine facade and shared orchestration types.

mod direct;
#[cfg(feature = "playback")]
mod runtime;

use crate::acoustics::{AcousticConfig, IrCacheConfig as AcousticIrCacheConfig, IrCacheQuery, IrSnapshot};
use crate::auralization::{SoundAsset, SoundId, VoiceId};
use crate::core::config::{EngineConfig, IrCacheConfig as CoreIrCacheConfig};
use crate::core::error::{SoundError, SoundResult};
use crate::scene::{SceneDescription, SceneUpdates, SceneVersion};
use direct::EngineCore;
use glam::Vec3;
use std::path::Path;

#[cfg(feature = "playback")]
use crate::playback::PlaybackConfig;
#[cfg(feature = "playback")]
use runtime::{EngineRuntime, RuntimeStartError};

/// Listener pose used for spatial acoustic queries.
#[derive(Debug, Clone, Copy)]
pub struct ListenerPose {
    /// Listener position in world meters.
    pub position: Vec3,
    /// Listener right-ear direction in world space.
    pub right: Vec3,
}

impl Default for ListenerPose {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            right: Vec3::X,
        }
    }
}

/// Point sound source rendered through the current acoustic scene.
#[derive(Debug, Clone, Copy)]
pub struct PointSource {
    /// Source position in world meters.
    pub position: Vec3,
    /// Acoustic source energy used when generating an impulse response.
    pub energy: f32,
}

impl PointSource {
    /// Creates a point source with unit acoustic energy.
    pub fn new(position: Vec3) -> Self {
        Self { position, energy: 1.0 }
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
    /// Creates a play request with unit source energy and unit volume.
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
    mode: Option<EngineMode>,
}

enum EngineMode {
    Direct(EngineCore),
    #[cfg(feature = "playback")]
    Runtime(EngineRuntime),
}

impl SoundEngine {
    /// Creates an engine with an empty scene and sound bank.
    pub fn new(config: EngineConfig) -> SoundResult<Self> {
        Ok(Self {
            mode: Some(EngineMode::Direct(EngineCore::new(config)?)),
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
            EngineMode::Direct(core) => {
                core.set_listener_pose(listener);
                Ok(())
            }
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

    /// Plays a sound through an acoustic IR generated from the current scene and listener.
    pub fn play_sound(&mut self, request: PlaySpatialSoundRequest) -> SoundResult<VoiceId> {
        match self.mode_mut()? {
            EngineMode::Direct(core) => core.play_sound(request),
            #[cfg(feature = "playback")]
            EngineMode::Runtime(runtime) => runtime.play_sound(request),
        }
    }

    /// Builds or reuses the raw acoustic impulse response for a source/listener pair.
    pub fn build_impulse_response(&mut self, source: PointSource, listener: ListenerPose) -> SoundResult<IrSnapshot> {
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
            return Err(SoundError::invalid_state("sound engine is already transitioning modes"));
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
            return Err(SoundError::invalid_state("sound engine is already transitioning modes"));
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

fn cache_query(scene_version: SceneVersion, source: PointSource, listener: ListenerPose) -> IrCacheQuery {
    IrCacheQuery {
        scene_version,
        query_id: 0,
        source_position: source.position,
        source_energy: source.energy,
        listener_position: listener.position,
        listener_right: listener.right,
    }
}

fn validate_source(source: PointSource) -> SoundResult<()> {
    if !source.energy.is_finite() {
        return Err(SoundError::invalid_argument("source energy must be finite"));
    }
    if !source.position.is_finite() {
        return Err(SoundError::invalid_argument("source position must be finite"));
    }
    Ok(())
}

fn validate_listener(listener: ListenerPose) -> SoundResult<()> {
    if !listener.position.is_finite() {
        return Err(SoundError::invalid_argument("listener position must be finite"));
    }
    if !listener.right.is_finite() {
        return Err(SoundError::invalid_argument("listener right vector must be finite"));
    }
    Ok(())
}

fn validate_play_request(request: PlaySpatialSoundRequest) -> SoundResult<()> {
    if !request.volume.is_finite() {
        return Err(SoundError::invalid_argument("sound volume must be finite"));
    }
    validate_source(request.source)
}

fn acoustic_config(config: EngineConfig) -> AcousticConfig {
    AcousticConfig {
        sample_rate: config.sound.sample_rate,
        num_samples: config.sound.ir_num_samples,
        output_channels: config.sound.output_channels,
        listener_half_extent: config.acoustics.listener_half_extent,
        rays_per_query: config.acoustics.rays_per_query,
        max_bounces: config.acoustics.max_bounces,
    }
}

fn cache_config(config: CoreIrCacheConfig) -> AcousticIrCacheConfig {
    AcousticIrCacheConfig {
        position_cell_meters: config.position_cell_meters,
        reuse_distance_meters: config.reuse_distance_meters,
        listener_right_dot_threshold: config.listener_right_dot_threshold,
        max_entries: config.max_entries,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::{AcousticsConfig, AuralizationConfig, OutputChannels, SoundConfig};

    #[test]
    fn play_request_defaults_to_unit_energy_and_volume() {
        let request = PlaySpatialSoundRequest::new(7, Vec3::new(1.0, 2.0, 3.0));

        assert_eq!(request.sound_id, 7);
        assert_eq!(request.source.position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(request.source.energy, 1.0);
        assert_eq!(request.volume, 1.0);
    }

    #[test]
    fn rejects_non_finite_play_request_values() {
        let mut request = PlaySpatialSoundRequest::new(1, Vec3::ZERO);
        request.volume = f32::NAN;
        assert!(validate_play_request(request).is_err());

        request = PlaySpatialSoundRequest::new(1, Vec3::new(f32::INFINITY, 0.0, 0.0));
        assert!(validate_play_request(request).is_err());

        request = PlaySpatialSoundRequest::new(1, Vec3::ZERO);
        request.source.energy = f32::NAN;
        assert!(validate_play_request(request).is_err());
    }

    #[test]
    fn maps_engine_config_to_acoustic_config() {
        let config = test_config();
        let acoustic = acoustic_config(config);

        assert_eq!(acoustic.sample_rate, config.sound.sample_rate);
        assert_eq!(acoustic.num_samples, config.sound.ir_num_samples);
        assert_eq!(acoustic.output_channels, config.sound.output_channels);
        assert_eq!(acoustic.listener_half_extent, config.acoustics.listener_half_extent);
        assert_eq!(acoustic.rays_per_query, config.acoustics.rays_per_query);
        assert_eq!(acoustic.max_bounces, config.acoustics.max_bounces);
    }

    #[test]
    fn maps_core_cache_config_to_acoustic_cache_config() {
        let config = CoreIrCacheConfig {
            position_cell_meters: 0.5,
            reuse_distance_meters: 0.25,
            listener_right_dot_threshold: 0.9,
            max_entries: 4,
        };
        let acoustic = cache_config(config);

        assert_eq!(acoustic.position_cell_meters, config.position_cell_meters);
        assert_eq!(acoustic.reuse_distance_meters, config.reuse_distance_meters);
        assert_eq!(
            acoustic.listener_right_dot_threshold,
            config.listener_right_dot_threshold
        );
        assert_eq!(acoustic.max_entries, config.max_entries);
    }

    #[test]
    fn builds_cache_query_from_listener_and_source() {
        let listener = ListenerPose {
            position: Vec3::new(1.0, 2.0, 3.0),
            right: Vec3::Y,
        };
        let request = PlaySpatialSoundRequest {
            sound_id: 1,
            source: PointSource {
                position: Vec3::new(4.0, 5.0, 6.0),
                energy: 0.75,
            },
            volume: 0.5,
        };

        let query = cache_query(9, request.source, listener);

        assert_eq!(query.scene_version, 9);
        assert_eq!(query.query_id, 0);
        assert_eq!(query.source_position, request.source.position);
        assert_eq!(query.source_energy, request.source.energy);
        assert_eq!(query.listener_position, listener.position);
        assert_eq!(query.listener_right, listener.right);
    }

    fn test_config() -> EngineConfig {
        EngineConfig {
            sound: SoundConfig {
                output_channels: OutputChannels::Stereo,
                sample_rate: 48_000,
                ir_num_samples: 128,
            },
            acoustics: AcousticsConfig {
                listener_half_extent: Vec3::splat(0.2),
                rays_per_query: 1,
                max_bounces: 0,
            },
            cache: CoreIrCacheConfig::default(),
            auralization: AuralizationConfig { block_size: 64 },
        }
    }
}
