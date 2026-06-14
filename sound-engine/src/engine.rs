use crate::acoustics::{
    AcousticConfig, AcousticPipeline, AcousticQuery, IrCache, IrCacheConfig as AcousticIrCacheConfig, IrCacheQuery,
    IrSnapshot,
};
use crate::auralization::{AuralizationEngine, ImpulseResponseId, PlaySoundRequest, SoundAsset, SoundId, VoiceId};
use crate::core::config::{EngineConfig, IrCacheConfig as CoreIrCacheConfig};
use crate::core::error::{SoundError, SoundResult};
use crate::gpu::backend::VkBackend;
use crate::scene::{SceneDescription, SceneManager, SceneUpdates, SceneVersion};
use glam::Vec3;
use std::path::Path;
use std::sync::Arc;

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
    /// Linear dry gain applied before convolution.
    pub gain: f32,
}

impl PlaySpatialSoundRequest {
    /// Creates a play request with unit source energy and unit gain.
    pub fn new(sound_id: SoundId, source_position: Vec3) -> Self {
        Self {
            sound_id,
            source: PointSource::new(source_position),
            gain: 1.0,
        }
    }
}

/// Synchronous scene-to-audio facade for the sound engine.
///
/// `SoundEngine` owns the GPU backend, scene state, acoustic IR pipeline,
/// IR cache, sound assets, active voices, and block rendering. It performs no
/// device playback; callers can pull interleaved blocks with [`render_block`].
///
/// [`render_block`]: Self::render_block
pub struct SoundEngine {
    gpu: Arc<VkBackend>,
    scene: SceneManager,
    pipeline: AcousticPipeline,
    cache: IrCache,
    auralization: AuralizationEngine,
    listener: ListenerPose,
    next_impulse_response_id: ImpulseResponseId,
}

impl SoundEngine {
    /// Creates an engine with an empty scene and sound bank.
    pub fn new(config: EngineConfig) -> SoundResult<Self> {
        let gpu = Arc::new(VkBackend::new()?);
        let pipeline = AcousticPipeline::new(gpu.as_ref(), acoustic_config(config))?;
        let cache = IrCache::new(cache_config(config.cache));
        let auralization = AuralizationEngine::new(gpu.clone(), config)?;

        Ok(Self {
            gpu,
            scene: SceneManager::new(),
            pipeline,
            cache,
            auralization,
            listener: ListenerPose::default(),
            next_impulse_response_id: 1,
        })
    }

    /// Replaces the scene and invalidates cached impulse responses.
    pub fn load_scene(&mut self, scene: SceneDescription) -> SoundResult<SceneVersion> {
        let version = self.scene.load(scene)?;
        self.cache.clear();
        Ok(version)
    }

    /// Applies scene updates and invalidates cached impulse responses when anything changed.
    pub fn apply_scene_updates(&mut self, updates: SceneUpdates) -> SoundResult<SceneVersion> {
        let changed = !updates.updates.is_empty();
        let version = self.scene.apply_updates(updates)?;
        if changed {
            self.cache.clear();
        }
        Ok(version)
    }

    /// Sets the listener pose used by subsequent spatial play requests.
    pub fn set_listener_pose(&mut self, listener: ListenerPose) {
        self.listener = listener;
    }

    /// Loads a WAV asset into the engine sound bank at the configured sample rate.
    pub fn load_wav_sound(&mut self, path: impl AsRef<Path>) -> SoundResult<SoundId> {
        self.auralization.load_wav_sound(path)
    }

    /// Inserts an already decoded mono sound asset into the engine sound bank.
    pub fn insert_sound(&mut self, asset: SoundAsset) -> SoundResult<SoundId> {
        self.auralization.insert_sound(asset)
    }

    /// Plays a sound through an acoustic IR generated from the current scene and listener.
    pub fn play_sound(&mut self, request: PlaySpatialSoundRequest) -> SoundResult<VoiceId> {
        validate_play_request(request)?;
        let snapshot = self.impulse_response_for(request)?;
        let impulse_response_id = self.next_ir_id();
        self.auralization
            .register_impulse_response(impulse_response_id, &snapshot.samples)?;
        self.auralization.play_sound(PlaySoundRequest {
            sound_id: request.sound_id,
            impulse_response_id,
            gain: request.gain,
        })
    }

    /// Stops an active voice.
    pub fn stop_voice(&mut self, voice_id: VoiceId) {
        self.auralization.stop_voice(voice_id);
    }

    /// Renders and mixes one interleaved output block from all active voices.
    pub fn render_block(&mut self, output_block: &mut [f32]) -> SoundResult<()> {
        self.auralization.render_block(output_block)
    }

    /// Returns whether the engine currently has active or tail-flushing voices.
    pub fn has_active_voices(&self) -> bool {
        self.auralization.has_active_voices()
    }

    /// Returns the number of active or tail-flushing voices.
    pub fn active_voice_count(&self) -> usize {
        self.auralization.active_voice_count()
    }

    /// Returns the configured engine sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.auralization.sample_rate()
    }

    /// Returns the number of frames rendered per block.
    pub fn block_size(&self) -> usize {
        self.auralization.block_size()
    }

    /// Returns the number of interleaved output channels per frame.
    pub fn output_channels(&self) -> usize {
        self.auralization.output_channels()
    }

    fn impulse_response_for(&mut self, request: PlaySpatialSoundRequest) -> SoundResult<IrSnapshot> {
        let cache_query = self.cache_query(request);
        if let Some(snapshot) = self.cache.get(cache_query) {
            return Ok(snapshot);
        }

        let snapshot = self.pipeline.build_ir(
            self.gpu.as_ref(),
            &mut self.scene,
            AcousticQuery {
                query_id: cache_query.query_id,
                source_position: request.source.position,
                listener_position: self.listener.position,
                listener_right: self.listener.right,
                source_energy: request.source.energy,
            },
        )?;
        self.cache.insert(cache_query, snapshot.clone());
        Ok(snapshot)
    }

    fn cache_query(&self, request: PlaySpatialSoundRequest) -> IrCacheQuery {
        cache_query(self.scene.version(), self.listener, request)
    }

    fn next_ir_id(&mut self) -> ImpulseResponseId {
        let id = self.next_impulse_response_id;
        self.next_impulse_response_id = self.next_impulse_response_id.saturating_add(1);
        id
    }
}

fn cache_query(scene_version: SceneVersion, listener: ListenerPose, request: PlaySpatialSoundRequest) -> IrCacheQuery {
    IrCacheQuery {
        scene_version,
        query_id: 0,
        source_position: request.source.position,
        source_energy: request.source.energy,
        listener_position: listener.position,
        listener_right: listener.right,
    }
}

fn validate_play_request(request: PlaySpatialSoundRequest) -> SoundResult<()> {
    if !request.gain.is_finite() {
        return Err(SoundError::invalid_argument("sound gain must be finite"));
    }
    if !request.source.energy.is_finite() {
        return Err(SoundError::invalid_argument("source energy must be finite"));
    }
    if !request.source.position.is_finite() {
        return Err(SoundError::invalid_argument("source position must be finite"));
    }
    Ok(())
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
    fn play_request_defaults_to_unit_energy_and_gain() {
        let request = PlaySpatialSoundRequest::new(7, Vec3::new(1.0, 2.0, 3.0));

        assert_eq!(request.sound_id, 7);
        assert_eq!(request.source.position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(request.source.energy, 1.0);
        assert_eq!(request.gain, 1.0);
    }

    #[test]
    fn rejects_non_finite_play_request_values() {
        let mut request = PlaySpatialSoundRequest::new(1, Vec3::ZERO);
        request.gain = f32::NAN;
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
            gain: 0.5,
        };

        let query = cache_query(9, listener, request);

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
