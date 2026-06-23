//! Direct, synchronous engine implementation.

use super::{
    ListenerPose, PlaySpatialSoundRequest, PointSource, cache_config, cache_query, validate_listener,
    validate_play_request, validate_source,
};
use crate::acoustics::{AcousticPipeline, AcousticQuery, IrCache, IrSnapshot};
use crate::auralization::{AuralizationEngine, ImpulseResponseId, PlaySoundRequest, SoundAsset, SoundId, VoiceId};
use crate::core::config::EngineConfig;
use crate::core::error::SoundResult;
use crate::gpu::backend::VkBackend;
use crate::scene::{SceneDescription, SceneManager, SceneUpdates, SceneVersion};
use std::path::Path;
use std::sync::Arc;

/// Private owner of GPU, scene, acoustic cache, assets, voices, and block rendering.
pub(super) struct EngineCore {
    /// Shared Vulkan backend used by acoustic tracing and auralization compute work.
    gpu: Arc<VkBackend>,
    /// Mutable scene state and versioning used when building acoustic queries.
    scene: SceneManager,
    /// Ray-tracing pipeline that produces impulse responses for source/listener pairs.
    pipeline: AcousticPipeline,
    /// Reuse cache for impulse responses derived from scene, source, and listener state.
    cache: IrCache,
    /// Asset bank, active voices, and convolver used to render output blocks.
    auralization: AuralizationEngine,
    /// Listener pose used by spatial play requests that do not supply one directly.
    listener: ListenerPose,
    /// Monotonic id assigned to impulse responses registered with the auralizer.
    next_impulse_response_id: ImpulseResponseId,
}

impl EngineCore {
    /// Creates a direct engine core with an empty scene and sound bank.
    pub(super) fn new(config: EngineConfig) -> SoundResult<Self> {
        let gpu = Arc::new(VkBackend::new()?);
        let pipeline = AcousticPipeline::new(gpu.as_ref(), config)?;
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

    /// Replaces the scene and clears IR cache entries tied to prior geometry.
    pub(super) fn load_scene(&mut self, scene: SceneDescription) -> SoundResult<SceneVersion> {
        let version = self.scene.load(scene)?;
        self.cache.clear();
        Ok(version)
    }

    /// Applies scene edits and clears cached IRs when scene contents changed.
    pub(super) fn apply_scene_updates(&mut self, updates: SceneUpdates) -> SoundResult<SceneVersion> {
        let changed = !updates.updates.is_empty();
        let version = self.scene.apply_updates(updates)?;
        if changed {
            self.cache.clear();
        }
        Ok(version)
    }

    /// Sets the listener pose used by future IR queries.
    pub(super) fn set_listener_pose(&mut self, listener: ListenerPose) {
        self.listener = listener;
    }

    /// Loads a WAV asset into the auralizer.
    pub(super) fn load_wav_sound(&mut self, path: impl AsRef<Path>) -> SoundResult<SoundId> {
        self.auralization.load_wav_sound(path)
    }

    /// Inserts a decoded mono asset into the auralizer.
    pub(super) fn insert_sound(&mut self, asset: SoundAsset) -> SoundResult<SoundId> {
        self.auralization.insert_sound(asset)
    }

    /// Builds or reuses an acoustic IR and starts a convolved voice.
    pub(super) fn play_sound(&mut self, request: PlaySpatialSoundRequest) -> SoundResult<VoiceId> {
        validate_play_request(request)?;
        let snapshot = self.impulse_response_for(request.source, self.listener)?;
        let impulse_response_id = self.next_ir_id();
        self.auralization
            .register_impulse_response(impulse_response_id, &snapshot.samples)?;
        self.auralization.play_sound(PlaySoundRequest {
            sound_id: request.sound_id,
            impulse_response_id,
            gain: request.volume,
        })
    }

    /// Builds or reuses the raw acoustic impulse response for a source/listener pair.
    pub(super) fn build_impulse_response(
        &mut self,
        source: PointSource,
        listener: ListenerPose,
    ) -> SoundResult<IrSnapshot> {
        validate_source(source)?;
        validate_listener(listener)?;
        self.impulse_response_for(source, listener)
    }

    /// Stops an active voice if it exists.
    pub(super) fn stop_voice(&mut self, voice_id: VoiceId) {
        self.auralization.stop_voice(voice_id);
    }

    /// Renders one interleaved block through the auralizer.
    pub(super) fn render_block(&mut self, output_block: &mut [f32]) -> SoundResult<()> {
        self.auralization.render_block(output_block)
    }

    /// Returns whether any voices are still active or flushing tails.
    pub(super) fn has_active_voices(&self) -> bool {
        self.auralization.has_active_voices()
    }

    /// Returns the number of voices still active or flushing tails.
    pub(super) fn active_voice_count(&self) -> usize {
        self.auralization.active_voice_count()
    }

    /// Returns the configured engine sample rate.
    pub(super) fn sample_rate(&self) -> u32 {
        self.auralization.sample_rate()
    }

    /// Returns the configured render block size in frames.
    pub(super) fn block_size(&self) -> usize {
        self.auralization.block_size()
    }

    /// Returns the configured number of interleaved output channels.
    pub(super) fn output_channels(&self) -> usize {
        self.auralization.output_channels()
    }

    fn impulse_response_for(&mut self, source: PointSource, listener: ListenerPose) -> SoundResult<IrSnapshot> {
        let cache_query = self.cache_query(source, listener);
        if let Some(snapshot) = self.cache.get(cache_query) {
            return Ok(snapshot);
        }

        // Building an IR can update scene-side acceleration data, so the scene
        // manager is passed mutably even though the acoustic query is read-only.
        let snapshot = self.pipeline.build_ir(
            self.gpu.as_ref(),
            &mut self.scene,
            AcousticQuery {
                query_id: cache_query.query_id,
                source_position: source.position,
                listener_position: listener.position,
                listener_right: listener.right,
                source_energy: source.energy,
            },
        )?;
        self.cache.insert(cache_query, snapshot.clone());
        Ok(snapshot)
    }

    /// Builds the cache key for the current scene version and acoustic endpoints.
    fn cache_query(&self, source: PointSource, listener: ListenerPose) -> crate::acoustics::IrCacheQuery {
        cache_query(self.scene.version(), source, listener)
    }

    /// Returns a fresh auralization impulse-response id.
    fn next_ir_id(&mut self) -> ImpulseResponseId {
        let id = self.next_impulse_response_id;
        self.next_impulse_response_id = self.next_impulse_response_id.saturating_add(1);
        id
    }
}
