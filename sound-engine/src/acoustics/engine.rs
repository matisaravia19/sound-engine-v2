use crate::acoustics::{
    AcousticPipeline, AcousticQuery, GpuIrSnapshot, IrCache, IrCacheQuery, IrSnapshot,
    ListenerPose, PointSource, canonicalize_listener, validate_source,
};
use crate::core::config::EngineConfig;
use crate::core::error::SoundResult;
use crate::gpu::backend::VkBackend;
use crate::scene::{SceneDescription, SceneManager, SceneUpdates, SceneVersion};

/// Scene-backed acoustic IR engine with synchronous cache-backed queries.
///
/// The engine owns scene state, the GPU acoustic pipeline, and the IR cache.
/// Cache hits share GPU snapshots, while misses may synchronize scene
/// acceleration data and block on GPU ray tracing before returning.
pub struct AcousticsEngine {
    /// Mutable scene state and versioning used when building acoustic queries.
    scene: SceneManager,
    /// Ray-tracing pipeline that produces impulse responses for source/listener pairs.
    pipeline: AcousticPipeline,
    /// Reuse cache for impulse responses derived from scene, source, and listener state.
    cache: IrCache,
}

impl AcousticsEngine {
    /// Creates the scalar acoustic engine with an empty scene and IR cache.
    pub fn new(gpu: &VkBackend, config: EngineConfig) -> SoundResult<Self> {
        Ok(Self {
            scene: SceneManager::new(),
            pipeline: AcousticPipeline::new(gpu, config)?,
            cache: IrCache::new(config.cache),
        })
    }

    /// Replaces the scene and clears IR cache entries tied to prior geometry.
    pub fn load_scene(&mut self, scene: SceneDescription) -> SoundResult<SceneVersion> {
        let version = self.scene.load(scene)?;
        self.cache.clear();
        Ok(version)
    }

    /// Applies scene edits and clears cached IRs when scene contents changed.
    pub fn apply_scene_updates(&mut self, updates: SceneUpdates) -> SoundResult<SceneVersion> {
        let changed = !updates.updates.is_empty();
        let version = self.scene.apply_updates(updates)?;
        if changed {
            self.cache.clear();
        }
        Ok(version)
    }

    /// Builds or reuses an acoustic impulse response for a source/listener pair.
    pub fn build_impulse_response(
        &mut self,
        gpu: &VkBackend,
        source: PointSource,
        listener: ListenerPose,
    ) -> SoundResult<IrSnapshot> {
        let snapshot = self.build_impulse_response_gpu(gpu, source, listener)?;
        self.pipeline.download_gpu_ir(gpu, &snapshot)
    }

    /// Builds or reuses a GPU-resident acoustic impulse response for a source/listener pair.
    pub fn build_impulse_response_gpu(
        &mut self,
        gpu: &VkBackend,
        source: PointSource,
        listener: ListenerPose,
    ) -> SoundResult<GpuIrSnapshot> {
        self.pipeline.validate_backend(gpu)?;
        validate_source(source)?;
        let listener = canonicalize_listener(listener)?;

        let cache_query = cache_query(self.scene.version(), source, listener);
        if let Some(snapshot) = self.cache.get_ir(cache_query) {
            return Ok(snapshot);
        }

        // Building an IR can update scene-side acceleration data, so the scene
        // manager remains mutable even though the acoustic query is read-only.
        let snapshot = self.pipeline.build_ir_gpu(
            gpu,
            &mut self.scene,
            AcousticQuery {
                query_id: cache_query.query_id,
                source_position: source.position,
                listener_position: listener.position,
                listener_forward: listener.forward,
                listener_up: listener.up,
                source_amplitude: source.amplitude,
            },
        )?;
        self.cache.insert_ir(cache_query, snapshot.clone());
        Ok(snapshot)
    }
}

/// Builds the impulse-response cache key for the current acoustic endpoints.
fn cache_query(
    scene_version: SceneVersion,
    source: PointSource,
    listener: ListenerPose,
) -> IrCacheQuery {
    IrCacheQuery {
        scene_version,
        query_id: 0,
        source_position: source.position,
        source_amplitude: source.amplitude,
        listener_position: listener.position,
        listener_forward: listener.forward,
        listener_up: listener.up,
    }
}

#[cfg(test)]
#[path = "../../tests/unit/acoustics/engine.rs"]
mod tests;
