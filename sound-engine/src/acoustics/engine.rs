use crate::acoustics::{
    AcousticPipeline, AcousticPipelineConfig, AcousticQuery, GpuIrSnapshot, IrCache, IrSnapshot,
    ListenerPose, PointSource,
};
use crate::core::config::IrCacheConfig;
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
    pub fn new(
        gpu: &VkBackend,
        config: AcousticPipelineConfig,
        cache: IrCacheConfig,
    ) -> SoundResult<Self> {
        Ok(Self {
            scene: SceneManager::new(),
            pipeline: AcousticPipeline::new(gpu, config)?,
            cache: IrCache::new(cache),
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
        let query = AcousticQuery {
            query_id: 0,
            source,
            listener,
        }
        .canonicalized()?;
        let scene_version = self.scene.version();
        if let Some(snapshot) = self.cache.get_ir(scene_version, query) {
            return Ok(snapshot);
        }

        // Building an IR can update scene-side acceleration data, so the scene
        // manager remains mutable even though the acoustic query is read-only.
        let snapshot = self.pipeline.build_ir_gpu(gpu, &mut self.scene, query)?;
        self.cache.insert_ir(scene_version, query, snapshot.clone());
        Ok(snapshot)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/acoustics/engine.rs"]
mod tests;
