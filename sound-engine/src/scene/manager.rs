use super::gpu::GpuSceneResources;
use super::store::SceneStore;
use super::types::{SceneDescription, SceneUpdates, SceneVersion};
use crate::error::{SoundError, SoundResult};
use crate::gpu::backend::VkBackend;

/// High-level scene owner used by the acoustic pipeline.
///
/// `SceneManager` owns CPU scene state and the latest GPU mirror. GPU resources
/// are rebuilt lazily when callers request them after a scene version change.
pub struct SceneManager {
    store: SceneStore,
    gpu_resources: Option<GpuSceneResources>,
}

impl SceneManager {
    /// Creates an empty scene manager.
    pub fn new() -> Self {
        Self {
            store: SceneStore::new(),
            gpu_resources: None,
        }
    }

    /// Replaces the scene and drops any stale GPU mirror.
    pub fn load(&mut self, scene: SceneDescription) -> SoundResult<SceneVersion> {
        self.store.load(scene)?;
        self.gpu_resources = None;
        Ok(self.store.version())
    }

    /// Applies scene updates; GPU resources are rebuilt on the next sync.
    pub fn apply_updates(&mut self, updates: SceneUpdates) -> SoundResult<SceneVersion> {
        self.store.apply_updates(updates)?;
        Ok(self.store.version())
    }

    /// Returns the canonical CPU scene store.
    pub fn store(&self) -> &SceneStore {
        &self.store
    }

    /// Returns the current CPU scene version.
    pub fn version(&self) -> SceneVersion {
        self.store.version()
    }

    /// Returns GPU scene resources, rebuilding them if the CPU scene changed.
    ///
    /// The returned resources are owned by the manager and remain valid until
    /// the next mutable scene operation or GPU sync that replaces them.
    pub fn sync_gpu_if_needed<'a>(&'a mut self, gpu: &VkBackend) -> SoundResult<&'a GpuSceneResources> {
        let needs_sync = self
            .gpu_resources
            .as_ref()
            .map_or(true, |resources| resources.version() != self.store.version());

        if needs_sync {
            let previous = self.gpu_resources.take();
            self.gpu_resources = Some(GpuSceneResources::build(gpu, &self.store, previous)?);
        }

        self.gpu_resources
            .as_ref()
            .ok_or_else(|| SoundError::invalid_state("GPU scene resources were not created"))
    }
}

impl Default for SceneManager {
    fn default() -> Self {
        Self::new()
    }
}
