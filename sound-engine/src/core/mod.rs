use crate::acoustics::{AcousticPipeline, AcousticQuery, IrCache, IrCacheConfig, IrCacheQuery, IrSnapshot};
use crate::auralization::convolver::PartitionedConvolver;
use crate::auralization::{ImpulseResponseId, VoiceId};
use crate::core::config::EngineConfig;
use crate::core::error::SoundResult;
use crate::gpu::backend::VkBackend;
use crate::scene::{SceneDescription, SceneManager, SceneUpdates};
use glam::Vec3;
use std::sync::Arc;

pub mod config;
pub mod debug;
pub mod error;

/// Listener pose used by the acoustic renderer.
#[derive(Debug, Clone, Copy)]
pub struct ListenerPose {
    /// Listener position in world meters.
    pub position: Vec3,
    /// Listener right-ear direction in world space.
    pub right: Vec3,
}

/// Point sound source rendered through the current acoustic scene.
#[derive(Debug, Clone, Copy)]
pub struct PointSource {
    /// Source position in world meters.
    pub position: Vec3,
    /// Acoustic source energy before acoustic attenuation.
    pub energy: f32,
}

/// Scene-to-audio renderer that owns scene sync, IR caching, and convolution.
///
/// This type is intended to run on a control/audio worker thread, not directly
/// inside a realtime audio callback, because cache misses perform GPU tracing
/// and can allocate or block.
pub struct AcousticRenderer {
    gpu: Arc<VkBackend>,
    scene: SceneManager,
    pipeline: AcousticPipeline,
    cache: IrCache,
    cache_cfg: IrCacheConfig,
    convolver: PartitionedConvolver,
    active_voice: Option<VoiceId>,
    active_ir_id: Option<ImpulseResponseId>,
    active_query: Option<IrCacheQuery>,
    next_ir_id: ImpulseResponseId,
}

// impl AcousticRenderer {
//     /// Creates a renderer with its own Vulkan backend and empty scene.
//     pub fn new(cfg: EngineConfig) -> SoundResult<Self> {
//         let gpu = Arc::new(VkBackend::new()?);
//         let pipeline = AcousticPipeline::new(gpu.as_ref(), cfg.acoustics)?;
//         let convolver = PartitionedConvolver::new(gpu.clone())?;

//         Ok(Self {
//             gpu,
//             scene: SceneManager::new(),
//             pipeline,
//             cache: IrCache::new(cfg.cache),
//             cache_cfg: cfg.cache,
//             convolver,
//             active_voice: None,
//             active_ir_id: None,
//             active_query: None,
//             next_ir_id: 1,
//         })
//     }

//     /// Replaces the scene and invalidates active IR state.
//     pub fn load_scene(&mut self, scene: SceneDescription) -> SoundResult<()> {
//         self.scene.load(scene)?;
//         self.cache.clear();
//         self.active_query = None;
//         Ok(())
//     }

//     /// Applies scene mutations and invalidates active IR state when anything changed.
//     pub fn apply_scene_updates(&mut self, updates: SceneUpdates) -> SoundResult<()> {
//         if updates.updates.is_empty() {
//             return Ok(());
//         }
//         self.scene.apply_updates(updates)?;
//         self.active_query = None;
//         Ok(())
//     }

//     /// Renders one mono dry block through the cached acoustic IR into interleaved stereo output.
//     pub fn render_stereo_block(
//         &mut self,
//         source: PointSource,
//         listener: ListenerPose,
//         dry_block: &[f32],
//         output_interleaved: &mut [f32],
//     ) -> SoundResult<()> {
//         let voice_id = self.ensure_ir(source, listener)?;
//         self.convolver
//             .process_stereo_block(voice_id, dry_block, output_interleaved)
//     }

//     /// Returns the latest active IR id, if a block has been rendered.
//     pub fn active_ir_id(&self) -> Option<ImpulseResponseId> {
//         self.active_ir_id
//     }

//     /// Returns the number of snapshots currently retained by the IR cache.
//     pub fn cached_ir_count(&self) -> usize {
//         self.cache.len()
//     }

//     fn ensure_ir(&mut self, source: PointSource, listener: ListenerPose) -> SoundResult<VoiceId> {
//         let cache_query = IrCacheQuery {
//             scene_version: self.scene.version(),
//             query_id: 0,
//             source_position: source.position,
//             source_energy: source.energy,
//             listener_position: listener.position,
//             listener_right: listener.right,
//         };

//         if let (Some(active_voice), Some(active_query)) = (self.active_voice, self.active_query) {
//             if self.can_reuse_active(active_query, cache_query) {
//                 return Ok(active_voice);
//             }
//         }

//         let snapshot = match self.cache.get(cache_query) {
//             Some(snapshot) => snapshot,
//             None => {
//                 let snapshot = self.pipeline.build_ir(
//                     self.gpu.as_ref(),
//                     &mut self.scene,
//                     AcousticQuery {
//                         query_id: cache_query.query_id,
//                         source_position: source.position,
//                         listener_position: listener.position,
//                         listener_right: listener.right,
//                         source_energy: source.energy,
//                     },
//                 )?;
//                 self.cache.insert(cache_query, snapshot.clone());
//                 snapshot
//             }
//         };

//         self.activate_snapshot(cache_query, &snapshot)
//     }

//     fn activate_snapshot(&mut self, query: IrCacheQuery, snapshot: &IrSnapshot) -> SoundResult<VoiceId> {
//         if let Some(voice_id) = self.active_voice.take() {
//             self.convolver.stop_sound(voice_id)?;
//         }

//         let ir_id = self.next_ir_id;
//         self.next_ir_id = self.next_ir_id.saturating_add(1);
//         self.convolver
//             .register_impulse_response(ir_id, &snapshot.samples)?;
//         let voice_id = self.convolver.start_sound(ir_id)?;

//         if let Some(old_ir_id) = self.active_ir_id.replace(ir_id) {
//             self.convolver.forget_impulse_response(old_ir_id);
//         }
//         self.active_voice = Some(voice_id);
//         self.active_query = Some(query);

//         Ok(voice_id)
//     }

//     fn can_reuse_active(&self, active: IrCacheQuery, query: IrCacheQuery) -> bool {
//         active.scene_version == query.scene_version
//             && active.source_position.distance(query.source_position) <= self.cache_cfg.reuse_distance_meters
//             && (active.source_energy - query.source_energy).abs() <= f32::EPSILON
//             && active.listener_position.distance(query.listener_position) <= self.cache_cfg.reuse_distance_meters
//             && active
//                 .listener_right
//                 .normalize_or_zero()
//                 .dot(query.listener_right.normalize_or_zero())
//                 >= self.cache_cfg.listener_right_dot_threshold
//     }
// }
