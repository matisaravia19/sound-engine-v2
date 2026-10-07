//! Direct, synchronous engine implementation.

mod spatial;

use super::{ListenerPose, PointSource, SoundSourceId};
use crate::acoustics::{AcousticsEngine, IrSnapshot};
use crate::auralization::{AuralizationEngine, SoundAsset, SoundId, VoiceId};
use crate::core::config::EngineConfig;
use crate::core::error::SoundResult;
use crate::gpu::backend::VkBackend;
use crate::scene::{SceneDescription, SceneUpdates, SceneVersion};
use spatial::SpatialVoice;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Private owner of GPU, scene, acoustic cache, assets, voices, and block rendering.
pub(super) struct EngineCore {
    /// Scene-backed acoustic IR builder and cache.
    acoustics: AcousticsEngine,
    /// Asset bank, active voices, and convolver used to render output blocks.
    auralization: AuralizationEngine,
    /// Movable acoustic sources addressable by public handles.
    sound_sources: HashMap<SoundSourceId, PointSource>,
    /// Spatial metadata for voices that should react to acoustic updates.
    spatial_voices: HashMap<VoiceId, SpatialVoice>,
    /// Round-robin cursor prevents a moving voice from starving other dirty voices.
    #[cfg(any(feature = "playback", test))]
    last_spatial_voice: Option<VoiceId>,
    /// Number of prepared spatial IR updates assigned to active voices.
    spatial_updates: u64,
    /// Age of the latest endpoint request when its prepared filter is assigned.
    spatial_update_age_ms: f64,
    /// Listener pose used by spatial play requests that do not supply one directly.
    listener: ListenerPose,
    /// Monotonic id assigned to tracked sound sources.
    next_sound_source_id: u64,
    /// Shared Vulkan backend kept after every dependent GPU resource.
    ///
    /// Rust drops fields in declaration order, so this owner must remain last.
    gpu: Arc<VkBackend>,
}

/// Timing and spatial-update counters captured after one realtime render.
#[cfg(any(feature = "playback", test))]
pub(super) struct RenderMetrics {
    /// Synchronous acoustic trace time for this block, in milliseconds.
    pub(super) trace_ms: f64,
    /// IR transform and voice assignment time for this block, in milliseconds.
    pub(super) ir_prepare_ms: f64,
    /// Voice convolution and finished-voice cleanup time, in milliseconds.
    pub(super) convolution_ms: f64,
    /// Prepared spatial IR updates assigned since engine creation.
    pub(super) spatial_updates: u64,
    /// Latest update's request-to-assignment age, excluding its crossfade.
    pub(super) spatial_update_age_ms: f64,
    /// Whether the scheduler deferred spatial work to repay earlier render cost.
    pub(super) spatial_deferred: bool,
}

impl EngineCore {
    /// Creates a direct scalar engine core with an empty scene and sound bank.
    pub(super) fn new(config: EngineConfig) -> SoundResult<Self> {
        config.validate()?;
        let gpu = Arc::new(VkBackend::new()?);
        let acoustics = AcousticsEngine::new(
            gpu.as_ref(),
            crate::acoustics::AcousticPipelineConfig {
                sound: config.sound,
                tracing: config.acoustics,
                binaural: config.binaural,
            },
            config.cache,
        )?;
        let auralization = AuralizationEngine::new(gpu.clone(), config.sound, config.auralization)?;

        Ok(Self {
            acoustics,
            auralization,
            sound_sources: HashMap::new(),
            spatial_voices: HashMap::new(),
            #[cfg(any(feature = "playback", test))]
            last_spatial_voice: None,
            spatial_updates: 0,
            spatial_update_age_ms: 0.0,
            listener: ListenerPose::default(),
            next_sound_source_id: 1,
            gpu,
        })
    }

    /// Replaces the scene and clears IR cache entries tied to prior geometry.
    pub(super) fn load_scene(&mut self, scene: SceneDescription) -> SoundResult<SceneVersion> {
        let version = self.acoustics.load_scene(scene)?;
        self.mark_all_spatial_voices_dirty();
        Ok(version)
    }

    /// Applies scene edits and clears cached IRs when scene contents changed.
    pub(super) fn apply_scene_updates(
        &mut self,
        updates: SceneUpdates,
    ) -> SoundResult<SceneVersion> {
        let changed = !updates.updates.is_empty();
        let version = self.acoustics.apply_scene_updates(updates)?;
        if changed {
            self.mark_all_spatial_voices_dirty();
        }
        Ok(version)
    }

    /// Loads a WAV asset into the auralizer.
    pub(super) fn load_wav_sound(&mut self, path: impl AsRef<Path>) -> SoundResult<SoundId> {
        self.auralization.load_wav_sound(path)
    }

    /// Inserts a decoded mono asset into the auralizer.
    pub(super) fn insert_sound(&mut self, asset: SoundAsset) -> SoundResult<SoundId> {
        self.auralization.insert_sound(asset)
    }

    /// Builds or reuses the raw acoustic impulse response for a source/listener pair.
    pub(super) fn build_impulse_response(
        &mut self,
        source: PointSource,
        listener: ListenerPose,
    ) -> SoundResult<IrSnapshot> {
        self.acoustics
            .build_impulse_response(self.gpu.as_ref(), source, listener)
    }

    /// Stops an active voice if it exists.
    pub(super) fn stop_voice(&mut self, voice_id: VoiceId) {
        self.auralization.stop_voice(voice_id);
        self.spatial_voices.remove(&voice_id);
    }

    /// Renders one interleaved block through the auralizer.
    pub(super) fn render_block(&mut self, output_block: &mut [f32]) -> SoundResult<()> {
        self.auralization.validate_output_block(output_block)?;
        self.refresh_dirty_spatial_voices()?;
        self.auralization.render_block(output_block)?;
        self.cleanup_finished_spatial_voices();
        Ok(())
    }

    /// Rebuilds at most one spatial IR per block before rendering audio.
    /// The previous IR crossfades into the replacement through the convolver.
    /// With `advance_spatial=false`, dirty voices retain their current IR.
    #[cfg(any(feature = "playback", test))]
    pub(super) fn render_realtime_block(
        &mut self,
        output_block: &mut [f32],
        advance_spatial: bool,
    ) -> SoundResult<RenderMetrics> {
        self.auralization.validate_output_block(output_block)?;
        let (trace_ms, prepare_ms) = if advance_spatial {
            self.advance_spatial_update()?
        } else {
            (0.0, 0.0)
        };
        let started = std::time::Instant::now();
        self.auralization.render_block(output_block)?;
        self.cleanup_finished_spatial_voices();
        Ok(RenderMetrics {
            trace_ms,
            ir_prepare_ms: prepare_ms,
            convolution_ms: started.elapsed().as_secs_f64() * 1000.0,
            spatial_updates: self.spatial_updates,
            spatial_update_age_ms: self.spatial_update_age_ms,
            spatial_deferred: !advance_spatial,
        })
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

    /// Releases spatial metadata as soon as the auralizer finishes its voice.
    fn cleanup_finished_spatial_voices(&mut self) {
        self.spatial_voices
            .retain(|voice_id, _| self.auralization.has_voice(*voice_id));
    }
}
