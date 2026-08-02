//! Direct, synchronous engine implementation.

use super::{ListenerPose, PlaySpatialSoundRequest, PointSource, SoundSourceId, validate_play_request};
use crate::acoustics::{AcousticsEngine, GpuIrSnapshot, IrSnapshot};
use crate::auralization::{AuralizationEngine, ImpulseResponseId, PlaySoundRequest, SoundAsset, SoundId, VoiceId};
use crate::core::config::EngineConfig;
use crate::core::error::{SoundError, SoundResult};
use crate::gpu::backend::VkBackend;
use crate::gpu::memory::BufferHandle;
use crate::scene::{SceneDescription, SceneUpdates, SceneVersion};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Private owner of GPU, scene, acoustic cache, assets, voices, and block rendering.
pub(super) struct EngineCore {
    /// Scene-backed acoustic IR builder and cache.
    acoustics: AcousticsEngine,
    /// Asset bank, active voices, and convolver used to render output blocks.
    auralization: AuralizationEngine,
    /// Acoustic sources addressable by public handles or one-shot voices.
    sound_sources: HashMap<SoundSourceId, SoundSourceRecord>,
    /// Spatial metadata for voices that should react to acoustic updates.
    spatial_voices: HashMap<VoiceId, SpatialVoice>,
    /// Listener pose used by spatial play requests that do not supply one directly.
    listener: ListenerPose,
    /// Monotonic id assigned to tracked sound sources.
    next_sound_source_id: SoundSourceId,
    /// Monotonic id assigned to impulse responses registered with the auralizer.
    next_impulse_response_id: ImpulseResponseId,
    /// Shared Vulkan backend kept after every dependent GPU resource.
    ///
    /// Rust drops fields in declaration order, so this owner must remain last.
    gpu: Arc<VkBackend>,
}

/// Stored source transform and ownership policy.
#[derive(Debug, Clone, Copy)]
struct SoundSourceRecord {
    /// Current source state used for acoustic IR queries.
    source: PointSource,
    /// Whether this source should be removed when its owning voice ends.
    internal: bool,
}

/// Acoustic state associated with one active spatial voice.
struct SpatialVoice {
    /// Source handle used when refreshing the voice IR.
    source_id: SoundSourceId,
    /// Currently registered IR id assigned to the voice.
    impulse_response_id: ImpulseResponseId,
    /// Time-domain acoustic buffer whose transformed IR is currently registered.
    ///
    /// Shared ownership keeps pointer identity stable even if the acoustic LRU
    /// evicts the entry while this voice remains active.
    acoustic_ir_buffer: Arc<BufferHandle>,
    /// Whether the voice should refresh its IR before the next rendered block.
    dirty: bool,
}

impl EngineCore {
    /// Creates a direct engine core with an empty scene and sound bank.
    pub(super) fn new(config: EngineConfig) -> SoundResult<Self> {
        let gpu = Arc::new(VkBackend::new()?);
        let acoustics = AcousticsEngine::new(gpu.as_ref(), config)?;
        let auralization = AuralizationEngine::new(gpu.clone(), config)?;

        Ok(Self {
            acoustics,
            auralization,
            sound_sources: HashMap::new(),
            spatial_voices: HashMap::new(),
            listener: ListenerPose::default(),
            next_sound_source_id: 1,
            next_impulse_response_id: 1,
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
    pub(super) fn apply_scene_updates(&mut self, updates: SceneUpdates) -> SoundResult<SceneVersion> {
        let changed = !updates.updates.is_empty();
        let version = self.acoustics.apply_scene_updates(updates)?;
        if changed {
            self.mark_all_spatial_voices_dirty();
        }
        Ok(version)
    }

    /// Sets the listener pose used by future IR queries.
    pub(super) fn set_listener_pose(&mut self, listener: ListenerPose) -> SoundResult<()> {
        let listener = crate::acoustics::canonicalize_listener(listener)?;
        if self.listener.position == listener.position
            && self.listener.forward == listener.forward
            && self.listener.up == listener.up
        {
            return Ok(());
        }
        self.listener = listener;
        self.mark_all_spatial_voices_dirty();
        Ok(())
    }

    /// Creates a movable sound source and returns its public handle.
    pub(super) fn create_sound_source(&mut self, source: PointSource) -> SoundResult<SoundSourceId> {
        crate::acoustics::validate_source(source)?;
        Ok(self.insert_sound_source(source, false))
    }

    /// Updates a tracked sound source and marks bound voices for IR refresh.
    pub(super) fn update_sound_source(&mut self, id: SoundSourceId, source: PointSource) -> SoundResult<()> {
        crate::acoustics::validate_source(source)?;
        let record = self
            .sound_sources
            .get_mut(&id)
            .ok_or_else(|| SoundError::not_found(format!("Sound source {} does not exist", id)))?;
        record.source = source;
        self.mark_source_voices_dirty(id);
        Ok(())
    }

    /// Removes a tracked sound source when no active voice still references it.
    pub(super) fn remove_sound_source(&mut self, id: SoundSourceId) -> SoundResult<()> {
        if self.spatial_voices.values().any(|voice| voice.source_id == id) {
            return Err(SoundError::invalid_state(format!(
                "Sound source {} is still referenced by an active voice",
                id
            )));
        }
        self.sound_sources
            .remove(&id)
            .ok_or_else(|| SoundError::not_found(format!("Sound source {} does not exist", id)))?;
        Ok(())
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
        crate::acoustics::validate_source(request.source)?;
        let source_id = self.insert_sound_source(request.source, true);
        match self.play_sound_from_source(request.sound_id, source_id, request.volume) {
            Ok(voice_id) => Ok(voice_id),
            Err(error) => {
                self.sound_sources.remove(&source_id);
                Err(error)
            }
        }
    }

    /// Builds or reuses an acoustic IR from a tracked source and starts a voice.
    pub(super) fn play_sound_from_source(
        &mut self,
        sound_id: SoundId,
        source_id: SoundSourceId,
        volume: f32,
    ) -> SoundResult<VoiceId> {
        if !volume.is_finite() {
            return Err(SoundError::invalid_argument("sound volume must be finite"));
        }
        let source = self.source(source_id)?;
        let snapshot = self.gpu_impulse_response_for(source, self.listener)?;
        let acoustic_ir_buffer = snapshot
            .buffer
            .as_ref()
            .ok_or_else(|| SoundError::invalid_state("acoustic GPU IR has no buffer"))?
            .clone();
        let impulse_response_id = self.register_gpu_impulse_response(&snapshot)?;
        let voice_id = match self.auralization.play_sound(PlaySoundRequest {
            sound_id,
            impulse_response_id,
            gain: volume,
        }) {
            Ok(voice_id) => voice_id,
            Err(error) => {
                self.auralization.forget_impulse_response(impulse_response_id);
                return Err(error);
            }
        };
        self.spatial_voices.insert(
            voice_id,
            SpatialVoice {
                source_id,
                impulse_response_id,
                acoustic_ir_buffer,
                dirty: false,
            },
        );
        Ok(voice_id)
    }

    /// Builds or reuses the raw acoustic impulse response for a source/listener pair.
    pub(super) fn build_impulse_response(
        &mut self,
        source: PointSource,
        listener: ListenerPose,
    ) -> SoundResult<IrSnapshot> {
        self.impulse_response_for(source, listener)
    }

    /// Stops an active voice if it exists.
    pub(super) fn stop_voice(&mut self, voice_id: VoiceId) {
        self.auralization.stop_voice(voice_id);
        self.remove_spatial_voice(voice_id);
    }

    /// Renders one interleaved block through the auralizer.
    pub(super) fn render_block(&mut self, output_block: &mut [f32]) -> SoundResult<()> {
        self.refresh_dirty_spatial_voices()?;
        self.auralization.render_block(output_block)?;
        self.cleanup_finished_spatial_voices();
        Ok(())
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
        self.acoustics
            .build_impulse_response(self.gpu.as_ref(), source, listener)
    }

    fn gpu_impulse_response_for(&mut self, source: PointSource, listener: ListenerPose) -> SoundResult<GpuIrSnapshot> {
        self.acoustics
            .build_impulse_response_gpu(self.gpu.as_ref(), source, listener)
    }

    /// Registers a GPU-resident IR snapshot for use by the auralizer and returns its id.
    fn register_gpu_impulse_response(&mut self, snapshot: &GpuIrSnapshot) -> SoundResult<ImpulseResponseId> {
        let impulse_response_id = self.next_ir_id();
        self.auralization
            .register_impulse_response_from_gpu_ir(impulse_response_id, snapshot)?;
        Ok(impulse_response_id)
    }

    /// Inserts a sound source with the requested ownership policy.
    fn insert_sound_source(&mut self, source: PointSource, internal: bool) -> SoundSourceId {
        let id = self.next_source_id();
        self.sound_sources.insert(id, SoundSourceRecord { source, internal });
        id
    }

    /// Returns the current source record for an id.
    fn source(&self, id: SoundSourceId) -> SoundResult<PointSource> {
        self.sound_sources
            .get(&id)
            .map(|record| record.source)
            .ok_or_else(|| SoundError::not_found(format!("Sound source {} does not exist", id)))
    }

    /// Marks every tracked spatial voice for refresh before the next rendered block.
    fn mark_all_spatial_voices_dirty(&mut self) {
        for voice in self.spatial_voices.values_mut() {
            voice.dirty = true;
        }
    }

    /// Marks voices bound to a source for refresh before the next rendered block.
    fn mark_source_voices_dirty(&mut self, source_id: SoundSourceId) {
        for voice in self.spatial_voices.values_mut() {
            if voice.source_id == source_id {
                voice.dirty = true;
            }
        }
    }

    /// Refreshes dirty active voices with IRs for their current source/listener state.
    fn refresh_dirty_spatial_voices(&mut self) -> SoundResult<()> {
        let dirty_voice_ids: Vec<_> = self
            .spatial_voices
            .iter()
            .filter_map(|(voice_id, voice)| voice.dirty.then_some(*voice_id))
            .collect();

        for voice_id in dirty_voice_ids {
            if !self.auralization.has_voice(voice_id) {
                self.remove_spatial_voice(voice_id);
                continue;
            }

            let source_id = self
                .spatial_voices
                .get(&voice_id)
                .map(|voice| voice.source_id)
                .ok_or_else(|| SoundError::not_found(format!("Unknown spatial voice id {}", voice_id)))?;
            let source = self.source(source_id)?;
            let snapshot = self.gpu_impulse_response_for(source, self.listener)?;
            let next_acoustic_ir_buffer = snapshot
                .buffer
                .as_ref()
                .ok_or_else(|| SoundError::invalid_state("acoustic GPU IR has no buffer"))?
                .clone();
            let reuses_registered_ir = self
                .spatial_voices
                .get(&voice_id)
                .is_some_and(|voice| Arc::ptr_eq(&voice.acoustic_ir_buffer, &next_acoustic_ir_buffer));
            if reuses_registered_ir {
                if let Some(voice) = self.spatial_voices.get_mut(&voice_id) {
                    voice.dirty = false;
                }
                continue;
            }

            let next_ir_id = self.register_gpu_impulse_response(&snapshot)?;
            if let Err(error) = self.auralization.update_voice_impulse_response(voice_id, next_ir_id) {
                self.auralization.forget_impulse_response(next_ir_id);
                return Err(error);
            }

            if let Some(voice) = self.spatial_voices.get_mut(&voice_id) {
                let previous_ir_id = voice.impulse_response_id;
                voice.impulse_response_id = next_ir_id;
                voice.acoustic_ir_buffer = next_acoustic_ir_buffer;
                voice.dirty = false;
                self.auralization.forget_impulse_response(previous_ir_id);
            }
        }

        Ok(())
    }

    /// Removes tracking state for voices that the auralizer has already finished.
    fn cleanup_finished_spatial_voices(&mut self) {
        let finished_voice_ids: Vec<_> = self
            .spatial_voices
            .keys()
            .copied()
            .filter(|voice_id| !self.auralization.has_voice(*voice_id))
            .collect();

        for voice_id in finished_voice_ids {
            self.remove_spatial_voice(voice_id);
        }
    }

    /// Removes spatial tracking and any internal source owned only by this voice.
    fn remove_spatial_voice(&mut self, voice_id: VoiceId) {
        let Some(voice) = self.spatial_voices.remove(&voice_id) else {
            return;
        };
        self.auralization.forget_impulse_response(voice.impulse_response_id);

        let should_remove_source = self
            .sound_sources
            .get(&voice.source_id)
            .map(|record| record.internal)
            .unwrap_or(false)
            && !self
                .spatial_voices
                .values()
                .any(|other_voice| other_voice.source_id == voice.source_id);
        if should_remove_source {
            self.sound_sources.remove(&voice.source_id);
        }
    }

    /// Returns a fresh sound source id.
    fn next_source_id(&mut self) -> SoundSourceId {
        let id = self.next_sound_source_id;
        self.next_sound_source_id = self.next_sound_source_id.saturating_add(1);
        id
    }

    /// Returns a fresh auralization impulse-response id.
    fn next_ir_id(&mut self) -> ImpulseResponseId {
        let id = self.next_impulse_response_id;
        self.next_impulse_response_id = self.next_impulse_response_id.saturating_add(1);
        id
    }
}
