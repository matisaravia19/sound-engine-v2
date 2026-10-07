//! Source ownership, spatial voice creation, and acoustic refresh scheduling.

use super::EngineCore;
use crate::acoustics::{GpuIrSnapshot, ListenerPose, PointSource, validate_source};
use crate::auralization::{PlaySoundRequest, SoundId, VoiceId};
use crate::core::error::{SoundError, SoundResult};
use crate::engine::{PlaySpatialSoundRequest, SoundSourceId, validate_play_request};
use crate::gpu::memory::BufferHandle;
use std::sync::Arc;

/// A voice owns its fixed source or follows an application-managed source.
#[derive(Debug, Clone, Copy, PartialEq)]
enum VoiceSource {
    /// Fixed source retained only for this voice's lifetime.
    Fixed(PointSource),
    /// Movable source that must remain registered while this voice exists.
    Tracked(SoundSourceId),
}

/// Acoustic state associated with one active spatial voice.
pub(super) struct SpatialVoice {
    /// Source used when refreshing the voice IR.
    source: VoiceSource,
    /// Time-domain acoustic buffer whose filter was most recently assigned to the voice.
    ///
    /// Shared ownership keeps pointer identity stable even if the acoustic LRU
    /// evicts the entry while this voice remains active.
    acoustic_ir_buffer: Arc<BufferHandle>,
    /// Latest accepted endpoint/scene change awaiting an IR refresh.
    /// `None` means the latest response has been prepared for the voice.
    refresh_requested_at: Option<std::time::Instant>,
}

impl EngineCore {
    /// Sets the listener pose used by future IR queries.
    pub(in crate::engine) fn set_listener_pose(
        &mut self,
        listener: ListenerPose,
    ) -> SoundResult<()> {
        let listener = crate::acoustics::canonicalize_listener(listener)?;
        if self.listener == listener {
            return Ok(());
        }
        self.listener = listener;
        self.mark_all_spatial_voices_dirty();
        Ok(())
    }

    /// Creates a movable sound source and returns its public handle.
    pub(in crate::engine) fn create_sound_source(
        &mut self,
        source: PointSource,
    ) -> SoundResult<SoundSourceId> {
        validate_source(source)?;
        let id = SoundSourceId(crate::core::id::allocate_id(
            &mut self.next_sound_source_id,
            "Sound source",
        )?);
        self.sound_sources.insert(id, source);
        Ok(id)
    }

    /// Updates a tracked sound source and marks bound voices for IR refresh.
    pub(in crate::engine) fn update_sound_source(
        &mut self,
        id: SoundSourceId,
        source: PointSource,
    ) -> SoundResult<()> {
        validate_source(source)?;
        let current = self
            .sound_sources
            .get_mut(&id)
            .ok_or_else(|| SoundError::not_found(format!("Sound source {} does not exist", id)))?;
        if *current == source {
            return Ok(());
        }
        *current = source;
        self.mark_source_voices_dirty(id);
        Ok(())
    }

    /// Removes a tracked sound source when no active voice still references it.
    pub(in crate::engine) fn remove_sound_source(&mut self, id: SoundSourceId) -> SoundResult<()> {
        if self
            .spatial_voices
            .values()
            .any(|voice| voice.source == VoiceSource::Tracked(id))
        {
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

    /// Builds or reuses an acoustic IR and starts a convolved voice.
    pub(in crate::engine) fn play_sound(
        &mut self,
        request: PlaySpatialSoundRequest,
    ) -> SoundResult<VoiceId> {
        validate_play_request(request)?;
        validate_source(request.source)?;
        self.start_spatial_voice(
            request.sound_id,
            VoiceSource::Fixed(request.source),
            request.volume,
        )
    }

    /// Builds or reuses an acoustic IR from a tracked source and starts a voice.
    pub(in crate::engine) fn play_sound_from_source(
        &mut self,
        sound_id: SoundId,
        source_id: SoundSourceId,
        volume: f32,
    ) -> SoundResult<VoiceId> {
        if !volume.is_finite() {
            return Err(SoundError::invalid_argument("sound volume must be finite"));
        }
        self.start_spatial_voice(sound_id, VoiceSource::Tracked(source_id), volume)
    }

    /// Starts a voice after checking its source and dry asset, before allocating GPU IRs.
    fn start_spatial_voice(
        &mut self,
        sound_id: SoundId,
        voice_source: VoiceSource,
        volume: f32,
    ) -> SoundResult<VoiceId> {
        let source = self.resolve_source(voice_source)?;
        self.auralization.playable_sound(sound_id)?;
        let snapshot = self.gpu_impulse_response_for(source, self.listener)?;
        let acoustic_ir_buffer = snapshot.buffer().clone();
        let impulse_response = self
            .auralization
            .prepare_impulse_response_from_gpu(&snapshot)?;
        let voice_id = self.auralization.play_sound(PlaySoundRequest {
            sound_id,
            impulse_response,
            gain: volume,
        })?;
        self.spatial_voices.insert(
            voice_id,
            SpatialVoice {
                source: voice_source,
                acoustic_ir_buffer,
                refresh_requested_at: None,
            },
        );
        Ok(voice_id)
    }

    /// Rebuilds one dirty voice IR per block, rotating fairly between active voices.
    #[cfg(any(feature = "playback", test))]
    pub(super) fn advance_spatial_update(&mut self) -> SoundResult<(f64, f64)> {
        let dirty = || {
            self.spatial_voices
                .iter()
                .filter_map(|(&id, voice)| voice.refresh_requested_at.map(|_| id))
        };
        let Some(voice_id) = dirty()
            .filter(|&id| self.last_spatial_voice.is_none_or(|previous| id > previous))
            .min()
            .or_else(|| dirty().min())
        else {
            return Ok((0.0, 0.0));
        };
        self.last_spatial_voice = Some(voice_id);
        self.refresh_spatial_voice(voice_id)
    }

    fn gpu_impulse_response_for(
        &mut self,
        source: PointSource,
        listener: ListenerPose,
    ) -> SoundResult<GpuIrSnapshot> {
        self.acoustics
            .build_impulse_response_gpu(self.gpu.as_ref(), source, listener)
    }

    /// Resolves a voice's fixed source or the latest state of its tracked source.
    fn resolve_source(&self, source: VoiceSource) -> SoundResult<PointSource> {
        match source {
            VoiceSource::Fixed(source) => Ok(source),
            VoiceSource::Tracked(id) => self.sound_sources.get(&id).copied().ok_or_else(|| {
                SoundError::not_found(format!("Sound source {} does not exist", id))
            }),
        }
    }

    /// Marks every tracked spatial voice for refresh before the next rendered block.
    pub(super) fn mark_all_spatial_voices_dirty(&mut self) {
        let requested_at = std::time::Instant::now();
        for voice in self.spatial_voices.values_mut() {
            voice.refresh_requested_at = Some(requested_at);
        }
    }

    /// Marks voices bound to a source for refresh before the next rendered block.
    fn mark_source_voices_dirty(&mut self, source_id: SoundSourceId) {
        let requested_at = std::time::Instant::now();
        for voice in self.spatial_voices.values_mut() {
            if voice.source == VoiceSource::Tracked(source_id) {
                voice.refresh_requested_at = Some(requested_at);
            }
        }
    }

    /// Refreshes every pending voice before a synchronous render.
    pub(super) fn refresh_dirty_spatial_voices(&mut self) -> SoundResult<()> {
        let dirty_voice_ids: Vec<_> = self
            .spatial_voices
            .iter()
            .filter_map(|(&id, voice)| voice.refresh_requested_at.map(|_| id))
            .collect();
        for voice_id in dirty_voice_ids {
            self.refresh_spatial_voice(voice_id)?;
        }
        Ok(())
    }

    /// Refreshes one tracked voice and returns trace/preparation times in milliseconds.
    fn refresh_spatial_voice(&mut self, voice_id: VoiceId) -> SoundResult<(f64, f64)> {
        if !self.auralization.has_voice(voice_id) {
            self.spatial_voices.remove(&voice_id);
            return Ok((0.0, 0.0));
        }
        let voice = &self.spatial_voices[&voice_id];
        let Some(requested_at) = voice.refresh_requested_at else {
            return Ok((0.0, 0.0));
        };
        let source = self.resolve_source(voice.source)?;
        let started = std::time::Instant::now();
        let snapshot = self.gpu_impulse_response_for(source, self.listener)?;
        let trace_ms = started.elapsed().as_secs_f64() * 1000.0;

        let started = std::time::Instant::now();
        // Exclusive engine ownership keeps this voice and its request stable during tracing.
        let voice = self
            .spatial_voices
            .get_mut(&voice_id)
            .expect("Spatial voice remains owned during synchronous refresh");
        if !Arc::ptr_eq(&voice.acoustic_ir_buffer, snapshot.buffer()) {
            let impulse_response = self
                .auralization
                .prepare_impulse_response_from_gpu(&snapshot)?;
            self.auralization
                .update_voice_impulse_response(voice_id, impulse_response)?;
            voice.acoustic_ir_buffer = snapshot.buffer().clone();
            self.spatial_updates += 1;
            self.spatial_update_age_ms = requested_at.elapsed().as_secs_f64() * 1000.0;
        }
        voice.refresh_requested_at = None;
        Ok((trace_ms, started.elapsed().as_secs_f64() * 1000.0))
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/engine/direct/spatial.rs"]
mod tests;
