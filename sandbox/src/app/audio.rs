//! Initialized audio resources shared by the playground's controls.
use sound_engine::auralization::{SoundId, VoiceId};
use sound_engine::core::error::SoundResult;
use sound_engine::playback::PlaybackConfig;
use sound_engine::{ListenerPose, PointSource, SoundEngine, SoundSourceId};
use std::path::Path;

/// Owns the running engine, dry asset, and tracked source used by the playground.
/// Construction completes before controls become available; dropping stops playback.
pub(super) struct AudioSession {
    /// Engine used by scene and listener controls; its worker owns GPU rendering.
    pub(super) engine: SoundEngine,
    sound: SoundId,
    source: SoundSourceId,
    gain: f32,
}

impl AudioSession {
    /// Loads playback resources into an engine with its initial scene and listener set.
    pub(super) fn new(
        mut engine: SoundEngine,
        source: PointSource,
        sound_path: &Path,
        gain: f32,
    ) -> SoundResult<Self> {
        let source = engine.create_sound_source(source)?;
        let sound = engine.load_wav_sound(sound_path)?;
        engine.start_sound_player(PlaybackConfig {
            prefill_blocks: 3,
            max_queue_seconds: 0.064,
        })?;
        Ok(Self {
            engine,
            sound,
            source,
            gain,
        })
    }

    /// Synchronizes the selected listener before starting the shared dry sound.
    pub(super) fn play(&mut self, listener: ListenerPose) -> SoundResult<VoiceId> {
        self.engine.set_listener_pose(listener)?;
        self.engine
            .play_sound_from_source(self.sound, self.source, self.gain)
    }

    /// Moves the tracked source so every active voice follows the same position.
    pub(super) fn update_source(&mut self, source: PointSource) -> SoundResult<()> {
        self.engine.update_sound_source(self.source, source)
    }
}
