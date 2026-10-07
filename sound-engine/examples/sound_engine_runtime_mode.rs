//! Play generated spatial tones through the background audio worker.
//! Requires Vulkan and an audio output device; no external WAV files are needed.

#[path = "support/demo.rs"]
mod demo;

use glam::{Vec3, vec3};
use sound_engine::core::error::SoundResult;
use sound_engine::playback::PlaybackConfig;
use sound_engine::{ListenerPose, PlaySpatialSoundRequest, SoundEngine};
use std::thread;
use std::time::Duration;

fn main() -> SoundResult<()> {
    let mut engine = SoundEngine::new(demo::engine_config())?;
    engine.load_scene(demo::box_room_scene())?;
    engine.set_listener_pose(ListenerPose::new(vec3(2.0, 0.0, -1.0), -Vec3::X, Vec3::Z))?;
    let low_tone = engine.insert_sound(demo::sine_tone(440.0, 0.75)?)?;
    let high_tone = engine.insert_sound(demo::sine_tone(660.0, 0.75)?)?;
    engine.start_sound_player(PlaybackConfig::default())?;

    for (sound_id, position) in [
        (low_tone, vec3(-3.0, 0.5, 2.0)),
        (high_tone, vec3(-2.5, -0.75, -3.0)),
    ] {
        let mut request = PlaySpatialSoundRequest::new(sound_id, position);
        request.source.amplitude = 1_000.0;
        request.volume = 0.6;
        let voice = engine.play_sound(request)?;
        println!("started voice {voice}");
        thread::sleep(Duration::from_millis(450));
    }
    while engine.has_active_voices() && engine.is_sound_player_running() {
        thread::sleep(Duration::from_millis(20));
    }
    // Voice completion precedes device consumption of the bounded output queue.
    thread::sleep(Duration::from_millis(250));
    engine.stop_sound_player()
}
