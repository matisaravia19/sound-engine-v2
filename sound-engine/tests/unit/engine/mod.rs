use super::*;
use glam::Vec3;

#[test]
fn play_request_defaults_to_unit_amplitude_and_volume() {
    let request = PlaySpatialSoundRequest::new(SoundId(7), Vec3::new(1.0, 2.0, 3.0));

    assert_eq!(request.sound_id, SoundId(7));
    assert_eq!(request.source.position, Vec3::new(1.0, 2.0, 3.0));
    assert_eq!(request.source.amplitude, 1.0);
    assert_eq!(request.volume, 1.0);
}

#[test]
fn rejects_non_finite_playback_volume() {
    let mut request = PlaySpatialSoundRequest::new(SoundId(1), Vec3::ZERO);
    request.volume = f32::NAN;
    assert!(validate_play_request(request).is_err());
}
