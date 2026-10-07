use super::*;

#[test]
fn invalid_options_fail_before_device_initialization() {
    for argument in [
        "--unknown",
        "--scene=",
        "--rays=0",
        "--volume=NaN",
        "--diffraction-radius=-1",
        "--camera-stress-seconds=no",
        "--camera-stress-seconds=NaN",
        "--camera-stress-seconds=0",
    ] {
        assert!(Options::parse([argument.to_owned()]).is_err(), "{argument}");
    }
}

#[test]
fn options_control_scene_audio_and_motion() {
    let options = Options::parse(
        [
            "--scene=wedge",
            "--rays=64",
            "--bounces=0",
            "--sound=signal.wav",
            "--volume=0.25",
            "--camera-stress-seconds=2.5",
        ]
        .map(str::to_owned),
    )
    .unwrap();
    let config = options.engine_config(48_000);
    assert_eq!(options.scene.as_deref(), Some("wedge"));
    assert_eq!(options.sound, PathBuf::from("signal.wav"));
    assert_eq!(options.volume, 0.25);
    assert_eq!(options.camera_stress, Some(Duration::from_millis(2500)));
    assert_eq!(config.acoustics.listener_ray_count, 64);
    assert_eq!(config.acoustics.max_bounces, 0);
}
