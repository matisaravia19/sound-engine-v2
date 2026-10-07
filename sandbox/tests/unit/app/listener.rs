use super::*;
use raylib::prelude::Vector3;

#[test]
fn spinning_starts_at_the_preset_and_fixed_controls_restore_preset_orientation() {
    let mut listener = ListenerController::default();
    assert!(!listener.advance_spin(4.0));
    assert_eq!(listener.forward(), Vec3::NEG_X);

    listener.toggle_spin();
    assert!(listener.is_spinning());
    assert!((listener.forward() - Vec3::NEG_X).length() < 1e-6);
    assert!(listener.advance_spin(4.0));
    assert!((listener.forward() - Vec3::NEG_Z).length() < 1e-6);
    listener.toggle_spin();
    assert_eq!(listener.forward(), Vec3::NEG_X);

    listener.next_orientation();
    assert_eq!(listener.forward(), Vec3::Z);
    listener.toggle_spin();
    listener.advance_spin(4.0);
    listener.next_orientation();
    assert!(!listener.is_spinning());
    assert_eq!(listener.forward(), Vec3::X);
    listener.next_orientation();
    assert_eq!(listener.forward(), Vec3::NEG_Z);
    listener.next_orientation();
    assert_eq!(listener.forward(), Vec3::NEG_X);
}

#[test]
fn camera_updates_observe_both_time_and_movement_tolerances() {
    let mut listener = ListenerController::default();
    listener.set_mode(ListenerMode::Camera);
    let now = Instant::now();
    let initial = listener
        .pending_camera_pose(&camera_at(0.0), now, false)
        .unwrap();
    listener.camera_synced(initial, now);

    assert!(
        listener
            .pending_camera_pose(&camera_at(0.02), now + Duration::from_millis(49), false)
            .is_none()
    );
    assert!(
        listener
            .pending_camera_pose(&camera_at(0.02), now + Duration::from_millis(50), false)
            .is_some()
    );
    assert!(
        listener
            .pending_camera_pose(&camera_at(0.005), now + Duration::from_secs(1), false)
            .is_none()
    );

    let mut rotated = camera_at(0.0);
    rotated.target = Vector3::new(1.0, 0.0, 0.0);
    assert!(
        listener
            .pending_camera_pose(&rotated, now + Duration::from_millis(50), false)
            .is_some()
    );
}

#[test]
fn unaccepted_updates_remain_pending_and_explicit_controls_only_bypass_the_interval() {
    let mut listener = ListenerController::default();
    listener.set_mode(ListenerMode::Camera);
    let now = Instant::now();
    let initial = listener
        .pending_camera_pose(&camera_at(0.0), now, false)
        .unwrap();
    // No acceptance means the audio update failed and the next frame must retry.
    assert!(
        listener
            .pending_camera_pose(&camera_at(0.0), now, false)
            .is_some()
    );
    listener.camera_synced(initial, now);
    assert!(
        listener
            .pending_camera_pose(&camera_at(0.0), now, true)
            .is_none()
    );
    assert!(
        listener
            .pending_camera_pose(&camera_at(0.02), now, true)
            .is_some()
    );
    assert!(
        listener
            .pending_camera_pose(&camera_at(0.02), now, false)
            .is_none()
    );
}

#[test]
fn mode_switching_selects_the_right_pose_and_invalidates_camera_history() {
    let mut listener = ListenerController::default();
    let camera = camera_at(2.0);
    let marker_position = Vec3::new(4.0, 3.0, 2.0);
    let marker = listener.active_pose(marker_position, &camera);
    assert_eq!(marker.position, marker_position);
    assert_eq!(marker.forward, Vec3::NEG_X);
    assert_eq!(marker.up, Vec3::Y);

    listener.toggle_mode();
    let now = Instant::now();
    let followed = listener.active_pose(marker_position, &camera);
    assert_eq!(followed.position, Vec3::new(2.0, 0.0, 0.0));
    assert_eq!(followed.forward, Vec3::NEG_Z);
    listener.camera_synced(followed, now);
    assert!(listener.pending_camera_pose(&camera, now, true).is_none());

    listener.toggle_mode();
    assert!(listener.pending_camera_pose(&camera, now, true).is_none());
    assert_eq!(
        listener.active_pose(marker_position, &camera).position,
        marker_position
    );
    listener.toggle_mode();
    assert!(listener.pending_camera_pose(&camera, now, false).is_some());
}

fn camera_at(x: f32) -> Camera3D {
    Camera3D::perspective(
        Vector3::new(x, 0.0, 0.0),
        Vector3::new(x, 0.0, -1.0),
        Vector3::new(0.0, 1.0, 0.0),
        60.0,
    )
}
