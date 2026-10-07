use super::*;
use glam::Quat;

#[test]
fn rejects_non_finite_acoustic_query_values() {
    let listener = ListenerPose::default();
    assert!(validate_source(PointSource::new(Vec3::new(f32::INFINITY, 0.0, 0.0))).is_err());

    let mut source = PointSource::new(Vec3::ZERO);
    source.amplitude = f32::NAN;
    assert!(validate_source(source).is_err());
    source.amplitude = -0.1;
    assert!(validate_source(source).is_err());

    assert!(
        canonicalize_listener(ListenerPose {
            position: Vec3::new(0.0, f32::NAN, 0.0),
            forward: -Vec3::Z,
            up: Vec3::Y,
        })
        .is_err()
    );
    assert!(
        canonicalize_listener(ListenerPose {
            position: Vec3::ZERO,
            forward: Vec3::new(0.0, 0.0, f32::INFINITY),
            up: Vec3::Y,
        })
        .is_err()
    );
    assert!(canonicalize_listener(listener).is_ok());
    assert!(canonicalize_listener(ListenerPose::new(Vec3::ZERO, Vec3::ZERO, Vec3::Y)).is_err());
    assert!(canonicalize_listener(ListenerPose::new(Vec3::ZERO, Vec3::Y, Vec3::Y)).is_err());

    let canonical = canonicalize_listener(ListenerPose::new(
        Vec3::ZERO,
        Vec3::new(0.0, 0.0, -2.0),
        Vec3::new(0.0, 3.0, -0.5),
    ))
    .unwrap();
    assert!((canonical.forward.length() - 1.0).abs() < 1.0e-6);
    assert!((canonical.up.length() - 1.0).abs() < 1.0e-6);
    assert!(canonical.forward.dot(canonical.up).abs() < 1.0e-6);
}

#[test]
fn listener_relative_lateral_direction_is_rotation_invariant() {
    let listener = canonicalize_listener(ListenerPose::default()).unwrap();
    let arrival = Vec3::new(0.7, 0.2, -0.4).normalize();
    let lateral = listener.right().dot(arrival);

    let rotation = Quat::from_rotation_y(0.8) * Quat::from_rotation_x(-0.35);
    let rotated = canonicalize_listener(ListenerPose::new(
        rotation * listener.position,
        rotation * listener.forward,
        rotation * listener.up,
    ))
    .unwrap();
    let rotated_lateral = rotated.right().dot(rotation * arrival);

    assert!((rotated_lateral - lateral).abs() < 1.0e-6);
}
