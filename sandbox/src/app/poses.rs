//! Fixed receiver/source presets and camera pose conversion.
use super::geometry::to_glam;
use glam::Vec3;
use raylib::prelude::{Camera3D, Vector3};
use sound_engine::ListenerPose;

const CAMERA_LISTENER_POSITION_EPSILON_METERS: f32 = 0.01;
const CAMERA_LISTENER_ORIENTATION_DOT_THRESHOLD: f32 = 0.999_96;

/// Number of receiver positions available in the editable room.
pub(super) fn listener_position_count() -> usize {
    7
}

/// Returns a preset receiver position in world meters.
pub(super) fn listener_position(index: usize) -> Vector3 {
    match index % listener_position_count() {
        0 => Vector3::new(4.5, 0.45, 0.0),
        1 => Vector3::new(5.4, 0.45, -3.2),
        2 => Vector3::new(5.4, 0.45, 3.2),
        3 => Vector3::new(1.8, 0.45, 0.0),
        4 => Vector3::new(0.55, 0.45, -1.6),
        5 => Vector3::new(0.55, 0.45, 1.6),
        _ => Vector3::new(7.35, 0.45, 0.0),
    }
}

/// Returns the display label for a receiver preset.
pub(super) fn listener_position_name(index: usize) -> &'static str {
    match index % listener_position_count() {
        0 => "right room center",
        1 => "right room back",
        2 => "right room front",
        3 => "near doorway",
        4 => "near north jamb",
        5 => "near south jamb",
        _ => "near outer wall",
    }
}

/// Number of fixed receiver orientations.
pub(super) fn listener_rotation_count() -> usize {
    4
}

/// Returns the forward axis for a fixed receiver orientation.
pub(super) fn listener_forward(index: usize) -> Vec3 {
    match index % listener_rotation_count() {
        0 => -Vec3::X,
        1 => Vec3::Z,
        2 => Vec3::X,
        _ => -Vec3::Z,
    }
}

/// Returns the horizontal forward axis at the given rotation angle.
pub(super) fn listener_spin_forward(angle_radians: f32) -> Vec3 {
    Vec3::new(angle_radians.cos(), 0.0, angle_radians.sin())
}

/// Converts a horizontal forward axis into the continuous spin angle.
pub(super) fn angle_from_forward(forward: Vec3) -> f32 {
    forward.z.atan2(forward.x)
}

/// Returns the display label for a fixed receiver orientation.
pub(super) fn listener_rotation_name(index: usize) -> &'static str {
    match index % listener_rotation_count() {
        0 => "facing doorway",
        1 => "facing +Z",
        2 => "facing away",
        _ => "facing -Z",
    }
}

/// Number of source positions available in the editable room.
pub(super) fn source_position_count() -> usize {
    8
}

/// Returns a preset source position in world meters.
pub(super) fn source_position(index: usize) -> Vector3 {
    match index % source_position_count() {
        0 => Vector3::new(-4.5, 0.45, 0.0),
        1 => Vector3::new(-5.4, 0.45, -3.2),
        2 => Vector3::new(-5.4, 0.45, 3.2),
        3 => Vector3::new(-1.8, 0.45, 0.0),
        4 => Vector3::new(4.5, 0.45, 0.0),
        5 => Vector3::new(5.4, 0.45, -3.2),
        6 => Vector3::new(5.4, 0.45, 3.2),
        _ => Vector3::new(1.8, 0.45, 0.0),
    }
}

/// Returns the display label for a source preset.
pub(super) fn source_position_name(index: usize) -> &'static str {
    match index % source_position_count() {
        0 => "left room center",
        1 => "left room back",
        2 => "left room front",
        3 => "left near doorway",
        4 => "same room center",
        5 => "same room back",
        6 => "same room front",
        _ => "same room doorway",
    }
}

/// Derives the acoustic listener position and orientation from the camera.
pub(super) fn camera_listener_pose(camera: &Camera3D) -> ListenerPose {
    ListenerPose {
        position: to_glam(camera.position),
        forward: (to_glam(camera.target) - to_glam(camera.position)).normalize_or_zero(),
        up: to_glam(camera.up).normalize_or_zero(),
    }
}

/// Applies movement tolerances before requesting another acoustic update.
pub(super) fn listener_poses_match(left: ListenerPose, right: ListenerPose) -> bool {
    left.position.distance_squared(right.position)
        <= CAMERA_LISTENER_POSITION_EPSILON_METERS.powi(2)
        && left.forward.dot(right.forward) >= CAMERA_LISTENER_ORIENTATION_DOT_THRESHOLD
        && left.up.dot(right.up) >= CAMERA_LISTENER_ORIENTATION_DOT_THRESHOLD
}
