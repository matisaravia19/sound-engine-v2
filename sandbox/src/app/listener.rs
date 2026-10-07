//! Listener orientation, camera-follow mode, and accepted camera-update state.

use super::geometry::to_glam;
use glam::Vec3;
use raylib::prelude::Camera3D;
use sound_engine::ListenerPose;
use std::time::{Duration, Instant};

const SPIN_RADIANS_PER_SECOND: f32 = std::f32::consts::PI * 0.125;
const CAMERA_UPDATE_INTERVAL: Duration = Duration::from_millis(50);
const CAMERA_POSITION_EPSILON_METERS: f32 = 0.01;
const CAMERA_ORIENTATION_DOT_THRESHOLD: f32 = 0.999_96;

struct Orientation {
    label: &'static str,
    forward: Vec3,
}

const ORIENTATIONS: [Orientation; 4] = [
    Orientation {
        label: "facing doorway",
        forward: Vec3::NEG_X,
    },
    Orientation {
        label: "facing +Z",
        forward: Vec3::Z,
    },
    Orientation {
        label: "facing away",
        forward: Vec3::X,
    },
    Orientation {
        label: "facing -Z",
        forward: Vec3::NEG_Z,
    },
];

/// Selects the world-space marker or the display camera as the acoustic receiver.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(super) enum ListenerMode {
    /// Uses the red sphere's selected position and orientation.
    #[default]
    RedSphere,
    /// Follows the display camera, subject to movement and timing tolerances.
    Camera,
}

impl ListenerMode {
    /// Returns the receiver name shown in the overlay.
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::RedSphere => "red sphere",
            Self::Camera => "camera",
        }
    }
}

/// Keeps receiver controls independent of scene selection, rendering, and audio I/O.
#[derive(Default)]
pub(super) struct ListenerController {
    mode: ListenerMode,
    orientation: usize,
    spin_angle: Option<f32>,
    last_camera_update: Option<CameraUpdate>,
}

struct CameraUpdate {
    pose: ListenerPose,
    accepted_at: Instant,
}

impl ListenerController {
    /// Returns the receiver currently driving audio.
    pub(super) fn mode(&self) -> ListenerMode {
        self.mode
    }

    /// Changes receiver mode and requires a fresh camera synchronization after switching.
    pub(super) fn set_mode(&mut self, mode: ListenerMode) {
        if self.mode != mode {
            self.mode = mode;
            self.last_camera_update = None;
        }
    }

    /// Switches between the marker and camera without changing marker orientation.
    pub(super) fn toggle_mode(&mut self) {
        self.set_mode(match self.mode {
            ListenerMode::RedSphere => ListenerMode::Camera,
            ListenerMode::Camera => ListenerMode::RedSphere,
        });
    }

    /// Selects the next fixed orientation and stops continuous rotation.
    pub(super) fn next_orientation(&mut self) {
        self.orientation = (self.orientation + 1) % ORIENTATIONS.len();
        self.spin_angle = None;
    }

    /// Starts spinning from the selected preset, or returns to that preset when stopped.
    pub(super) fn toggle_spin(&mut self) {
        self.spin_angle = if self.spin_angle.is_some() {
            None
        } else {
            let forward = ORIENTATIONS[self.orientation].forward;
            Some(forward.z.atan2(forward.x))
        };
    }

    /// Advances a spinning marker; returns false when spin is disabled.
    pub(super) fn advance_spin(&mut self, frame_seconds: f32) -> bool {
        let Some(angle) = &mut self.spin_angle else {
            return false;
        };
        *angle =
            (*angle + SPIN_RADIANS_PER_SECOND * frame_seconds).rem_euclid(std::f32::consts::TAU);
        true
    }

    /// Returns whether the marker is rotating continuously.
    pub(super) fn is_spinning(&self) -> bool {
        self.spin_angle.is_some()
    }

    /// Returns the marker's horizontal forward direction.
    pub(super) fn forward(&self) -> Vec3 {
        match self.spin_angle {
            Some(angle) => Vec3::new(angle.cos(), 0.0, angle.sin()),
            None => ORIENTATIONS[self.orientation].forward,
        }
    }

    /// Describes the marker's preset or current spin angle for the overlay.
    pub(super) fn rotation_label(&self) -> String {
        match self.spin_angle {
            Some(angle) => format!(
                "spinning {:.0} deg",
                angle.rem_euclid(std::f32::consts::TAU).to_degrees()
            ),
            None => ORIENTATIONS[self.orientation].label.to_owned(),
        }
    }

    /// Combines scene-owned marker placement with the selected receiver orientation.
    pub(super) fn marker_pose(&self, position: Vec3) -> ListenerPose {
        ListenerPose {
            position,
            forward: self.forward().normalize_or_zero(),
            up: Vec3::Y,
        }
    }

    /// Returns the acoustic receiver pose independently of camera-update throttling.
    pub(super) fn active_pose(&self, marker_position: Vec3, camera: &Camera3D) -> ListenerPose {
        match self.mode {
            ListenerMode::RedSphere => self.marker_pose(marker_position),
            ListenerMode::Camera => camera_pose(camera),
        }
    }

    /// Returns a changed camera pose when eligible; explicit controls can bypass the interval.
    /// Failed audio updates remain eligible until the caller records an accepted pose.
    pub(super) fn pending_camera_pose(
        &self,
        camera: &Camera3D,
        now: Instant,
        immediate: bool,
    ) -> Option<ListenerPose> {
        if self.mode != ListenerMode::Camera {
            return None;
        }
        let previous = self.last_camera_update.as_ref();
        if !immediate
            && previous.is_some_and(|update| {
                now.duration_since(update.accepted_at) < CAMERA_UPDATE_INTERVAL
            })
        {
            return None;
        }
        let pose = camera_pose(camera);
        if previous.is_some_and(|update| poses_match(update.pose, pose)) {
            return None;
        }
        Some(pose)
    }

    /// Records a successful audio update, preserving the pose and acceptance time together.
    pub(super) fn camera_synced(&mut self, pose: ListenerPose, accepted_at: Instant) {
        self.last_camera_update = Some(CameraUpdate {
            pose,
            accepted_at,
        });
    }
}

fn camera_pose(camera: &Camera3D) -> ListenerPose {
    ListenerPose {
        position: to_glam(camera.position),
        forward: (to_glam(camera.target) - to_glam(camera.position)).normalize_or_zero(),
        up: to_glam(camera.up).normalize_or_zero(),
    }
}

fn poses_match(left: ListenerPose, right: ListenerPose) -> bool {
    left.position.distance_squared(right.position) <= CAMERA_POSITION_EPSILON_METERS.powi(2)
        && left.forward.dot(right.forward) >= CAMERA_ORIENTATION_DOT_THRESHOLD
        && left.up.dot(right.up) >= CAMERA_ORIENTATION_DOT_THRESHOLD
}

#[cfg(test)]
#[path = "../../tests/unit/app/listener.rs"]
mod tests;
