//! Fixed receiver and source position presets.
use raylib::prelude::Vector3;

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
