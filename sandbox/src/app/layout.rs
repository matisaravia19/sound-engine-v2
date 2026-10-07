//! Shared bounds for drawing interactive controls and resolving pointer input.

use raylib::prelude::{Rectangle, Vector2};
use std::ops::Range;

/// Starts playback from the active source and listener.
pub(super) const START_SOUND_BUTTON: Rectangle = Rectangle {
    x: 24.0,
    y: 24.0,
    width: 160.0,
    height: 40.0,
};

/// Toggles the displayed diffraction bisector planes.
pub(super) const BISECTOR_BUTTON: Rectangle = Rectangle {
    x: 24.0,
    y: 70.0,
    width: 230.0,
    height: 34.0,
};

/// Switches between the full acoustic scene and the distance-only comparison.
pub(super) const PLAYBACK_MODE_BUTTON: Rectangle = Rectangle {
    x: 278.0,
    y: 70.0,
    width: 310.0,
    height: 34.0,
};

const MENU_ROW_HEIGHT: i32 = 26;
const MENU_ROWS_TOP: i32 = 112;

/// One catalog page, with matching visible rows and pointer targets at any window size.
pub(super) struct SceneMenuLayout {
    page: Range<usize>,
    row_width: i32,
    capacity: usize,
}

impl SceneMenuLayout {
    /// Fits the page to the window and keeps the highlighted catalog entry visible.
    pub(super) fn new(width: i32, height: i32, choice: usize, entry_count: usize) -> Self {
        let capacity = ((height - 180) / MENU_ROW_HEIGHT).clamp(3, 18) as usize;
        let choice = choice.min(entry_count.saturating_sub(1));
        let start = choice / capacity * capacity;
        Self {
            page: start..start.saturating_add(capacity).min(entry_count),
            row_width: (width - 80).clamp(240, 760),
            capacity,
        }
    }

    /// Returns the catalog indices drawn on this page.
    pub(super) fn entries(&self) -> Range<usize> {
        self.page.clone()
    }

    /// Returns the background panel enclosing the heading, rows, and footer.
    pub(super) fn panel(&self) -> Rectangle {
        Rectangle {
            x: 32.0,
            y: 48.0,
            width: (self.row_width + 16) as f32,
            height: (self.capacity as i32 * MENU_ROW_HEIGHT + 104) as f32,
        }
    }

    /// Returns a visible entry's bounds, leaving one pixel between adjacent rows.
    pub(super) fn row(&self, index: usize) -> Rectangle {
        debug_assert!(self.page.contains(&index));
        Rectangle {
            x: 40.0,
            y: (MENU_ROWS_TOP + (index - self.page.start) as i32 * MENU_ROW_HEIGHT) as f32,
            width: self.row_width as f32,
            height: (MENU_ROW_HEIGHT - 1) as f32,
        }
    }

    /// Returns the entry under the pointer; headings, gaps, and unused rows select nothing.
    pub(super) fn hit_test(&self, point: Vector2) -> Option<usize> {
        self.entries()
            .find(|&index| self.row(index).check_collision_point_rec(point))
    }

    /// Returns the footer's text origin below the page's reserved row area.
    pub(super) fn footer_y(&self) -> i32 {
        MENU_ROWS_TOP + self.capacity as i32 * MENU_ROW_HEIGHT + 8
    }
}

#[cfg(test)]
#[path = "../../tests/unit/app/layout.rs"]
mod tests;
