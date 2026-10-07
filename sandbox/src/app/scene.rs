//! Selected geometry, exploration positions, and accepted acoustic scene changes.

use super::geometry::{PlaygroundScene, SceneConfig, build_scene, from_glam, scene_description};
use super::poses::{
    listener_position, listener_position_count, listener_position_name, source_position,
    source_position_count, source_position_name,
};
use crate::scenes;
use raylib::prelude::Vector3;
use sound_engine::core::error::{SoundError, SoundResult};
use sound_engine::scene::{SceneDescription, SceneVersion};

/// Selects full geometry or a distance-only comparison without moving visible markers.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum PlaybackMode {
    /// Includes reflections and diffraction from the selected scene.
    FullSimulation,
    /// Uses empty acoustic geometry with the same source and receiver.
    DistanceOnly,
}

impl PlaybackMode {
    /// Returns the playback control's display label.
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::FullSimulation => "Full simulation",
            Self::DistanceOnly => "Distance only",
        }
    }
}

/// Retains editable geometry and presets while benchmark scenes are selected.
pub(super) struct EditableRoom {
    /// Settings used to build the accepted room geometry.
    pub(super) config: SceneConfig,
    /// Matching display and acoustic geometry.
    pub(super) geometry: PlaygroundScene,
    listener_index: usize,
    source_index: usize,
}

/// Keeps a benchmark's validated geometry and current measurement pair together.
pub(super) struct BenchmarkScene {
    /// Geometry, render triangles, and nonempty measurement pairs.
    pub(super) loaded: scenes::Loaded,
    /// Zero-based index of the selected measurement pair.
    pub(super) pair_index: usize,
}

impl BenchmarkScene {
    /// Returns the source and receiver currently explored in this benchmark.
    pub(super) fn pair(&self) -> &scenes::Pair {
        &self.loaded.pairs[self.pair_index]
    }
}

struct Selection {
    catalog_index: usize,
    benchmark: Option<BenchmarkScene>,
}

impl Selection {
    fn load(catalog: &[scenes::Entry], index: usize) -> SoundResult<Self> {
        let entry = catalog
            .get(index)
            .ok_or_else(|| SoundError::invalid_argument("Unknown scene index"))?;
        let benchmark = if entry.id == "playground" {
            None
        } else {
            Some(BenchmarkScene {
                loaded: entry.load().map_err(SoundError::external)?,
                pair_index: 0,
            })
        };
        Ok(Self {
            catalog_index: index,
            benchmark,
        })
    }

    fn description(&self, room: &PlaygroundScene, mode: PlaybackMode) -> SceneDescription {
        // Empty acoustic geometry retains distance, delay, and spatialization.
        // The visible geometry and selected positions remain available in either mode.
        match mode {
            PlaybackMode::FullSimulation => self
                .benchmark
                .as_ref()
                .map(|scene| scene.loaded.description.clone())
                .unwrap_or_else(|| scene_description(room)),
            PlaybackMode::DistanceOnly => SceneDescription::default(),
        }
    }
}

/// Owns scene selection and commits geometry changes only after audio accepts them.
/// Audio loading is supplied by the caller; this state owns no device resources.
pub(super) struct SceneState {
    catalog: Vec<scenes::Entry>,
    selection: Selection,
    room: EditableRoom,
    playback_mode: PlaybackMode,
}

impl SceneState {
    /// Loads the initial selection before initializing audio or display resources.
    pub(super) fn new(
        catalog: Vec<scenes::Entry>,
        diffraction_radius: f32,
        initial_scene: Option<&str>,
    ) -> SoundResult<Self> {
        let index = match initial_scene {
            Some(id) => catalog
                .iter()
                .position(|entry| entry.id.eq_ignore_ascii_case(id))
                .ok_or_else(|| {
                    SoundError::invalid_argument(format!("Unknown scene {id}; use --list-scenes"))
                })?,
            None => 0,
        };
        let config = SceneConfig {
            roof_enabled: false,
            door_open: true,
            outer_walls_enabled: true,
            diffraction_radius,
        };
        Ok(Self {
            selection: Selection::load(&catalog, index)?,
            catalog,
            room: EditableRoom {
                config,
                geometry: build_scene(config),
                listener_index: 0,
                source_index: 0,
            },
            playback_mode: PlaybackMode::FullSimulation,
        })
    }

    /// Returns the available built-in and locally installed benchmark scenes.
    pub(super) fn catalog(&self) -> &[scenes::Entry] {
        &self.catalog
    }

    /// Returns the accepted selection's catalog index.
    pub(super) fn index(&self) -> usize {
        self.selection.catalog_index
    }

    /// Returns the selected benchmark, or none for the editable room.
    pub(super) fn benchmark(&self) -> Option<&BenchmarkScene> {
        self.selection.benchmark.as_ref()
    }

    /// Returns the retained editable room without permitting unsynchronized edits.
    pub(super) fn room(&self) -> &EditableRoom {
        &self.room
    }

    /// Returns the acoustic mode most recently accepted by audio.
    pub(super) fn playback_mode(&self) -> PlaybackMode {
        self.playback_mode
    }

    /// Builds the selected acoustic geometry for initial engine loading.
    pub(super) fn description(&self) -> SceneDescription {
        self.selection
            .description(&self.room.geometry, self.playback_mode)
    }

    /// Loads a candidate and replaces the selected scene only after audio accepts it.
    pub(super) fn select(
        &mut self,
        index: usize,
        load_audio: impl FnOnce(SceneDescription) -> SoundResult<SceneVersion>,
    ) -> SoundResult<SceneVersion> {
        let selection = Selection::load(&self.catalog, index)?;
        let version = load_audio(selection.description(&self.room.geometry, self.playback_mode))?;
        self.selection = selection;
        Ok(version)
    }

    /// Applies room settings and matching display geometry after a successful audio load.
    pub(super) fn update_room(
        &mut self,
        config: SceneConfig,
        load_audio: impl FnOnce(SceneDescription) -> SoundResult<SceneVersion>,
    ) -> SoundResult<SceneVersion> {
        if self.benchmark().is_some() {
            return Err(SoundError::invalid_state(
                "The selected scene is not editable",
            ));
        }
        let geometry = build_scene(config);
        let version = load_audio(self.selection.description(&geometry, self.playback_mode))?;
        self.room.config = config;
        self.room.geometry = geometry;
        Ok(version)
    }

    /// Switches acoustic mode after audio accepts the replacement geometry.
    pub(super) fn toggle_playback(
        &mut self,
        load_audio: impl FnOnce(SceneDescription) -> SoundResult<SceneVersion>,
    ) -> SoundResult<SceneVersion> {
        let mode = match self.playback_mode {
            PlaybackMode::FullSimulation => PlaybackMode::DistanceOnly,
            PlaybackMode::DistanceOnly => PlaybackMode::FullSimulation,
        };
        let version = load_audio(self.selection.description(&self.room.geometry, mode))?;
        self.playback_mode = mode;
        Ok(version)
    }

    /// Returns the selected receiver marker's position in world meters.
    pub(super) fn listener_position(&self) -> Vector3 {
        match self.benchmark() {
            Some(scene) => from_glam(scene.pair().listener),
            None => listener_position(self.room.listener_index),
        }
    }

    /// Returns the selected source marker's position in world meters.
    pub(super) fn source_position(&self) -> Vector3 {
        match self.benchmark() {
            Some(scene) => from_glam(scene.pair().source),
            None => source_position(self.room.source_index),
        }
    }

    /// Returns the receiver preset or measurement-pair name.
    pub(super) fn listener_label(&self) -> &str {
        match self.benchmark() {
            Some(scene) => &scene.pair().name,
            None => listener_position_name(self.room.listener_index),
        }
    }

    /// Returns the source preset or measurement-pair name.
    pub(super) fn source_label(&self) -> &str {
        match self.benchmark() {
            Some(scene) => &scene.pair().name,
            None => source_position_name(self.room.source_index),
        }
    }

    /// Advances the receiver preset, or both markers to the next benchmark pair.
    pub(super) fn next_listener(&mut self) {
        if let Some(scene) = &mut self.selection.benchmark {
            scene.pair_index = (scene.pair_index + 1) % scene.loaded.pairs.len();
        } else {
            self.room.listener_index = (self.room.listener_index + 1) % listener_position_count();
        }
    }

    /// Advances the source preset, or both markers to the previous benchmark pair.
    pub(super) fn next_source(&mut self) {
        if let Some(scene) = &mut self.selection.benchmark {
            scene.pair_index =
                (scene.pair_index + scene.loaded.pairs.len() - 1) % scene.loaded.pairs.len();
        } else {
            self.room.source_index = (self.room.source_index + 1) % source_position_count();
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/app/scene.rs"]
mod tests;
