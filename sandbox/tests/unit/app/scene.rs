use super::*;
use sound_engine::core::error::ErrorCode;
use sound_engine::scene::SceneManager;
use std::path::Path;

#[test]
fn selection_keeps_pairs_with_their_geometry_and_restores_editable_presets() {
    let mut state = new_state(None);
    let mut audio = SceneManager::new();
    state.next_listener();
    state.next_source();
    let room_listener = state.listener_position();
    let room_source = state.source_position();

    let free_field = catalog_index(&state, "free-field");
    state.select(free_field, |scene| audio.load(scene)).unwrap();
    state.next_source();
    assert_eq!(state.benchmark().unwrap().pair_index, 2);
    assert_eq!(state.listener_label(), "Campo libre 5 m");
    state.next_listener();
    assert_eq!(state.benchmark().unwrap().pair_index, 0);
    state.next_listener();
    let selected_listener = state.listener_position();

    let shoebox = catalog_index(&state, "shoebox");
    assert!(state.select(shoebox, reject_audio).is_err());
    assert_eq!(state.index(), free_field);
    assert_eq!(state.listener_position(), selected_listener);
    assert_eq!(state.benchmark().unwrap().pair_index, 1);
    assert!(state.description().objects.is_empty());

    let missing_fixture = catalog_index(&state, "RS5");
    assert!(
        state
            .select(missing_fixture, |_| panic!("invalid fixture reached audio"))
            .is_err()
    );
    assert!(
        state
            .select(usize::MAX, |_| panic!("invalid index reached audio"))
            .is_err()
    );
    assert_eq!(state.index(), free_field);
    assert_eq!(state.listener_position(), selected_listener);

    state.select(0, |scene| audio.load(scene)).unwrap();
    assert!(state.benchmark().is_none());
    assert_eq!(state.listener_position(), room_listener);
    assert_eq!(state.source_position(), room_source);
    state.select(free_field, |scene| audio.load(scene)).unwrap();
    assert_eq!(state.benchmark().unwrap().pair_index, 0);
}

#[test]
fn rejected_room_edits_preserve_display_geometry_and_settings() {
    let mut state = new_state(None);
    let original_blocks = state.room().geometry.blocks.len();
    let original_vertices = state.description().meshes[0].vertices.len();
    let mut config = state.room().config;
    config.door_open = false;
    let error = state
        .update_room(config, |candidate| {
            assert!(candidate.diffraction_edges.is_empty());
            assert!(candidate.meshes[0].vertices.len() > original_vertices);
            reject_audio(candidate)
        })
        .unwrap_err();
    assert_eq!(error.code(), ErrorCode::BackendUnavailable);
    assert!(state.room().config.door_open);
    assert_eq!(state.room().geometry.blocks.len(), original_blocks);
    assert!(!state.room().geometry.diffraction_edges.is_empty());
    assert_eq!(
        state.description().meshes[0].vertices.len(),
        original_vertices
    );

    let mut audio = SceneManager::new();
    state
        .update_room(config, |scene| audio.load(scene))
        .unwrap();
    assert!(!state.room().config.door_open);
    assert_eq!(state.room().geometry.blocks.len(), original_blocks + 1);
    assert!(state.room().geometry.diffraction_edges.is_empty());
    assert!(state.description().diffraction_edges.is_empty());
}

#[test]
fn playback_mode_applies_to_scene_selection_and_commits_only_after_acceptance() {
    let mut state = new_state(Some("WEDGE"));
    let mut audio = SceneManager::new();
    let initial_listener = state.listener_position();
    assert!(
        state
            .toggle_playback(|candidate| {
                assert_empty(&candidate);
                reject_audio(candidate)
            })
            .is_err()
    );
    assert!(state.playback_mode() == PlaybackMode::FullSimulation);
    assert!(!state.description().objects.is_empty());
    assert_eq!(state.listener_position(), initial_listener);

    state.toggle_playback(|scene| audio.load(scene)).unwrap();
    let bell = catalog_index(&state, "bell-baffle");
    state
        .select(bell, |candidate| {
            assert_empty(&candidate);
            audio.load(candidate)
        })
        .unwrap();
    assert_eq!(
        state
            .benchmark()
            .unwrap()
            .loaded
            .description
            .diffraction_edges
            .len(),
        1
    );
    let selected_listener = state.listener_position();
    assert!(state.toggle_playback(reject_audio).is_err());
    assert!(state.playback_mode() == PlaybackMode::DistanceOnly);
    assert_empty(&state.description());

    state
        .toggle_playback(|candidate| {
            assert_eq!(candidate.diffraction_edges.len(), 1);
            audio.load(candidate)
        })
        .unwrap();
    assert!(state.playback_mode() == PlaybackMode::FullSimulation);
    assert_eq!(state.index(), bell);
    assert_eq!(state.listener_position(), selected_listener);
    assert_eq!(state.description().diffraction_edges.len(), 1);
}

#[test]
fn distance_only_edits_are_retained_for_full_simulation_and_benchmarks_reject_edits() {
    let mut state = new_state(None);
    let mut audio = SceneManager::new();
    state.toggle_playback(|scene| audio.load(scene)).unwrap();
    let original_blocks = state.room().geometry.blocks.len();
    let mut config = state.room().config;
    config.roof_enabled = true;
    state
        .update_room(config, |candidate| {
            assert_empty(&candidate);
            audio.load(candidate)
        })
        .unwrap();
    assert!(state.room().config.roof_enabled);
    assert_eq!(state.room().geometry.blocks.len(), original_blocks + 1);
    state.toggle_playback(|scene| audio.load(scene)).unwrap();
    assert!(!state.description().objects.is_empty());

    let shoebox = catalog_index(&state, "shoebox");
    state.select(shoebox, |scene| audio.load(scene)).unwrap();
    config.roof_enabled = false;
    let error = state
        .update_room(config, |_| panic!("benchmark edit reached audio"))
        .unwrap_err();
    assert_eq!(error.code(), ErrorCode::InvalidState);
    state.select(0, |scene| audio.load(scene)).unwrap();
    assert!(state.room().config.roof_enabled);
}

fn new_state(initial: Option<&str>) -> SceneState {
    SceneState::new(
        scenes::catalog(Path::new("missing-runtime-assets")),
        1.25,
        initial,
    )
    .unwrap()
}

fn catalog_index(state: &SceneState, id: &str) -> usize {
    state
        .catalog()
        .iter()
        .position(|entry| entry.id == id)
        .unwrap()
}

fn reject_audio(_: SceneDescription) -> SoundResult<SceneVersion> {
    Err(SoundError::backend_unavailable(
        "simulated scene-load failure",
    ))
}

fn assert_empty(scene: &SceneDescription) {
    assert!(scene.meshes.is_empty());
    assert!(scene.objects.is_empty());
    assert!(scene.materials.is_empty());
    assert!(scene.diffraction_edges.is_empty());
}
