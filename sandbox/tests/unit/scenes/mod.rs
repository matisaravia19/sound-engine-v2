use super::*;
#[test]
fn builtins_have_valid_geometry_and_pairs() {
    for entry in catalog(Path::new("missing-runtime-assets"))
        .into_iter()
        .filter(|e| e.id != "playground" && e.positions.is_none())
    {
        let scene = entry.load().unwrap();
        assert!(!scene.pairs.is_empty());
        if entry.id == "bell-baffle" {
            assert_eq!(scene.description.diffraction_edges.len(), 1);
        }
        if entry.id == "bell-no-diffraction" {
            assert!(scene.description.diffraction_edges.is_empty());
        }
    }
}
