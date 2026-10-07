use super::*;

#[test]
fn menu_rows_follow_the_selected_page_and_exclude_headings_and_gaps() {
    let layout = SceneMenuLayout::new(640, 330, 7, 12);
    assert_eq!(layout.entries(), 5..10);
    assert_eq!(layout.hit_test(Vector2::new(48.0, 120.0)), Some(5));
    assert_eq!(layout.hit_test(Vector2::new(48.0, 224.0)), Some(9));
    for point in [
        Vector2::new(48.0, 80.0),
        Vector2::new(39.0, 120.0),
        Vector2::new(601.0, 120.0),
        Vector2::new(48.0, 137.5),
        Vector2::new(48.0, 250.0),
    ] {
        assert_eq!(layout.hit_test(point), None, "{point:?}");
    }
}

#[test]
fn narrow_windows_keep_the_drawn_rows_clickable() {
    let layout = SceneMenuLayout::new(200, 720, 0, 2);
    assert_eq!(layout.row(0).width, 240.0);
    assert_eq!(layout.hit_test(Vector2::new(180.0, 120.0)), Some(0));
    assert_eq!(layout.hit_test(Vector2::new(281.0, 120.0)), None);

    let layout = SceneMenuLayout::new(1280, 720, 0, 2);
    assert_eq!(layout.hit_test(Vector2::new(799.0, 120.0)), Some(0));
    assert_eq!(layout.hit_test(Vector2::new(801.0, 120.0)), None);
}

#[test]
fn empty_and_partial_pages_have_no_phantom_rows() {
    let empty = SceneMenuLayout::new(640, 720, 0, 0);
    assert_eq!(empty.entries(), 0..0);
    assert_eq!(empty.hit_test(Vector2::new(48.0, 120.0)), None);

    let last = SceneMenuLayout::new(640, 720, usize::MAX, 20);
    assert_eq!(last.entries(), 18..20);
    assert_eq!(last.hit_test(Vector2::new(48.0, 120.0)), Some(18));
    assert_eq!(last.hit_test(Vector2::new(48.0, 146.0)), Some(19));
    assert_eq!(last.hit_test(Vector2::new(48.0, 172.0)), None);
}
