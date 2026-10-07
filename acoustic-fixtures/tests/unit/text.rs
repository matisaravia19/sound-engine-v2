use super::*;

const HEADER: &str = "case,sx,sy,sz,rx,ry,rz\n";
const TRIANGLE: &str = "0 0 0 1 0 0 0 1 0";
const EDGE: &str = "edge 0 0 0 0 1 0 1 0 0 6.283185307 0.2";

#[test]
fn scene_reader_preserves_large_identifiers_and_groups_triangles() {
    let scene = parse_scene(&format!(
        "  # fixture\n\nmaterial 16777217 0.25 0.5\nmaterial 1 1 0\n\
         tri 16777217 {TRIANGLE}\ntri 1 {TRIANGLE}\ntri 16777217 {TRIANGLE}\n"
    ))
    .unwrap();
    assert_eq!(scene.meshes.len(), 2);
    assert_eq!(scene.meshes[0].id, 1);
    assert_eq!(scene.meshes[1].id, 16_777_217);
    assert_eq!(scene.meshes[1].vertices.len(), 6);
    assert_eq!(scene.objects[1].material_id, 16_777_217);
    assert_eq!(scene.materials[0].absorption, 0.25);
}

#[test]
fn malformed_scene_records_report_the_source_line() {
    for row in [
        "material 1.5 0.2 0",
        "material -1 0.2 0",
        "material 0 0.2 0",
        "material 4294967296 0.2 0",
        "material 1 NaN 0",
        "tri 1 0 0 0",
        "unknown 1 2 3",
    ] {
        let error = parse_scene(&format!("# header\n{row}\n")).unwrap_err();
        assert!(error.starts_with("line 2:"), "{row}: {error}");
    }
}

#[test]
fn scene_reader_uses_engine_validation_for_geometry_and_materials() {
    for text in [
        format!("tri 1 {TRIANGLE}"),
        "material 1 0.2 0\nmaterial 1 0.4 0".into(),
        "material 1 1.1 0".into(),
        "material 1 0.2 -0.1".into(),
        "edge 0 0 0 0 0 0 1 0 0 6.283185307 0.2".into(),
    ] {
        assert!(
            parse_scene(&text).is_err(),
            "accepted invalid scene: {text}"
        );
    }
}

#[test]
fn cached_edge_scales_preserve_only_the_disabled_state() {
    let scene = parse_scene(&format!("{EDGE}\n{EDGE} 0\n{EDGE} 0.85\n")).unwrap();
    assert_eq!(scene.diffraction_edges.len(), 2);
    assert_eq!(scene.diffraction_edges[0].id, 1);
    assert_eq!(scene.diffraction_edges[1].id, 2);
    assert_eq!(scene.diffraction_edges[0].diffraction_radius, 0.2);
    assert_eq!(scene.diffraction_edges[1].diffraction_radius, 0.2);
    assert!(parse_scene(&format!("{EDGE} -1")).is_err());
}

#[test]
fn position_reader_preserves_engine_axes_under_both_headers() {
    for header in [
        HEADER,
        "case,source_x,source_y,source_z,listener_x,listener_y,listener_z\n",
    ] {
        let pairs = parse_positions(&format!("{header}\nRS5_LS1_MP1, 1, 2, 3, 4, 5, 6\n")).unwrap();
        assert_eq!(
            pairs,
            vec![PositionPair {
                name: "RS5_LS1_MP1".into(),
                source: Vec3::new(1.0, 2.0, 3.0),
                listener: Vec3::new(4.0, 5.0, 6.0),
            }]
        );
    }
}

#[test]
fn position_reader_rejects_missing_duplicate_and_invalid_pairs() {
    assert!(parse_positions(HEADER).is_err());
    assert!(parse_positions("case,x,y,z\n").is_err());
    for rows in [
        "A,1,2,3,4,5,6\nA,2,3,4,5,6,7\n",
        "A,NaN,2,3,4,5,6\n",
        "A,1,2,3,4,5\n",
        "../A,1,2,3,4,5,6\n",
        ",1,2,3,4,5,6\n",
    ] {
        assert!(
            parse_positions(&format!("{HEADER}{rows}")).is_err(),
            "accepted {rows}"
        );
    }
}
