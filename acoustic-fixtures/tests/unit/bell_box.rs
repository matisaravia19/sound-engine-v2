use super::*;

#[test]
fn figure_8_matches_table_1_shortest_diffraction_and_reflection() {
    let radial = |v: Vec3| vec3(v.x - EDGE_START.x, 0.0, v.z - EDGE_START.z).length();
    let diffraction = (radial(BAFFLE_SOURCE) + radial(BAFFLE_LISTENER))
        .hypot(BAFFLE_SOURCE.y - BAFFLE_LISTENER.y);
    let image_source = vec3(-BAFFLE_SOURCE.x, BAFFLE_SOURCE.y, BAFFLE_SOURCE.z);

    assert!((diffraction - 1.6758).abs() < 0.001);
    assert!((image_source.distance(BAFFLE_LISTENER) - 3.0154).abs() < 0.001);
}
