use super::*;

#[test]
fn replacing_the_same_query_returns_the_new_snapshot_and_recounts_bytes() {
    let mut cache = IrCache::new(IrCacheConfig::default());
    let query = query(Vec3::X);
    cache.insert_ir(query, ir_snapshot(query));
    let mut newer = ir_snapshot(query);
    newer.energy = Some(4.0);
    newer.byte_size = 48;
    cache.insert_ir(query, newer);
    assert_eq!(cache.get_ir(query).unwrap().energy, Some(4.0));
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.gpu_bytes(), 48);
}

#[test]
fn snapshot_metadata_must_match_the_cache_query() {
    let mut cache = IrCache::new(IrCacheConfig::default());
    let query = query(Vec3::X);
    let mut wrong_scene = ir_snapshot(query);
    wrong_scene.scene_version += 1;
    cache.insert_ir(query, wrong_scene);
    let mut wrong_query = ir_snapshot(query);
    wrong_query.query_id += 1;
    cache.insert_ir(query, wrong_query);
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.gpu_bytes(), 0);
}

#[test]
fn duplicate_quantized_keys_cannot_escape_cache_limits() {
    let mut cache = IrCache::new(IrCacheConfig {
        max_entries: 2,
        max_gpu_cache_bytes: 64,
        ..IrCacheConfig::default()
    });
    let mut query = query(Vec3::X);
    for step in 0..100 {
        // Distinct exact queries still collide in the same quantized bucket.
        query.source_position.x = step as f32 * 0.001;
        cache.insert_ir(query, ir_snapshot(query));
        assert!(cache.len() <= 2);
        assert!(cache.gpu_bytes() <= 64);
    }
}

#[test]
fn cache_respects_listener_direction() {
    let mut cache = IrCache::new(IrCacheConfig::default());
    let query = query(Vec3::X);
    cache.insert_ir(query, ir_snapshot(query));

    let mut rotated = query;
    rotated.listener_forward = Vec3::Z;

    assert!(cache.get_ir(rotated).is_none());

    let mut tilted = query;
    tilted.listener_up = Vec3::Z;
    assert!(cache.get_ir(tilted).is_none());
}

#[test]
fn cache_rejects_moved_locations_and_changed_amplitude() {
    let mut cache = IrCache::new(IrCacheConfig::default());
    let query = query(Vec3::X);
    cache.insert_ir(query, ir_snapshot(query));

    let mut moved = query;
    moved.listener_position = Vec3::new(10.0, 0.0, 0.0);
    assert!(cache.get_ir(moved).is_none());

    let mut changed = query;
    changed.source_amplitude = 2.0;
    assert!(cache.get_ir(changed).is_none());
}

#[test]
fn cache_reuses_nearby_positions_across_bucket_boundaries() {
    let mut cache = IrCache::new(IrCacheConfig::default());
    let mut original = query(Vec3::X);
    original.source_position = Vec3::new(0.249, 0.0, 0.0);
    original.listener_position = Vec3::new(1.249, 0.0, 0.0);
    cache.insert_ir(original, ir_snapshot(original));

    let mut across_boundary = original;
    across_boundary.source_position.x = 0.251;
    across_boundary.listener_position.x = 1.251;

    assert!(cache.get_ir(across_boundary).is_some());
}

#[test]
fn gpu_byte_budget_evicts_old_entries() {
    let mut cache = IrCache::new(IrCacheConfig {
        max_gpu_cache_bytes: 48,
        ..IrCacheConfig::default()
    });
    let first = query(Vec3::X);
    let mut second = query(Vec3::X);
    second.listener_position = Vec3::new(1.0, 0.0, 0.0);

    cache.insert_ir(first, ir_snapshot(first));
    cache.insert_ir(second, ir_snapshot(second));

    assert!(cache.gpu_bytes() <= 48);
    assert!(cache.get_ir(first).is_none());
    assert!(cache.get_ir(second).is_some());
}

fn query(listener_forward: Vec3) -> IrCacheQuery {
    IrCacheQuery {
        scene_version: 1,
        query_id: 0,
        source_position: Vec3::ZERO,
        source_amplitude: 1.0,
        listener_position: Vec3::Y,
        listener_forward,
        listener_up: Vec3::Y,
    }
}

fn ir_snapshot(query: IrCacheQuery) -> GpuIrSnapshot {
    GpuIrSnapshot {
        buffer: None,
        byte_size: 32,
        sample_rate: 48_000,
        sample_count: 1,
        energy: Some(2.0),
        scene_version: query.scene_version,
        query_id: query.query_id,
    }
}
