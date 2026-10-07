use super::*;
use crate::{ListenerPose, PointSource};
use glam::Vec3;

#[test]
fn replacing_the_same_query_returns_the_new_snapshot_and_recounts_bytes() {
    let mut cache = IrCache::new(IrCacheConfig::default());
    let query = query(Vec3::X);
    cache.insert_ir(1, query, ir_snapshot(query));
    let mut newer = ir_snapshot(query);
    newer.energy = Some(4.0);
    newer.byte_size = 48;
    cache.insert_ir(1, query, newer);
    assert_eq!(cache.get_ir(1, query).unwrap().energy, Some(4.0));
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.gpu_bytes(), 48);
}

#[test]
fn snapshot_metadata_must_match_the_cache_query() {
    let mut cache = IrCache::new(IrCacheConfig::default());
    let query = query(Vec3::X);
    let mut wrong_scene = ir_snapshot(query);
    wrong_scene.scene_version += 1;
    cache.insert_ir(1, query, wrong_scene);
    let mut wrong_query = ir_snapshot(query);
    wrong_query.query_id += 1;
    cache.insert_ir(1, query, wrong_query);
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.gpu_bytes(), 0);
}

#[test]
fn nearby_entries_cannot_escape_cache_limits() {
    let mut cache = IrCache::new(IrCacheConfig {
        max_entries: 2,
        max_gpu_cache_bytes: 64,
        ..IrCacheConfig::default()
    });
    let mut query = query(Vec3::X);
    for step in 0..100 {
        query.source.position.x = step as f32 * 0.001;
        cache.insert_ir(1, query, ir_snapshot(query));
        assert!(cache.len() <= 2);
        assert!(cache.gpu_bytes() <= 64);
    }
}

#[test]
fn cache_respects_listener_direction() {
    let mut cache = IrCache::new(IrCacheConfig::default());
    let query = query(Vec3::X);
    cache.insert_ir(1, query, ir_snapshot(query));

    let mut rotated = query;
    rotated.listener.forward = Vec3::Z;

    assert!(cache.get_ir(1, rotated).is_none());

    let mut tilted = query;
    tilted.listener.up = Vec3::Z;
    assert!(cache.get_ir(1, tilted).is_none());
}

#[test]
fn cache_rejects_moved_locations_and_changed_amplitude() {
    let mut cache = IrCache::new(IrCacheConfig::default());
    let query = query(Vec3::X);
    cache.insert_ir(1, query, ir_snapshot(query));

    let mut moved = query;
    moved.listener.position = Vec3::new(10.0, 0.0, 0.0);
    assert!(cache.get_ir(1, moved).is_none());

    let mut changed = query;
    changed.source.amplitude = 2.0;
    assert!(cache.get_ir(1, changed).is_none());
}

#[test]
fn cache_reuses_nearby_source_and_listener_positions() {
    let mut cache = IrCache::new(IrCacheConfig::default());
    let mut original = query(Vec3::X);
    original.source.position = Vec3::new(0.249, 0.0, 0.0);
    original.listener.position = Vec3::new(1.249, 0.0, 0.0);
    cache.insert_ir(1, original, ir_snapshot(original));

    let mut nearby = original;
    nearby.source.position.x = 0.251;
    nearby.listener.position.x = 1.251;

    assert!(cache.get_ir(1, nearby).is_some());
}

#[test]
fn gpu_byte_budget_evicts_old_entries() {
    let mut cache = IrCache::new(IrCacheConfig {
        max_gpu_cache_bytes: 48,
        ..IrCacheConfig::default()
    });
    let first = query(Vec3::X);
    let mut second = query(Vec3::X);
    second.listener.position = Vec3::new(1.0, 0.0, 0.0);

    cache.insert_ir(1, first, ir_snapshot(first));
    cache.insert_ir(1, second, ir_snapshot(second));

    assert!(cache.gpu_bytes() <= 48);
    assert!(cache.get_ir(1, first).is_none());
    assert!(cache.get_ir(1, second).is_some());
}

#[test]
fn a_cache_hit_refreshes_only_that_entry_before_eviction() {
    let mut cache = IrCache::new(IrCacheConfig {
        max_entries: 2,
        reuse_distance_meters: 0.0,
        ..IrCacheConfig::default()
    });
    let first = query(Vec3::X);
    let mut second = first;
    second.source.position.x = 0.01;
    let mut third = first;
    third.source.position.x = 0.02;

    cache.insert_ir(1, first, ir_snapshot(first));
    cache.insert_ir(1, second, ir_snapshot(second));
    assert!(cache.get_ir(1, first).is_some());
    cache.insert_ir(1, third, ir_snapshot(third));

    assert_eq!(cache.len(), 2);
    assert_eq!(cache.gpu_bytes(), 64);
    assert!(cache.get_ir(1, second).is_none());
    assert!(cache.get_ir(1, first).is_some());
    assert!(cache.get_ir(1, third).is_some());
}

#[test]
fn lookup_prefers_the_closest_reusable_response() {
    let mut cache = IrCache::new(IrCacheConfig::default());
    let first = query(Vec3::X);
    let mut second = first;
    second.listener.position.x = 0.1;
    let mut snapshot = ir_snapshot(second);
    snapshot.energy = Some(4.0);
    cache.insert_ir(1, first, ir_snapshot(first));
    cache.insert_ir(1, second, snapshot);

    let mut nearby = second;
    nearby.listener.position.x = 0.09;
    assert_eq!(cache.get_ir(1, nearby).unwrap().energy, Some(4.0));
}

#[test]
fn lookup_never_crosses_scene_versions_or_query_ids() {
    let mut cache = IrCache::new(IrCacheConfig::default());
    let original = query(Vec3::X);
    cache.insert_ir(1, original, ir_snapshot(original));

    assert!(cache.get_ir(2, original).is_none());
    let mut other_query = original;
    other_query.query_id += 1;
    assert!(cache.get_ir(1, other_query).is_none());
}

fn query(listener_forward: Vec3) -> AcousticQuery {
    AcousticQuery {
        query_id: 0,
        source: PointSource::new(Vec3::ZERO),
        listener: ListenerPose::new(Vec3::Y, listener_forward, Vec3::Y),
    }
}

fn ir_snapshot(query: AcousticQuery) -> GpuIrSnapshot {
    GpuIrSnapshot {
        buffer: None,
        byte_size: 32,
        sample_rate: 48_000,
        sample_count: 1,
        energy: Some(2.0),
        scene_version: 1,
        query_id: query.query_id,
    }
}
