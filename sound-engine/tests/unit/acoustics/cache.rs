//! Pure policy checks; allocation and eviction tests use real buffers separately.

use super::*;
use crate::{ListenerPose, PointSource};
use glam::Vec3;

#[path = "cache/gpu.rs"]
mod gpu;

#[test]
fn reuse_requires_matching_orientation_locations_amplitude_and_query_id() {
    let cache = IrCache::new(IrCacheConfig::default()).unwrap();
    let original = query();
    let mut variants = [original; 6];
    variants[0].listener.forward = Vec3::Z;
    variants[1].listener.up = Vec3::Z;
    variants[2].listener.position = Vec3::splat(10.0);
    variants[3].source.position = Vec3::splat(10.0);
    variants[4].source.amplitude = 2.0;
    variants[5].query_id = 1;
    assert_eq!(cache.reuse_score(original, original), Some(0.0));
    for changed in variants {
        assert!(cache.reuse_score(original, changed).is_none());
    }
}

#[test]
fn reuse_accepts_nearby_endpoints_and_prefers_the_closest_match() {
    let cache = IrCache::new(IrCacheConfig::default()).unwrap();
    let mut first = query();
    first.source.position.x = 0.249;
    first.listener.position.x = 1.249;
    let mut second = first;
    second.source.position.x = 0.251;
    second.listener.position.x = 1.251;
    let mut requested = second;
    requested.listener.position.x = 1.252;
    assert!(
        cache.reuse_score(first, requested).unwrap()
            > cache.reuse_score(second, requested).unwrap()
    );
}

#[test]
fn zero_capacity_or_budget_disables_reuse() {
    for config in [
        IrCacheConfig {
            max_entries: 0,
            ..IrCacheConfig::default()
        },
        IrCacheConfig {
            max_gpu_cache_bytes: 0,
            ..IrCacheConfig::default()
        },
    ] {
        let mut cache = IrCache::new(config).unwrap();
        assert!(!cache.is_enabled());
        assert!(cache.get_ir(1, query()).is_none());
    }
}

#[test]
fn invalid_reuse_tolerances_are_rejected_at_construction() {
    for distance in [f32::NAN, f32::INFINITY, -1.0] {
        assert!(
            IrCache::new(IrCacheConfig {
                reuse_distance_meters: distance,
                ..IrCacheConfig::default()
            })
            .is_err()
        );
    }
    for threshold in [f32::NAN, f32::INFINITY, -1.01, 1.01] {
        assert!(
            IrCache::new(IrCacheConfig {
                listener_orientation_dot_threshold: threshold,
                ..IrCacheConfig::default()
            })
            .is_err()
        );
    }
    for threshold in [-1.0, 1.0] {
        assert!(
            IrCache::new(IrCacheConfig {
                listener_orientation_dot_threshold: threshold,
                ..IrCacheConfig::default()
            })
            .is_ok()
        );
    }
}

fn query() -> AcousticQuery {
    AcousticQuery {
        query_id: 0,
        source: PointSource::new(Vec3::ZERO),
        listener: ListenerPose::new(Vec3::Y, Vec3::X, Vec3::Y),
    }
}
