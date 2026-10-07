use super::*;
use crate::gpu::backend::VkBackend;
use std::sync::Arc;

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn replacing_a_query_recounts_the_full_retained_allocation() {
    let gpu = VkBackend::new().unwrap();
    let mut cache = IrCache::new(IrCacheConfig::default()).unwrap();
    let query = query();
    cache.insert_ir(snapshot(&gpu, 1, query, 32));
    let newer = snapshot(&gpu, 1, query, 48);
    cache.insert_ir(newer.clone());
    let cached = cache.get_ir(1, query.canonicalized().unwrap()).unwrap();
    assert!(Arc::ptr_eq(cached.buffer(), newer.buffer()));
    assert_eq!(cache.entries.len(), 1);
    // Only one sample is declared, but the cache owns all 48 allocation bytes.
    assert_eq!(cache.gpu_bytes, 48);
    cache.clear();
    assert!(cache.entries.is_empty());
    assert_eq!(cache.gpu_bytes, 0);
}

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn nearby_entries_cannot_escape_cache_limits() {
    let gpu = VkBackend::new().unwrap();
    let mut cache = IrCache::new(IrCacheConfig {
        max_entries: 2,
        max_gpu_cache_bytes: 64,
        ..IrCacheConfig::default()
    })
    .unwrap();
    let mut query = query();
    for step in 0..100 {
        query.source.position.x = step as f32 * 0.001;
        cache.insert_ir(snapshot(&gpu, 1, query, 32));
        assert!(cache.entries.len() <= 2);
        assert!(cache.gpu_bytes <= 64);
    }
}

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn gpu_byte_budget_evicts_old_entries_without_invalidating_external_owners() {
    let gpu = VkBackend::new().unwrap();
    let mut cache = IrCache::new(IrCacheConfig {
        max_gpu_cache_bytes: 48,
        ..IrCacheConfig::default()
    })
    .unwrap();
    let first = query();
    let mut second = first;
    second.listener.position.x = 1.0;
    let retained = snapshot(&gpu, 1, first, 32);
    let weak = Arc::downgrade(retained.buffer());
    cache.insert_ir(retained.clone());
    cache.insert_ir(snapshot(&gpu, 1, second, 32));
    assert_eq!(cache.gpu_bytes, 32);
    assert!(cache.get_ir(1, first.canonicalized().unwrap()).is_none());
    assert!(cache.get_ir(1, second.canonicalized().unwrap()).is_some());
    assert!(weak.upgrade().is_some());
    drop(retained);
    assert!(weak.upgrade().is_none());

    // An impossible replacement must not evict otherwise useful responses.
    cache.insert_ir(snapshot(&gpu, 1, second, 56));
    assert_eq!(cache.gpu_bytes, 32);
    assert_eq!(
        cache
            .get_ir(1, second.canonicalized().unwrap())
            .unwrap()
            .byte_size(),
        32
    );
}

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn a_cache_hit_refreshes_only_that_entry_before_eviction() {
    let gpu = VkBackend::new().unwrap();
    let mut cache = IrCache::new(IrCacheConfig {
        max_entries: 2,
        reuse_distance_meters: 0.0,
        ..IrCacheConfig::default()
    })
    .unwrap();
    let first = query();
    let mut second = first;
    second.source.position.x = 0.01;
    let mut third = first;
    third.source.position.x = 0.02;
    cache.insert_ir(snapshot(&gpu, 1, first, 32));
    cache.insert_ir(snapshot(&gpu, 1, second, 32));
    assert!(cache.get_ir(1, first.canonicalized().unwrap()).is_some());
    cache.insert_ir(snapshot(&gpu, 1, third, 32));
    assert_eq!(cache.entries.len(), 2);
    assert_eq!(cache.gpu_bytes, 64);
    assert!(cache.get_ir(1, second.canonicalized().unwrap()).is_none());
    assert!(cache.get_ir(1, first.canonicalized().unwrap()).is_some());
    assert!(cache.get_ir(1, third.canonicalized().unwrap()).is_some());
}

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn lookup_prefers_the_closest_response_with_matching_provenance() {
    let gpu = VkBackend::new().unwrap();
    let mut cache = IrCache::new(IrCacheConfig::default()).unwrap();
    let first = query();
    let mut second = first;
    second.listener.position.x = 0.1;
    let closest = snapshot(&gpu, 1, second, 32);
    cache.insert_ir(snapshot(&gpu, 1, first, 32));
    cache.insert_ir(closest.clone());
    let mut nearby = second;
    nearby.listener.position.x = 0.09;
    assert!(Arc::ptr_eq(
        cache
            .get_ir(1, nearby.canonicalized().unwrap())
            .unwrap()
            .buffer(),
        closest.buffer()
    ));
    assert!(cache.get_ir(2, nearby.canonicalized().unwrap()).is_none());
    nearby.query_id += 1;
    assert!(cache.get_ir(1, nearby.canonicalized().unwrap()).is_none());
}

fn snapshot(
    gpu: &VkBackend,
    scene: SceneVersion,
    query: AcousticQuery,
    bytes: u64,
) -> GpuIrSnapshot {
    GpuIrSnapshot::new(
        Arc::new(gpu.memory().create_storage_buffer(bytes).unwrap()),
        48_000,
        1,
        scene,
        query,
    )
    .unwrap()
}
