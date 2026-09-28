use crate::acoustics::GpuIrSnapshot;
use crate::scene::SceneVersion;
use glam::Vec3;
use std::collections::{HashMap, VecDeque};

/// Shared public policy for acoustic impulse-response reuse.
pub use crate::core::config::IrCacheConfig;

/// Source/listener state used to query the acoustic IR cache.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IrCacheQuery {
    /// Scene version represented by any reusable IR.
    pub scene_version: SceneVersion,
    /// Application query id copied into the generated IR.
    pub query_id: u32,
    /// Source position in world meters.
    pub source_position: Vec3,
    /// Reference source pressure amplitude baked into the cached IR.
    pub source_amplitude: f32,
    /// Listener position in world meters.
    pub listener_position: Vec3,
    /// Listener forward direction in world space.
    pub listener_forward: Vec3,
    /// Listener up direction in world space.
    pub listener_up: Vec3,
}

/// Small GPU IR LRU cache keyed by scene and oriented source/listener state.
///
/// Entries retain reference-counted GPU buffers; eviction releases only the
/// cache's ownership and requires neither readback nor a hit-level snapshot.
pub struct IrCache {
    cfg: IrCacheConfig,
    entries: HashMap<LocationCacheKey, Vec<IrCacheEntry>>,
    lru: VecDeque<IrCacheKey>,
    gpu_bytes: usize,
}

#[derive(Clone)]
struct IrCacheEntry {
    query: IrCacheQuery,
    snapshot: GpuIrSnapshot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct LocationCacheKey {
    scene_version: SceneVersion,
    query_id: u32,
    source_cell: [i32; 3],
    listener_cell: [i32; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct IrCacheKey {
    location: LocationCacheKey,
    listener_forward_cell: [i32; 3],
    listener_up_cell: [i32; 3],
}

impl IrCache {
    /// Creates an empty IR cache with the provided reuse policy.
    pub fn new(cfg: IrCacheConfig) -> Self {
        Self {
            cfg,
            entries: HashMap::new(),
            lru: VecDeque::new(),
            gpu_bytes: 0,
        }
    }

    /// Returns a reusable GPU IR when the full acoustic query remains within tolerance.
    pub fn get_ir(&mut self, query: IrCacheQuery) -> Option<GpuIrSnapshot> {
        if !self.is_enabled() {
            return None;
        }

        // The cache is deliberately small, so scanning its populated buckets
        // is cheaper and more reliable than enumerating up to 27 source cells
        // times 27 listener cells. Exact distance/orientation checks below are
        // authoritative; quantization is only an insertion/eviction bucket.
        let mut best: Option<(f32, LocationCacheKey, usize, IrCacheKey, GpuIrSnapshot)> = None;
        for (&location, entries) in &self.entries {
            if location.scene_version != query.scene_version || location.query_id != query.query_id
            {
                continue;
            }
            for (index, entry) in entries.iter().enumerate() {
                if !self.can_reuse(entry, query) {
                    continue;
                }
                let score = reuse_score(entry.query, query);
                if best
                    .as_ref()
                    .is_none_or(|(best_score, ..)| score < *best_score)
                {
                    best = Some((
                        score,
                        location,
                        index,
                        self.key_for_query(entry.query),
                        entry.snapshot.clone(),
                    ));
                }
            }
        }
        let (_, location, index, entry_key, snapshot) = best?;

        self.touch(entry_key);
        if let Some(entries) = self.entries.get_mut(&location) {
            let entry = entries.remove(index);
            entries.push(entry);
        }
        Some(snapshot)
    }

    /// Inserts or replaces the exact source/listener state; inconsistent snapshot
    /// scene/query metadata is rejected without changing the cache.
    pub fn insert_ir(&mut self, query: IrCacheQuery, mut snapshot: GpuIrSnapshot) {
        if !self.is_enabled()
            || snapshot.scene_version != query.scene_version
            || snapshot.query_id != query.query_id
        {
            return;
        }
        if let Some(buffer) = &snapshot.buffer {
            // Public snapshot metadata must not undercount the retained allocation.
            snapshot.byte_size = usize::try_from(buffer.size).unwrap_or(usize::MAX);
        }

        let location = self.location_key(query);
        let entries = self.entries.entry(location).or_default();
        // Equal-distance lookup used to prefer the older duplicate. Replace it
        // explicitly, and release its retained bytes before charging the new IR.
        entries.retain(|entry| {
            if entry.query == query {
                self.gpu_bytes = self.gpu_bytes.saturating_sub(entry.snapshot.byte_size);
                false
            } else {
                true
            }
        });
        self.gpu_bytes = self.gpu_bytes.saturating_add(snapshot.byte_size);
        entries.push(IrCacheEntry {
            query,
            snapshot,
        });
        self.touch(self.key_for_query(query));
        self.evict_to_limit();
    }

    /// Drops all cached snapshots.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.lru.clear();
        self.gpu_bytes = 0;
    }

    /// Returns the number of retained IR snapshots.
    pub fn len(&self) -> usize {
        self.entries.values().map(Vec::len).sum()
    }

    /// Returns whether no IR snapshots are retained.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns GPU bytes currently retained by the cache.
    pub fn gpu_bytes(&self) -> usize {
        self.gpu_bytes
    }

    fn is_enabled(&self) -> bool {
        self.cfg.max_entries > 0
            && self.cfg.position_cell_meters.is_finite()
            && self.cfg.position_cell_meters > 0.0
            && self.cfg.reuse_distance_meters.is_finite()
            && self.cfg.reuse_distance_meters >= 0.0
            && self.cfg.max_gpu_cache_bytes > 0
    }

    fn can_reuse(&self, entry: &IrCacheEntry, query: IrCacheQuery) -> bool {
        entry.query.scene_version == query.scene_version
            && entry.query.source_position.distance(query.source_position)
                <= self.cfg.reuse_distance_meters
            && (entry.query.source_amplitude - query.source_amplitude).abs() <= f32::EPSILON
            && entry
                .query
                .listener_position
                .distance(query.listener_position)
                <= self.cfg.reuse_distance_meters
            && entry
                .query
                .listener_forward
                .normalize_or_zero()
                .dot(query.listener_forward.normalize_or_zero())
                >= self.cfg.listener_orientation_dot_threshold
            && entry
                .query
                .listener_up
                .normalize_or_zero()
                .dot(query.listener_up.normalize_or_zero())
                >= self.cfg.listener_orientation_dot_threshold
    }

    fn location_key(&self, query: IrCacheQuery) -> LocationCacheKey {
        LocationCacheKey {
            scene_version: query.scene_version,
            query_id: query.query_id,
            source_cell: quantize_position(query.source_position, self.cfg.position_cell_meters),
            listener_cell: quantize_position(
                query.listener_position,
                self.cfg.position_cell_meters,
            ),
        }
    }

    fn key_for_query(&self, query: IrCacheQuery) -> IrCacheKey {
        IrCacheKey {
            location: self.location_key(query),
            listener_forward_cell: quantize_direction(
                query.listener_forward,
                self.cfg.listener_orientation_dot_threshold,
            ),
            listener_up_cell: quantize_direction(
                query.listener_up,
                self.cfg.listener_orientation_dot_threshold,
            ),
        }
    }

    fn touch(&mut self, key: IrCacheKey) {
        self.lru.retain(|existing| *existing != key);
        self.lru.push_back(key);
    }

    fn evict_to_limit(&mut self) {
        while self.len() > self.cfg.max_entries || self.gpu_bytes > self.cfg.max_gpu_cache_bytes {
            let Some(oldest) = self.lru.pop_front() else {
                break;
            };
            let threshold = self.cfg.listener_orientation_dot_threshold;
            if let Some(entries) = self.entries.get_mut(&oldest.location) {
                // Several exact queries can share a quantized key. The LRU
                // contains that key only once, so evict its whole bucket; leaving
                // siblings behind would strand buffers outside the eviction list.
                entries.retain(|entry| {
                    let key = IrCacheKey {
                        location: oldest.location,
                        listener_forward_cell: quantize_direction(
                            entry.query.listener_forward,
                            threshold,
                        ),
                        listener_up_cell: quantize_direction(entry.query.listener_up, threshold),
                    };
                    if key == oldest {
                        self.gpu_bytes = self.gpu_bytes.saturating_sub(entry.snapshot.byte_size);
                        false
                    } else {
                        true
                    }
                });
                if entries.is_empty() {
                    self.entries.remove(&oldest.location);
                }
            }
        }
    }
}

fn reuse_score(entry: IrCacheQuery, query: IrCacheQuery) -> f32 {
    let forward_alignment = entry
        .listener_forward
        .normalize_or_zero()
        .dot(query.listener_forward.normalize_or_zero());
    let up_alignment = entry
        .listener_up
        .normalize_or_zero()
        .dot(query.listener_up.normalize_or_zero());
    entry
        .source_position
        .distance_squared(query.source_position)
        + entry
            .listener_position
            .distance_squared(query.listener_position)
        + (1.0 - forward_alignment)
        + (1.0 - up_alignment)
}

fn quantize_direction(direction: Vec3, dot_threshold: f32) -> [i32; 3] {
    if !dot_threshold.is_finite() {
        return [0; 3];
    }
    quantize_position(
        direction.normalize_or_zero(),
        (1.0 - dot_threshold).max(0.001),
    )
}

fn quantize_position(position: Vec3, cell_size: f32) -> [i32; 3] {
    [
        (position.x / cell_size).floor() as i32,
        (position.y / cell_size).floor() as i32,
        (position.z / cell_size).floor() as i32,
    ]
}

#[cfg(test)]
mod tests {
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
}
