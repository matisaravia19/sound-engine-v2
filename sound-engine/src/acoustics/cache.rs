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
#[path = "../../tests/unit/acoustics/cache.rs"]
mod tests;
