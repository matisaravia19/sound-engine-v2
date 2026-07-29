use crate::acoustics::GpuIrSnapshot;
use crate::scene::SceneVersion;
use glam::Vec3;
use std::collections::{HashMap, VecDeque};

/// Runtime policy for acoustic impulse-response reuse.
#[derive(Debug, Clone, Copy)]
pub struct IrCacheConfig {
    /// Position quantization cell size in world meters.
    pub position_cell_meters: f32,
    /// Maximum source/listener movement that may reuse an IR inside a cell.
    pub reuse_distance_meters: f32,
    /// Dot-product tolerance for listener right-vector reuse.
    pub listener_right_dot_threshold: f32,
    /// Maximum cached IR snapshots retained per engine instance.
    pub max_entries: usize,
    /// Maximum GPU memory retained by cached IR buffers.
    pub max_gpu_cache_bytes: usize,
}

impl Default for IrCacheConfig {
    fn default() -> Self {
        Self {
            position_cell_meters: 0.25,
            reuse_distance_meters: 0.2,
            listener_right_dot_threshold: 0.98,
            max_entries: 32,
            max_gpu_cache_bytes: 512 * 1024 * 1024,
        }
    }
}

/// Source/listener state used to query the acoustic IR cache.
#[derive(Debug, Clone, Copy)]
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
    /// Listener right-ear direction in world space.
    pub listener_right: Vec3,
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
    listener_right_cell: [i32; 3],
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

        let location = self.location_key(query);
        let (index, snapshot) = {
            let entries = self.entries.get(&location)?;
            let (index, entry) = entries
                .iter()
                .enumerate()
                .find(|(_, entry)| self.can_reuse(entry, query))?;
            (index, entry.snapshot.clone())
        };

        self.touch(self.key_for_query(query));
        if let Some(entries) = self.entries.get_mut(&location) {
            let entry = entries.remove(index);
            entries.push(entry);
        }
        Some(snapshot)
    }

    /// Inserts a GPU IR for the given source/listener state.
    pub fn insert_ir(&mut self, query: IrCacheQuery, snapshot: GpuIrSnapshot) {
        if !self.is_enabled() {
            return;
        }

        let location = self.location_key(query);
        self.gpu_bytes = self.gpu_bytes.saturating_add(snapshot.byte_size);
        self.entries
            .entry(location)
            .or_default()
            .push(IrCacheEntry { query, snapshot });
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
            && entry.query.source_position.distance(query.source_position) <= self.cfg.reuse_distance_meters
            && (entry.query.source_amplitude - query.source_amplitude).abs() <= f32::EPSILON
            && entry.query.listener_position.distance(query.listener_position) <= self.cfg.reuse_distance_meters
            && entry
                .query
                .listener_right
                .normalize_or_zero()
                .dot(query.listener_right.normalize_or_zero())
                >= self.cfg.listener_right_dot_threshold
    }

    fn location_key(&self, query: IrCacheQuery) -> LocationCacheKey {
        LocationCacheKey {
            scene_version: query.scene_version,
            query_id: query.query_id,
            source_cell: quantize_position(query.source_position, self.cfg.position_cell_meters),
            listener_cell: quantize_position(query.listener_position, self.cfg.position_cell_meters),
        }
    }

    fn key_for_query(&self, query: IrCacheQuery) -> IrCacheKey {
        IrCacheKey {
            location: self.location_key(query),
            listener_right_cell: quantize_direction(query.listener_right, self.cfg.listener_right_dot_threshold),
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
            let threshold = self.cfg.listener_right_dot_threshold;
            if let Some(entries) = self.entries.get_mut(&oldest.location) {
                if let Some(index) = entries.iter().position(|entry| {
                    IrCacheKey {
                        location: oldest.location,
                        listener_right_cell: quantize_direction(entry.query.listener_right, threshold),
                    } == oldest
                }) {
                    let entry = entries.remove(index);
                    self.gpu_bytes = self.gpu_bytes.saturating_sub(entry.snapshot.byte_size);
                }
                if entries.is_empty() {
                    self.entries.remove(&oldest.location);
                }
            }
        }
    }
}

fn quantize_direction(direction: Vec3, dot_threshold: f32) -> [i32; 3] {
    if !dot_threshold.is_finite() {
        return [0; 3];
    }
    quantize_position(direction.normalize_or_zero(), (1.0 - dot_threshold).max(0.001))
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
    fn cache_respects_listener_direction() {
        let mut cache = IrCache::new(IrCacheConfig::default());
        let query = query(Vec3::X);
        cache.insert_ir(query, ir_snapshot(query));

        let mut rotated = query;
        rotated.listener_right = Vec3::Z;

        assert!(cache.get_ir(rotated).is_none());
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

    fn query(listener_right: Vec3) -> IrCacheQuery {
        IrCacheQuery {
            scene_version: 1,
            query_id: 0,
            source_position: Vec3::ZERO,
            source_amplitude: 1.0,
            listener_position: Vec3::Y,
            listener_right,
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
