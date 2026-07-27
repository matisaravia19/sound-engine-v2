use crate::acoustics::{GpuAcousticHitSnapshot, GpuIrSnapshot};
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
    /// Maximum GPU memory retained by cached hit and IR buffers.
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
    /// Acoustic source energy baked into the cached IR.
    pub source_energy: f32,
    /// Listener position in world meters.
    pub listener_position: Vec3,
    /// Listener right-ear direction in world space.
    pub listener_right: Vec3,
}

/// Small LRU cache for acoustic hits and direction-specific IR snapshots.
///
/// Hit entries are invalidated by scene version and quantized source/listener
/// position. IR entries add listener orientation tolerance on top, so turning
/// the listener can rebuild an IR from cached hits without relaunching tracing.
pub struct IrCache {
    cfg: IrCacheConfig,
    hit_entries: HashMap<LocationCacheKey, HitCacheEntry>,
    hit_lru: VecDeque<LocationCacheKey>,
    ir_entries: HashMap<LocationCacheKey, Vec<IrCacheEntry>>,
    ir_lru: VecDeque<IrCacheKey>,
    gpu_bytes: usize,
}

#[derive(Clone)]
struct HitCacheEntry {
    query: IrCacheQuery,
    snapshot: GpuAcousticHitSnapshot,
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
            hit_entries: HashMap::new(),
            hit_lru: VecDeque::new(),
            ir_entries: HashMap::new(),
            ir_lru: VecDeque::new(),
            gpu_bytes: 0,
        }
    }

    /// Returns reusable hit data when scene, source, and listener position remain within tolerance.
    pub fn get_hits(&mut self, query: IrCacheQuery) -> Option<GpuAcousticHitSnapshot> {
        if !self.is_enabled() {
            return None;
        }

        let key = self.location_key(query);
        let snapshot = {
            let entry = self.hit_entries.get(&key)?;
            if !self.can_reuse_hits(entry, query) {
                return None;
            }
            entry.snapshot.clone()
        };

        self.touch_hits(key);
        Some(snapshot)
    }

    /// Inserts or replaces hit data for the given source/listener location state.
    pub fn insert_hits(&mut self, query: IrCacheQuery, snapshot: GpuAcousticHitSnapshot) {
        if !self.is_enabled() {
            return;
        }

        let key = self.location_key(query);
        if let Some(previous) = self.hit_entries.insert(key, HitCacheEntry { query, snapshot }) {
            self.gpu_bytes = self.gpu_bytes.saturating_sub(previous.snapshot.byte_size);
        }
        if let Some(entry) = self.hit_entries.get(&key) {
            self.gpu_bytes = self.gpu_bytes.saturating_add(entry.snapshot.byte_size);
        }
        self.touch_hits(key);
        self.evict_to_limit();
    }

    /// Returns a reusable IR snapshot when location and listener orientation remain within tolerance.
    pub fn get_ir(&mut self, query: IrCacheQuery) -> Option<GpuIrSnapshot> {
        if !self.is_enabled() {
            return None;
        }

        let location = self.location_key(query);
        let (index, snapshot) = {
            let entries = self.ir_entries.get(&location)?;
            let (index, entry) = entries
                .iter()
                .enumerate()
                .find(|(_, entry)| self.can_reuse_ir(entry, query))?;
            (index, entry.snapshot.clone())
        };

        self.touch_ir(self.ir_key_for_query(query));
        if let Some(entries) = self.ir_entries.get_mut(&location) {
            let entry = entries.remove(index);
            entries.push(entry);
        }
        Some(snapshot)
    }

    /// Inserts or replaces an IR snapshot for the given source/listener direction state.
    pub fn insert_ir(&mut self, query: IrCacheQuery, snapshot: GpuIrSnapshot) {
        if !self.is_enabled() {
            return;
        }

        let location = self.location_key(query);
        self.gpu_bytes = self.gpu_bytes.saturating_add(snapshot.byte_size);
        self.ir_entries
            .entry(location)
            .or_default()
            .push(IrCacheEntry { query, snapshot });
        self.touch_ir(self.ir_key_for_query(query));
        self.evict_to_limit();
    }

    /// Drops all cached snapshots.
    pub fn clear(&mut self) {
        self.hit_entries.clear();
        self.hit_lru.clear();
        self.ir_entries.clear();
        self.ir_lru.clear();
        self.gpu_bytes = 0;
    }

    /// Returns the number of retained IR snapshots.
    pub fn len(&self) -> usize {
        self.ir_entries.values().map(Vec::len).sum()
    }

    /// Returns the number of retained hit snapshots.
    pub fn hit_len(&self) -> usize {
        self.hit_entries.len()
    }

    /// Returns GPU cache bytes currently retained by this cache.
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

    fn can_reuse_hits(&self, entry: &HitCacheEntry, query: IrCacheQuery) -> bool {
        entry.query.scene_version == query.scene_version
            && entry.query.source_position.distance(query.source_position) <= self.cfg.reuse_distance_meters
            && (entry.query.source_energy - query.source_energy).abs() <= f32::EPSILON
            && entry.query.listener_position.distance(query.listener_position) <= self.cfg.reuse_distance_meters
    }

    fn can_reuse_ir(&self, entry: &IrCacheEntry, query: IrCacheQuery) -> bool {
        self.can_reuse_location(entry.query, query)
            && self.can_reuse_listener_right(entry.query.listener_right, query.listener_right)
    }

    fn can_reuse_listener_right(&self, cached: Vec3, requested: Vec3) -> bool {
        cached.normalize_or_zero().dot(requested.normalize_or_zero()) >= self.cfg.listener_right_dot_threshold
    }

    fn can_reuse_location(&self, cached: IrCacheQuery, requested: IrCacheQuery) -> bool {
        cached.scene_version == requested.scene_version
            && cached.source_position.distance(requested.source_position) <= self.cfg.reuse_distance_meters
            && (cached.source_energy - requested.source_energy).abs() <= f32::EPSILON
            && cached.listener_position.distance(requested.listener_position) <= self.cfg.reuse_distance_meters
    }

    fn location_key(&self, query: IrCacheQuery) -> LocationCacheKey {
        LocationCacheKey {
            scene_version: query.scene_version,
            query_id: query.query_id,
            source_cell: quantize_position(query.source_position, self.cfg.position_cell_meters),
            listener_cell: quantize_position(query.listener_position, self.cfg.position_cell_meters),
        }
    }

    fn ir_key_for_query(&self, query: IrCacheQuery) -> IrCacheKey {
        IrCacheKey {
            location: self.location_key(query),
            listener_right_cell: quantize_direction(query.listener_right, self.cfg.listener_right_dot_threshold),
        }
    }

    fn touch_hits(&mut self, key: LocationCacheKey) {
        self.hit_lru.retain(|existing| *existing != key);
        self.hit_lru.push_back(key);
    }

    fn touch_ir(&mut self, key: IrCacheKey) {
        self.ir_lru.retain(|existing| *existing != key);
        self.ir_lru.push_back(key);
    }

    fn evict_to_limit(&mut self) {
        while self.hit_entries.len() > self.cfg.max_entries
            || self.len() > self.cfg.max_entries
            || self.gpu_bytes > self.cfg.max_gpu_cache_bytes
        {
            if self.len() > self.cfg.max_entries || self.should_evict_ir_first() {
                if !self.evict_oldest_ir() && !self.evict_oldest_hits() {
                    break;
                }
            } else if !self.evict_oldest_hits() && !self.evict_oldest_ir() {
                break;
            }
        }
    }

    fn should_evict_ir_first(&self) -> bool {
        self.hit_entries.len() <= self.cfg.max_entries && !self.ir_lru.is_empty()
    }

    fn evict_oldest_hits(&mut self) -> bool {
        let Some(oldest) = self.hit_lru.pop_front() else {
            return false;
        };
        if let Some(entry) = self.hit_entries.remove(&oldest) {
            self.gpu_bytes = self.gpu_bytes.saturating_sub(entry.snapshot.byte_size);
        }
        if let Some(entries) = self.ir_entries.remove(&oldest) {
            for entry in entries {
                self.gpu_bytes = self.gpu_bytes.saturating_sub(entry.snapshot.byte_size);
            }
        }
        self.ir_lru.retain(|key| key.location != oldest);
        true
    }

    fn evict_oldest_ir(&mut self) -> bool {
        let Some(oldest) = self.ir_lru.pop_front() else {
            return false;
        };
        let listener_right_dot_threshold = self.cfg.listener_right_dot_threshold;
        if let Some(entries) = self.ir_entries.get_mut(&oldest.location) {
            if let Some(index) = entries.iter().position(|entry| {
                IrCacheKey {
                    location: oldest.location,
                    listener_right_cell: quantize_direction(entry.query.listener_right, listener_right_dot_threshold),
                } == oldest
            }) {
                let entry = entries.remove(index);
                self.gpu_bytes = self.gpu_bytes.saturating_sub(entry.snapshot.byte_size);
            }
            if entries.is_empty() {
                self.ir_entries.remove(&oldest.location);
            }
            return true;
        }
        true
    }
}

fn quantize_direction(direction: Vec3, dot_threshold: f32) -> [i32; 3] {
    if !dot_threshold.is_finite() {
        return [0; 3];
    }
    let cell_size = (1.0 - dot_threshold).max(0.001);
    quantize_position(direction.normalize_or_zero(), cell_size)
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
    fn hit_cache_ignores_listener_direction() {
        let mut cache = IrCache::new(IrCacheConfig::default());
        let query = query(Vec3::X);
        let snapshot = hit_snapshot(query);
        cache.insert_hits(query, snapshot.clone());

        let mut rotated = query;
        rotated.listener_right = Vec3::Z;

        assert!(cache.get_hits(rotated).is_some());
    }

    #[test]
    fn ir_cache_respects_listener_direction() {
        let mut cache = IrCache::new(IrCacheConfig {
            listener_right_dot_threshold: 0.98,
            ..IrCacheConfig::default()
        });
        let query = query(Vec3::X);
        cache.insert_ir(query, ir_snapshot(query));

        let mut rotated = query;
        rotated.listener_right = Vec3::Z;

        assert!(cache.get_ir(rotated).is_none());
    }

    #[test]
    fn hit_cache_rejects_moved_locations() {
        let mut cache = IrCache::new(IrCacheConfig::default());
        let query = query(Vec3::X);
        cache.insert_hits(query, hit_snapshot(query));

        let mut moved = query;
        moved.listener_position = Vec3::new(10.0, 0.0, 0.0);

        assert!(cache.get_hits(moved).is_none());
    }

    #[test]
    fn gpu_byte_budget_evicts_old_entries() {
        let mut cache = IrCache::new(IrCacheConfig {
            max_gpu_cache_bytes: 96,
            ..IrCacheConfig::default()
        });
        let first = query(Vec3::X);
        let mut second = query(Vec3::X);
        second.listener_position = Vec3::new(1.0, 0.0, 0.0);

        cache.insert_hits(first, hit_snapshot(first));
        cache.insert_ir(first, ir_snapshot(first));
        cache.insert_hits(second, hit_snapshot(second));

        assert!(cache.gpu_bytes() <= 96);
        assert!(cache.get_hits(first).is_none());
        assert!(cache.get_hits(second).is_some());
    }

    fn query(listener_right: Vec3) -> IrCacheQuery {
        IrCacheQuery {
            scene_version: 1,
            query_id: 0,
            source_position: Vec3::ZERO,
            source_energy: 1.0,
            listener_position: Vec3::Y,
            listener_right,
        }
    }

    fn hit_snapshot(query: IrCacheQuery) -> GpuAcousticHitSnapshot {
        GpuAcousticHitSnapshot {
            buffer: None,
            hit_capacity: 1,
            byte_size: 64,
            scene_version: query.scene_version,
            query_id: query.query_id,
            source_position: query.source_position,
            source_energy: query.source_energy,
            listener_position: query.listener_position,
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
