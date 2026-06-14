use crate::acoustics::IrSnapshot;
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
}

impl Default for IrCacheConfig {
    fn default() -> Self {
        Self {
            position_cell_meters: 0.25,
            reuse_distance_meters: 0.2,
            listener_right_dot_threshold: 0.98,
            max_entries: 32,
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

/// Small LRU cache for immutable impulse-response snapshots.
///
/// Entries are invalidated by scene version and quantized source/listener pose.
/// The cache owns snapshots and returns clones so audio/control threads can
/// keep using previous IRs after the cache evicts an entry.
pub struct IrCache {
    cfg: IrCacheConfig,
    entries: HashMap<IrCacheKey, IrCacheEntry>,
    lru: VecDeque<IrCacheKey>,
}

#[derive(Debug, Clone)]
struct IrCacheEntry {
    query: IrCacheQuery,
    snapshot: IrSnapshot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct IrCacheKey {
    scene_version: SceneVersion,
    query_id: u32,
    source_cell: [i32; 3],
    listener_cell: [i32; 3],
}

impl IrCache {
    /// Creates an empty IR cache with the provided reuse policy.
    pub fn new(cfg: IrCacheConfig) -> Self {
        Self {
            cfg,
            entries: HashMap::new(),
            lru: VecDeque::new(),
        }
    }

    /// Returns a reusable IR snapshot when movement and orientation remain within tolerance.
    pub fn get(&mut self, query: IrCacheQuery) -> Option<IrSnapshot> {
        if !self.is_enabled() {
            return None;
        }

        let key = self.key(query);
        let snapshot = {
            let entry = self.entries.get(&key)?;
            if !self.can_reuse(entry, query) {
                return None;
            }
            entry.snapshot.clone()
        };

        self.touch(key);
        Some(snapshot)
    }

    /// Inserts or replaces an IR snapshot for the given source/listener state.
    pub fn insert(&mut self, query: IrCacheQuery, snapshot: IrSnapshot) {
        if !self.is_enabled() {
            return;
        }

        let key = self.key(query);
        self.entries.insert(key, IrCacheEntry { query, snapshot });
        self.touch(key);
        self.evict_to_limit();
    }

    /// Drops all cached snapshots.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.lru.clear();
    }

    /// Returns the number of retained snapshots.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    fn is_enabled(&self) -> bool {
        self.cfg.max_entries > 0
            && self.cfg.position_cell_meters.is_finite()
            && self.cfg.position_cell_meters > 0.0
            && self.cfg.reuse_distance_meters.is_finite()
            && self.cfg.reuse_distance_meters >= 0.0
    }

    fn can_reuse(&self, entry: &IrCacheEntry, query: IrCacheQuery) -> bool {
        entry.query.scene_version == query.scene_version
            && entry.query.source_position.distance(query.source_position) <= self.cfg.reuse_distance_meters
            && (entry.query.source_energy - query.source_energy).abs() <= f32::EPSILON
            && entry.query.listener_position.distance(query.listener_position) <= self.cfg.reuse_distance_meters
            && entry
                .query
                .listener_right
                .normalize_or_zero()
                .dot(query.listener_right.normalize_or_zero())
                >= self.cfg.listener_right_dot_threshold
    }

    fn key(&self, query: IrCacheQuery) -> IrCacheKey {
        IrCacheKey {
            scene_version: query.scene_version,
            query_id: query.query_id,
            source_cell: quantize_position(query.source_position, self.cfg.position_cell_meters),
            listener_cell: quantize_position(query.listener_position, self.cfg.position_cell_meters),
        }
    }

    fn touch(&mut self, key: IrCacheKey) {
        self.lru.retain(|existing| *existing != key);
        self.lru.push_back(key);
    }

    fn evict_to_limit(&mut self) {
        while self.entries.len() > self.cfg.max_entries {
            let Some(oldest) = self.lru.pop_front() else {
                break;
            };
            self.entries.remove(&oldest);
        }
    }
}

fn quantize_position(position: Vec3, cell_size: f32) -> [i32; 3] {
    [
        (position.x / cell_size).floor() as i32,
        (position.y / cell_size).floor() as i32,
        (position.z / cell_size).floor() as i32,
    ]
}
