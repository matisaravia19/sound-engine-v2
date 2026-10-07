use crate::acoustics::{AcousticQuery, GpuIrSnapshot};
use crate::scene::SceneVersion;
use std::collections::VecDeque;

/// Shared public policy for acoustic impulse-response reuse.
pub use crate::core::config::IrCacheConfig;

/// Small GPU IR LRU cache keyed by scene and oriented source/listener state.
///
/// Entries retain reference-counted GPU buffers; eviction releases only the
/// cache's ownership and requires neither readback nor a hit-level snapshot.
pub struct IrCache {
    config: IrCacheConfig,
    // Oldest to most recently used; lookup and eviction share one ordering.
    entries: VecDeque<IrCacheEntry>,
    gpu_bytes: usize,
}

struct IrCacheEntry {
    query: AcousticQuery,
    snapshot: GpuIrSnapshot,
}

impl IrCache {
    /// Creates an empty IR cache with the provided reuse policy.
    pub fn new(config: IrCacheConfig) -> Self {
        Self {
            config,
            entries: VecDeque::new(),
            gpu_bytes: 0,
        }
    }

    /// Returns a reusable GPU IR when the full acoustic query remains within tolerance.
    pub fn get_ir(
        &mut self,
        scene_version: SceneVersion,
        query: AcousticQuery,
    ) -> Option<GpuIrSnapshot> {
        if !self.is_enabled() {
            return None;
        }

        // The cache is small and reuse depends on exact distances and orientation,
        // so a linear search avoids maintaining an index that cannot decide a hit.
        let index = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.snapshot.scene_version == scene_version)
            .filter_map(|(index, entry)| {
                self.reuse_score(entry.query, query)
                    .map(|score| (index, score))
            })
            .min_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index)?;
        let entry = self.entries.remove(index)?;
        let snapshot = entry.snapshot.clone();
        self.entries.push_back(entry);
        Some(snapshot)
    }

    /// Inserts or replaces the exact source/listener state; inconsistent snapshot
    /// scene/query metadata is rejected without changing the cache.
    pub fn insert_ir(
        &mut self,
        scene_version: SceneVersion,
        query: AcousticQuery,
        mut snapshot: GpuIrSnapshot,
    ) {
        if !self.is_enabled()
            || snapshot.scene_version != scene_version
            || snapshot.query_id != query.query_id
        {
            return;
        }
        if let Some(buffer) = &snapshot.buffer {
            // Public snapshot metadata must not undercount the retained allocation.
            snapshot.byte_size = usize::try_from(buffer.size).unwrap_or(usize::MAX);
        }

        // Replace the exact query before accounting for its new allocation.
        self.entries.retain(|entry| {
            if entry.snapshot.scene_version == scene_version && entry.query == query {
                self.gpu_bytes = self.gpu_bytes.saturating_sub(entry.snapshot.byte_size);
                false
            } else {
                true
            }
        });
        self.gpu_bytes = self.gpu_bytes.saturating_add(snapshot.byte_size);
        self.entries.push_back(IrCacheEntry {
            query,
            snapshot,
        });
        while self.len() > self.config.max_entries
            || self.gpu_bytes > self.config.max_gpu_cache_bytes
        {
            let Some(oldest) = self.entries.pop_front() else {
                break;
            };
            self.gpu_bytes = self.gpu_bytes.saturating_sub(oldest.snapshot.byte_size);
        }
    }

    /// Drops all cached snapshots.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.gpu_bytes = 0;
    }

    /// Returns the number of retained IR snapshots.
    pub fn len(&self) -> usize {
        self.entries.len()
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
        self.config.max_entries > 0
            && self.config.reuse_distance_meters.is_finite()
            && self.config.reuse_distance_meters >= 0.0
            && self.config.max_gpu_cache_bytes > 0
    }

    fn reuse_score(&self, cached: AcousticQuery, requested: AcousticQuery) -> Option<f32> {
        if cached.query_id != requested.query_id {
            return None;
        }
        let source_distance = cached
            .source
            .position
            .distance_squared(requested.source.position);
        let listener_distance = cached
            .listener
            .position
            .distance_squared(requested.listener.position);
        let forward_alignment = cached
            .listener
            .forward
            .normalize_or_zero()
            .dot(requested.listener.forward.normalize_or_zero());
        let up_alignment = cached
            .listener
            .up
            .normalize_or_zero()
            .dot(requested.listener.up.normalize_or_zero());
        let reusable = (cached.source.amplitude - requested.source.amplitude).abs() <= f32::EPSILON
            && source_distance.sqrt() <= self.config.reuse_distance_meters
            && listener_distance.sqrt() <= self.config.reuse_distance_meters
            && forward_alignment >= self.config.listener_orientation_dot_threshold
            && up_alignment >= self.config.listener_orientation_dot_threshold;
        reusable.then_some(
            source_distance + listener_distance + (1.0 - forward_alignment) + (1.0 - up_alignment),
        )
    }
}

#[cfg(test)]
#[path = "../../tests/unit/acoustics/cache.rs"]
mod tests;
