use super::spatial::ValidatedAcousticQuery;
use crate::acoustics::{AcousticQuery, GpuIrSnapshot};
use crate::core::error::SoundResult;
use crate::scene::SceneVersion;
use glam::Vec3;
use std::collections::VecDeque;

/// Shared public policy for acoustic impulse-response reuse.
pub use crate::core::config::IrCacheConfig;

/// Small GPU IR LRU cache keyed by scene and oriented source/listener state.
///
/// Entries retain reference-counted GPU buffers; eviction releases only the
/// cache's ownership and requires neither readback nor a hit-level snapshot.
pub(super) struct IrCache {
    config: IrCacheConfig,
    // Oldest to most recently used; lookup and eviction share one ordering.
    entries: VecDeque<GpuIrSnapshot>,
    gpu_bytes: u64,
}

impl IrCache {
    /// Creates an empty IR cache with the provided reuse policy.
    pub(super) fn new(config: IrCacheConfig) -> SoundResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            entries: VecDeque::new(),
            gpu_bytes: 0,
        })
    }

    /// Returns a reusable GPU IR when the full acoustic query remains within tolerance.
    pub(super) fn get_ir(
        &mut self,
        scene_version: SceneVersion,
        query: ValidatedAcousticQuery,
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
            .filter(|(_, entry)| entry.scene_version() == scene_version)
            .filter_map(|(index, entry)| {
                self.reuse_score(entry.query(), query.as_query())
                    .map(|score| (index, score))
            })
            .min_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index)?;
        let snapshot = self.entries.remove(index)?;
        self.entries.push_back(snapshot.clone());
        Some(snapshot)
    }

    /// Retains a response using its immutable scene/query provenance as the key.
    /// Responses larger than the entire budget leave the existing cache unchanged.
    pub(super) fn insert_ir(&mut self, snapshot: GpuIrSnapshot) {
        let bytes = snapshot.byte_size();
        let budget = self.config.max_gpu_cache_bytes as u64;
        if !self.is_enabled() || bytes > budget {
            return;
        }

        self.entries.retain(|entry| {
            if entry.scene_version() == snapshot.scene_version()
                && entry.query() == snapshot.query()
            {
                self.gpu_bytes -= entry.byte_size();
                false
            } else {
                true
            }
        });
        // Evict before insertion so accounting stays within the budget even
        // when adding the next allocation would overflow the byte counter.
        while self.entries.len() >= self.config.max_entries || self.gpu_bytes > budget - bytes {
            let oldest = self
                .entries
                .pop_front()
                .expect("Only nonempty caches can exceed their limits");
            self.gpu_bytes -= oldest.byte_size();
        }
        self.gpu_bytes += bytes;
        self.entries.push_back(snapshot);
    }

    /// Drops all cached snapshots.
    pub(super) fn clear(&mut self) {
        self.entries.clear();
        self.gpu_bytes = 0;
    }

    fn is_enabled(&self) -> bool {
        self.config.max_entries > 0 && self.config.max_gpu_cache_bytes > 0
    }

    fn reuse_score(&self, cached: AcousticQuery, requested: AcousticQuery) -> Option<f32> {
        if cached.query_id != requested.query_id
            || cached.source.amplitude != requested.source.amplitude
        {
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
        let forward_alignment =
            orientation_alignment(cached.listener.forward, requested.listener.forward);
        let up_alignment = orientation_alignment(cached.listener.up, requested.listener.up);
        let reusable = source_distance.sqrt() <= self.config.reuse_distance_meters
            && listener_distance.sqrt() <= self.config.reuse_distance_meters
            && forward_alignment >= self.config.listener_orientation_dot_threshold
            && up_alignment >= self.config.listener_orientation_dot_threshold;
        reusable.then_some(
            source_distance + listener_distance + (1.0 - forward_alignment) + (1.0 - up_alignment),
        )
    }
}

fn orientation_alignment(left: Vec3, right: Vec3) -> f32 {
    // Queries already have unit axes. Equal axes must match even at threshold 1;
    // f32 normalization and dot products can round to either side of that limit.
    if left == right {
        1.0
    } else {
        left.dot(right).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/acoustics/cache.rs"]
mod tests;
