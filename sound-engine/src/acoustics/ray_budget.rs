//! Scheduling of complete ray queries without changing their sample set.

/// Chooses disjoint launch sizes from recent wall cost, leaving time for audio.
/// This bounds work granularity, not driver latency or worst-case GPU execution.
pub(super) struct RayBatchBudget {
    /// Smallest batch, limiting a full query to at most sixteen ray stages.
    minimum: u32,
    /// Largest batch, preserving at least two stages for ordinary queries.
    maximum: u32,
    /// Current launch limit; the final range may contain fewer rays.
    limit: u32,
    /// Target wall time per trace stage, in milliseconds.
    target_ms: f64,
    /// Recent per-ray wall costs, including setup; zero entries are unobserved.
    costs: [f64; 8],
    /// Ring position replaced by the next measured stage.
    next_cost: usize,
}

impl RayBatchBudget {
    /// Starts conservatively, using 45% of an audio block as the trace target.
    /// The caller has already validated positive ray count, sample rate and block size.
    pub(super) fn new(rays: u32, sample_rate: u32, block_size: usize) -> Self {
        let minimum = rays.div_ceil(16).max(1);
        let maximum = rays.div_ceil(2).max(1);
        Self {
            minimum,
            maximum,
            limit: rays.div_ceil(8).max(1),
            target_ms: 450.0 * block_size as f64 / sample_rate as f64,
            costs: [0.0; 8],
            next_cost: 0,
        }
    }

    /// Returns a nonempty prefix of the remaining global ray-index range.
    pub(super) fn count(&self, remaining: u32) -> u32 {
        self.limit.min(remaining)
    }

    /// Updates granularity after a successful synchronous trace, preserving total rays.
    /// Retaining the slowest of eight stages prevents a cheap direction range from
    /// immediately scheduling a large batch into a more expensive part of the scene.
    pub(super) fn observe(&mut self, rays: u32, elapsed_ms: f64) {
        // A short final range (or direct-only query) mostly measures fixed
        // submission overhead, which must not throttle the next full query.
        if rays < self.minimum || !elapsed_ms.is_finite() || elapsed_ms < 0.0 {
            return;
        }
        self.costs[self.next_cost] = elapsed_ms.max(0.001) / f64::from(rays);
        self.next_cost = (self.next_cost + 1) % self.costs.len();
        let worst_cost = self.costs.iter().copied().fold(0.0_f64, f64::max);
        let proposed = (self.target_ms / worst_cost).floor() as u32;
        self.limit = proposed.clamp(self.minimum, self.maximum);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slow_stages_shrink_batches_and_expire_after_new_measurements() {
        let mut budget = RayBatchBudget::new(65_537, 48_000, 1024);
        assert_eq!(budget.count(65_537), 8193);
        budget.observe(8193, 40.0);
        assert_eq!(budget.count(65_537), 4097);
        for _ in 0..7 {
            budget.observe(4097, 0.1);
            assert_eq!(budget.count(65_537), 4097);
        }
        budget.observe(4097, 0.1);
        assert_eq!(budget.count(65_537), 32_769);
        assert_eq!(budget.count(13), 13);
    }

    #[test]
    fn changing_limits_cover_an_odd_query_once_with_bounded_stage_count() {
        let total = 65_537;
        let mut budget = RayBatchBudget::new(total, 48_000, 1024);
        let mut next = 0;
        let mut stages = 0;
        while next < total {
            let count = budget.count(total - next);
            assert!(count > 0 && next + count <= total);
            next += count;
            stages += 1;
            budget.observe(
                count,
                if stages % 2 == 0 {
                    1.0
                } else {
                    50.0
                },
            );
        }
        assert_eq!(next, total);
        assert!(stages <= 16);
    }

    #[test]
    fn short_tails_do_not_turn_fixed_setup_cost_into_full_query_cost() {
        let mut budget = RayBatchBudget::new(65_537, 48_000, 1024);
        budget.observe(8193, 1.0);
        let before = budget.count(65_537);
        budget.observe(1, 5.0);
        budget.observe(4097, f64::NAN);
        assert_eq!(budget.count(65_537), before);
    }
}
