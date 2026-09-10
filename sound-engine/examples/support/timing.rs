//! Common latency conventions for benchmark output.
#![allow(dead_code)]
use std::time::Duration;

/// Returns the upper empirical quantile at `ceil((n - 1) * fraction)`.
/// The caller must supply a nonempty sorted slice and a fraction in [0, 1].
pub fn percentile<T: Copy>(sorted: &[T], fraction: f64) -> T {
    assert!(!sorted.is_empty(), "latency samples must not be empty");
    assert!((0.0..=1.0).contains(&fraction), "quantile must be in [0, 1]");
    sorted[((sorted.len() - 1) as f64 * fraction).ceil() as usize]
}

/// Converts a measured duration to milliseconds without integer truncation.
pub fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}
