//! Offline accuracy/cost reference for deterministic first-order BTM diffraction.
//!
//! The production engine uses procedural edge geometry in its ray-tracing
//! pipeline. This standalone CPU calculation is only a mathematical comparison;
//! it is not compiled into or dispatched by the runtime transport path.

#[path = "support/timing.rs"]
mod timing;
use timing::{milliseconds, percentile};

use std::f64::consts::PI;
use std::time::Instant;

const SAMPLE_RATE: f64 = 48_000.0;
const SAMPLE_COUNT: usize = 16_384;
const SPEED_OF_SOUND: f64 = 343.0;
const WEDGE_INDEX: f64 = 2.0 / 3.0;
const SOURCE_RADIUS: f64 = 3.605_551_275_463_989;
const RECEIVER_RADIUS: f64 = 3.605_551_275_463_989;
const SOURCE_ANGLE: f64 = 0.197_395_559_849_880_78;
const RECEIVER_ANGLE: f64 = 4.515_789_093_296_905;
const EDGE_START: f64 = -20.0;
const EDGE_END: f64 = 20.0;
const FREQUENCIES_HZ: &[f64] = &[125.0, 250.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0];
const RESOLUTIONS: &[usize] = &[512, 2_048, 8_192, 131_072];
const REPEATS: usize = 101;

#[derive(Clone, Copy)]
struct Complex {
    re: f64,
    im: f64,
}

fn main() {
    let reference = build_btm_ir(131_072);
    println!("segments,max_spectral_relative_error,p50_ms,p95_ms,p99_ms,max_ms");
    for &segments in RESOLUTIONS {
        // Warm instruction/data caches before recording latency tails.
        let _ = build_btm_ir(segments);
        let mut durations = Vec::with_capacity(REPEATS);
        let mut candidate = Vec::new();
        for _ in 0..REPEATS {
            let start = Instant::now();
            candidate = build_btm_ir(segments);
            durations.push(start.elapsed());
        }
        durations.sort_unstable();
        let maximum_error = FREQUENCIES_HZ
            .iter()
            .map(|&frequency| {
                let measured = transfer(&candidate, frequency);
                let expected = transfer(&reference, frequency);
                complex_distance(measured, expected) / magnitude(expected).max(1.0e-12)
            })
            .fold(0.0_f64, f64::max);
        println!(
            "{segments},{maximum_error:.9},{:.6},{:.6},{:.6},{:.6}",
            milliseconds(percentile(&durations, 0.50)),
            milliseconds(percentile(&durations, 0.95)),
            milliseconds(percentile(&durations, 0.99)),
            milliseconds(*durations.last().unwrap()),
        );
    }
}

fn build_btm_ir(segment_count: usize) -> Vec<f64> {
    let mut ir = vec![0.0; SAMPLE_COUNT];
    let segment_length = (EDGE_END - EDGE_START) / segment_count as f64;
    for segment in 0..segment_count {
        // Midpoints avoid sampling the integrable onset singularity exactly.
        let edge_coordinate = EDGE_START + (segment as f64 + 0.5) * segment_length;
        let source_distance = SOURCE_RADIUS.hypot(edge_coordinate);
        let receiver_distance = RECEIVER_RADIUS.hypot(edge_coordinate);
        let eta_argument = (source_distance * receiver_distance
            + edge_coordinate * edge_coordinate)
            / (SOURCE_RADIUS * RECEIVER_RADIUS);
        let eta = eta_argument.max(1.0).acosh();

        let mut beta_sum = 0.0;
        for source_sign in [-1.0, 1.0] {
            for receiver_sign in [-1.0, 1.0] {
                let phi = PI + source_sign * SOURCE_ANGLE + receiver_sign * RECEIVER_ANGLE;
                beta_sum += (WEDGE_INDEX * phi).sin()
                    / ((WEDGE_INDEX * eta).cosh() - (WEDGE_INDEX * phi).cos());
            }
        }
        // This coefficient uses the same source convention whose direct field
        // is 1/r; no additional 4*pi calibration is required.
        let pressure = -WEDGE_INDEX / (4.0 * PI) * beta_sum * segment_length
            / (source_distance * receiver_distance);
        let sample_position = (source_distance + receiver_distance) / SPEED_OF_SOUND * SAMPLE_RATE;
        deposit_fractional(&mut ir, sample_position, pressure);
    }
    ir
}

fn deposit_fractional(ir: &mut [f64], position: f64, pressure: f64) {
    let base = position.floor() as usize;
    let fraction = position - base as f64;
    if let Some(sample) = ir.get_mut(base) {
        *sample += pressure * (1.0 - fraction);
    }
    if let Some(sample) = ir.get_mut(base + 1) {
        *sample += pressure * fraction;
    }
}

fn transfer(ir: &[f64], frequency: f64) -> Complex {
    let angular_step = -2.0 * PI * frequency / SAMPLE_RATE;
    ir.iter().enumerate().fold(
        Complex {
            re: 0.0,
            im: 0.0,
        },
        |sum, (index, &sample)| {
            let phase = angular_step * index as f64;
            Complex {
                re: sum.re + sample * phase.cos(),
                im: sum.im + sample * phase.sin(),
            }
        },
    )
}

fn complex_distance(a: Complex, b: Complex) -> f64 {
    (a.re - b.re).hypot(a.im - b.im)
}

fn magnitude(value: Complex) -> f64 {
    value.re.hypot(value.im)
}
