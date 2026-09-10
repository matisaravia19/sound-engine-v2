//! Immutable source radiation tables applied to the pressure carried by rays.
use crate::core::error::{SoundError, SoundResult};
use glam::Quat;
use std::path::Path;

/// Engine-local immutable directivity asset. Zero selects an omnidirectional source.
pub type SourceDirectivityId = u32;

/// Source radiation asset and its local-to-world rotation. Local +X is forward,
/// +Y is up and +Z completes the right-handed frame. Assets live with the engine.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourceRadiation {
    /// Asset returned by this engine's `register_source_directivity`; zero is unity.
    pub profile: SourceDirectivityId,
    /// Unit quaternion rotating the source's local axes into world coordinates.
    pub orientation: Quat,
}

impl Default for SourceRadiation {
    fn default() -> Self {
        Self {
            profile: 0,
            orientation: Quat::IDENTITY,
        }
    }
}

impl SourceRadiation {
    /// Rejects invalid rotations before they reach a ray or an acoustic cache key.
    pub(crate) fn validate(self) -> SoundResult<()> {
        if !self.orientation.is_finite() || (self.orientation.length_squared() - 1.0).abs() > 1e-4 {
            return Err(SoundError::invalid_argument(
                "source orientation must be a finite unit quaternion",
            ));
        }
        Ok(())
    }

    /// Exact conservative cache identity; rotations never reuse a different profile.
    pub(crate) fn cache_key(self) -> [u32; 5] {
        let q = if self.profile == 0 {
            Quat::IDENTITY
        } else {
            self.orientation
        };
        let [x, y, z, w] = q.to_array().map(f32::to_bits);
        [self.profile, x, y, z, w]
    }
}

/// Measured or authored complex source-pressure multipliers on a front-pole grid.
/// No propagation paths or room responses are stored. Angular coordinates are
/// P in [0, 360) around +X (P=0 toward +Y), T in [0, 180] away from +X.
/// Registration copies/resamples the table to immutable engine-owned GPU storage.
#[derive(Debug, Clone)]
pub struct SourceDirectivity {
    /// Uniform periodic azimuth count, with no duplicated 360-degree row.
    pub(super) phi_count: u32,
    /// Uniform polar count, including both poles.
    pub(super) theta_count: u32,
    /// Strictly increasing positive frequency nodes in Hz.
    frequencies: Vec<f32>,
    /// Complex pressure in P-major, T-middle, frequency-minor order.
    pressure: Vec<[f32; 2]>,
}

impl SourceDirectivity {
    /// Validates a rectangular complex-pressure table. Polar values must agree
    /// for all azimuths. Frequency interpolation uses complex pressure versus
    /// log frequency; outside the measured range, the nearest endpoint is held.
    /// Multipliers are not automatically normalized or fitted to room responses.
    pub fn new(
        phi_count: u32,
        theta_count: u32,
        frequencies: Vec<f32>,
        pressure: Vec<[f32; 2]>,
    ) -> SoundResult<Self> {
        let size = (phi_count as usize)
            .checked_mul(theta_count as usize)
            .and_then(|n| n.checked_mul(frequencies.len()));
        if phi_count == 0
            || theta_count < 2
            || frequencies.len() < 2
            || size != Some(pressure.len())
            || frequencies.iter().any(|f| !f.is_finite() || *f <= 0.0)
            || frequencies.windows(2).any(|v| v[0] >= v[1])
            || pressure.iter().flatten().any(|v| !v.is_finite())
        {
            return Err(SoundError::invalid_argument(
                "invalid source directivity dimensions or pressure/frequency data",
            ));
        }
        let bands = frequencies.len();
        for phi in 1..phi_count as usize {
            for theta in [0, theta_count as usize - 1] {
                for band in 0..bands {
                    let reference = pressure[theta * bands + band];
                    if pressure[(phi * theta_count as usize + theta) * bands + band] != reference {
                        return Err(SoundError::invalid_argument(
                            "source directivity polar rows must be identical across azimuths",
                        ));
                    }
                }
            }
        }
        Ok(Self {
            phi_count,
            theta_count,
            frequencies,
            pressure,
        })
    }

    /// Loads DIR1: four-byte magic, three little-endian u32 counts (P,T,F),
    /// F float32 frequencies, then P*T*F pairs of float32 complex pressure.
    /// Data provenance and calibration conventions belong in a companion ledger.
    pub fn read(path: impl AsRef<Path>) -> SoundResult<Self> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    /// Decodes and validates the portable DIR1 representation without GPU access.
    pub fn from_bytes(bytes: &[u8]) -> SoundResult<Self> {
        if bytes.len() < 16 || &bytes[..4] != b"DIR1" {
            return Err(SoundError::invalid_argument(
                "invalid DIR1 source directivity header",
            ));
        }
        let word = |offset| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        let (p, t, f) = (word(4), word(8), word(12));
        let samples = (p as usize)
            .checked_mul(t as usize)
            .and_then(|n| n.checked_mul(f as usize));
        let start = (f as usize).checked_mul(4).and_then(|n| n.checked_add(16));
        let expected = samples
            .and_then(|n| n.checked_mul(8))
            .and_then(|n| start.and_then(|start| n.checked_add(start)));
        if expected != Some(bytes.len()) {
            return Err(SoundError::invalid_argument(
                "source directivity payload length mismatch",
            ));
        }
        let frequencies = (0..f as usize)
            .map(|i| f32::from_bits(word(16 + 4 * i)))
            .collect();
        let start = start.unwrap();
        let pressure = (0..samples.unwrap())
            .map(|i| {
                [
                    f32::from_bits(word(start + 8 * i)),
                    f32::from_bits(word(start + 8 * i + 4)),
                ]
            })
            .collect();
        Self::new(p, t, frequencies, pressure)
    }

    /// Resamples only source calibration to the tracer's pressure bands at setup.
    /// The same immutable table is shared by every ray using this source asset.
    pub(super) fn resample(&self, nodes: &[f32]) -> Vec<[f32; 2]> {
        let mut result =
            Vec::with_capacity(self.phi_count as usize * self.theta_count as usize * nodes.len());
        for direction in self.pressure.chunks_exact(self.frequencies.len()) {
            for &f in nodes {
                let hi = self
                    .frequencies
                    .partition_point(|&node| node < f)
                    .min(self.frequencies.len() - 1);
                let lo = hi.saturating_sub(1);
                let fraction = if hi == lo {
                    0.0
                } else {
                    ((f.ln() - self.frequencies[lo].ln())
                        / (self.frequencies[hi].ln() - self.frequencies[lo].ln()))
                    .clamp(0.0, 1.0)
                };
                result.push(std::array::from_fn(|c| {
                    direction[lo][c] * (1.0 - fraction) + direction[hi][c] * fraction
                }));
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pressure_resampling_retains_phase_and_holds_endpoints() {
        let table = SourceDirectivity::new(
            1,
            2,
            vec![100.0, 400.0],
            vec![[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        )
        .unwrap();
        let values = table.resample(&[20.0, 200.0, 2000.0]);
        assert_eq!(values[0], [1.0, 0.0]);
        assert!((values[1][0] - 0.5).abs() < 1e-6 && (values[1][1] - 0.5).abs() < 1e-6);
        assert_eq!(values[2], [0.0, 1.0]);
    }

    #[test]
    fn invalid_calibration_and_rotation_are_rejected() {
        assert!(SourceDirectivity::new(1, 2, vec![400.0, 100.0], vec![[1.0, 0.0]; 4]).is_err());
        assert!(SourceDirectivity::from_bytes(b"DIR1").is_err());
        assert!(
            SourceRadiation {
                profile: 1,
                orientation: Quat::from_array([0.0; 4])
            }
            .validate()
            .is_err()
        );
        assert!(
            SourceDirectivity::new(
                2,
                2,
                vec![100.0, 400.0],
                vec![
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.0]
                ]
            )
            .is_err()
        );
    }
}
