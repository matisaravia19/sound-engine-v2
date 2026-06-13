use crate::error::{SoundError, SoundResult};
use crate::scene::SceneVersion;

/// Maximum number of path contributions downloaded for one acoustic query.
pub const MAX_CONTRIBUTIONS_PER_QUERY: usize = 4096;

/// Configuration for CPU mono IR construction.
#[derive(Debug, Clone, Copy)]
pub struct IrConfig {
    /// Sample rate used for converting arrival time to sample index.
    pub sample_rate: u32,
    /// Number of mono samples in the generated IR.
    pub ir_len_samples: u32,
}

/// One acoustic contribution produced by ray tracing.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ContributionRecord {
    /// `x = arrival time seconds`, `y = linear gain`; remaining lanes reserved.
    pub timing_gain: [f32; 4],
    /// `xyz = incoming direction`, `w = path order`.
    pub direction_order: [f32; 4],
}

unsafe impl bytemuck::Zeroable for ContributionRecord {}
unsafe impl bytemuck::Pod for ContributionRecord {}

/// CPU snapshot ready for convolution after IR construction.
#[derive(Debug, Clone)]
pub struct IrSnapshot {
    /// Sample rate used to generate `samples`.
    pub sample_rate: u32,
    /// Mono impulse-response samples.
    pub samples: Vec<f32>,
    /// Sum of squared sample values for diagnostics and sanity checks.
    pub energy: f32,
    /// Scene version represented by this IR.
    pub scene_version: SceneVersion,
    /// Query identifier copied from the acoustic request.
    pub query_id: u32,
}

/// CPU helper that bins downloaded path contributions into a mono IR.
pub struct IrBuilder {
    cfg: IrConfig,
}

impl IrBuilder {
    /// Creates a CPU IR builder with fixed output dimensions.
    pub fn new(cfg: IrConfig) -> SoundResult<Self> {
        if cfg.sample_rate == 0 {
            return Err(SoundError::invalid_argument("IR sample_rate must be > 0"));
        }
        if cfg.ir_len_samples == 0 {
            return Err(SoundError::invalid_argument("IR length must be > 0"));
        }

        Ok(Self { cfg })
    }

    /// Bins contributions by arrival time and returns an immutable IR snapshot.
    pub fn build_from_contributions(
        &self,
        scene_version: SceneVersion,
        query_id: u32,
        contributions: &[ContributionRecord],
    ) -> IrSnapshot {
        let mut samples = vec![0.0_f32; self.cfg.ir_len_samples as usize];
        for contribution in contributions {
            let arrival_time_seconds = contribution.timing_gain[0];
            let gain = contribution.timing_gain[1];
            if !arrival_time_seconds.is_finite() || !gain.is_finite() || gain == 0.0 {
                continue;
            }

            let sample_idx = (arrival_time_seconds * self.cfg.sample_rate as f32).round() as isize;
            if sample_idx >= 0 {
                if let Some(sample) = samples.get_mut(sample_idx as usize) {
                    *sample += gain;
                }
            }
        }

        let energy = samples.iter().map(|sample| sample * sample).sum();
        IrSnapshot {
            sample_rate: self.cfg.sample_rate,
            samples,
            energy,
            scene_version,
            query_id,
        }
    }
}
