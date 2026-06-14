use crate::core::config::OutputChannels;
use crate::core::error::{SoundError, SoundResult};
use crate::scene::SceneVersion;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// Maximum number of path contributions downloaded for one acoustic query.
pub const MAX_CONTRIBUTIONS_PER_QUERY: usize = 4096;

/// One acoustic contribution produced by ray tracing.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ContributionRecord {
    /// Arrival time in seconds from the source to the listener.
    pub arrival_time_seconds: f32,
    /// Linear gain applied to the impulse response at the arrival time.
    pub linear_gain: f32,
    /// Padding that aligns the following shader-written `vec3` field.
    pub _timing_padding: [f32; 2],
    /// Incoming ray direction in world space. Expected to already be normalized.
    pub incoming_direction: Vec3,
    /// Reflection path order reported by the ray payload.
    pub path_order: f32,
}

unsafe impl bytemuck::Zeroable for ContributionRecord {}
unsafe impl bytemuck::Pod for ContributionRecord {}

/// Stereo impulse-response sample stored as left and right channel values.
#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct IrSample {
    /// Left output channel sample value.
    pub left: f32,
    /// Right output channel sample value.
    pub right: f32,
}

impl IrSample {
    /// Creates a new stereo IR sample with the given left and right values.
    pub fn new(left: f32, right: f32) -> Self {
        Self { left, right }
    }

    /// Creates a zero-valued stereo IR sample.
    pub fn zero() -> Self {
        Self { left: 0.0, right: 0.0 }
    }
}

/// CPU snapshot ready for convolution after IR construction.
#[derive(Debug, Clone)]
pub struct IrSnapshot {
    /// Sample rate used to generate `samples`.
    pub sample_rate: u32,
    /// Impulse-response samples.
    pub samples: Vec<IrSample>,
    /// Sum of squared sample values for diagnostics and sanity checks.
    pub energy: f32,
    /// Scene version represented by this IR.
    pub scene_version: SceneVersion,
    /// Query identifier copied from the acoustic request.
    pub query_id: u32,
}

/// CPU helper that bins downloaded path contributions into a stereo IR.
pub struct IrBuilder {
    /// Sample rate used for converting arrival time to sample index.
    pub sample_rate: u32,
    /// Number of samples in the generated IR.
    pub num_samples: u32,
    /// Output layout used to decide whether contributions are duplicated or panned.
    pub output_channels: OutputChannels,
}

impl IrBuilder {
    /// Creates a CPU IR builder with fixed output dimensions.
    pub fn new(sample_rate: u32, num_samples: u32, output_channels: OutputChannels) -> SoundResult<Self> {
        if sample_rate == 0 {
            return Err(SoundError::invalid_argument("IR sample_rate must be > 0"));
        }

        if num_samples == 0 {
            return Err(SoundError::invalid_argument("IR length must be > 0"));
        }

        Ok(Self {
            sample_rate,
            num_samples,
            output_channels,
        })
    }

    /// Bins contributions by arrival time and returns an immutable IR snapshot.
    ///
    /// # Arguements
    /// - `scene_version`: Version of the scene used to generate the contributions.
    /// - `query_id`: Identifier for the acoustic query that produced the contributions.
    /// - `contributions`: List of path contributions downloaded from the GPU.
    /// - `listener_right`: Normalized right direction of the listener used for stereo panning.
    pub fn build_from_contributions(
        &self,
        scene_version: SceneVersion,
        query_id: u32,
        contributions: &[ContributionRecord],
        listener_right: Vec3,
    ) -> IrSnapshot {
        let mut samples = vec![IrSample::zero(); self.num_samples as usize];
        let mut energy = 0.0f32;
        for contribution in contributions {
            let arrival_time_seconds = contribution.arrival_time_seconds;
            let gain = contribution.linear_gain;
            if !arrival_time_seconds.is_finite() || !gain.is_finite() || gain == 0.0 {
                continue;
            }

            let sample_idx = (arrival_time_seconds * self.sample_rate as f32).round() as isize;
            if sample_idx < 0 || sample_idx >= self.num_samples as isize {
                continue;
            }

            if let Some(sample) = samples.get_mut(sample_idx as usize) {
                let [left_gain, right_gain] = self.channel_gains(contribution.incoming_direction, listener_right);
                sample.left += gain * left_gain;
                sample.right += gain * right_gain;

                energy += gain * gain;
            }
        }

        IrSnapshot {
            sample_rate: self.sample_rate,
            samples,
            energy,
            scene_version,
            query_id,
        }
    }

    fn channel_gains(&self, incoming_direction: Vec3, listener_right: Vec3) -> [f32; 2] {
        match self.output_channels {
            OutputChannels::Mono => [1.0, 1.0],
            OutputChannels::Stereo | OutputChannels::Binaural => {
                let pan = incoming_direction.dot(listener_right).clamp(-1.0, 1.0);
                [(0.5 * (1.0 - pan)).sqrt(), (0.5 * (1.0 + pan)).sqrt()]
            }
        }
    }
}
