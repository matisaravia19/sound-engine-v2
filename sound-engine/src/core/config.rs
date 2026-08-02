/// End-to-end runtime configuration for acoustic rendering.
#[derive(Debug, Clone, Copy)]
pub struct EngineConfig {
    /// Configuration for sound generation and IR construction.
    pub sound: SoundConfig,
    /// Configuration for acoustic simulation parameters.
    pub acoustics: AcousticsConfig,
    /// Configuration for acoustic IR caching and reuse.
    pub cache: IrCacheConfig,
    /// Configuration for analytic binaural rendering.
    pub binaural: BinauralConfig,
    /// Configuration for the auralization engine and output generation.
    pub auralization: AuralizationConfig,
}

/// Parameters for the angle-derived binaural estimator.
#[derive(Debug, Clone, Copy)]
pub struct BinauralConfig {
    /// Radius of the spherical head model in meters.
    pub head_radius_meters: f32,
    /// Maximum broadband interaural level difference in decibels.
    pub max_ild_db: f32,
}

impl Default for BinauralConfig {
    fn default() -> Self {
        Self {
            head_radius_meters: 0.0875,
            max_ild_db: 12.0,
        }
    }
}

/// Runtime configuration for sound generation and impulse response construction.
#[derive(Debug, Clone, Copy)]
pub struct SoundConfig {
    /// Channels used for output audio generation.
    pub output_channels: OutputChannels,
    /// Sample rate for sound generation and IR construction.
    pub sample_rate: u32,
    /// Number of samples in each generated impulse response.
    pub ir_num_samples: u32,
}

/// Output channel layout used by the auralization engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputChannels {
    /// Mono output.
    Mono,
    /// Two-channel stereo output.
    Stereo,
    /// Two-channel binaural output.
    Binaural,
}

impl OutputChannels {
    /// Returns the number of interleaved output channels.
    pub fn count(self) -> usize {
        match self {
            OutputChannels::Mono => 1,
            OutputChannels::Stereo | OutputChannels::Binaural => 2,
        }
    }
}

/// Runtime configuration for acoustic simulation and ray tracing.
#[derive(Debug, Clone, Copy)]
pub struct AcousticsConfig {
    /// Number of Monte Carlo paths launched from the listener for each query.
    pub listener_ray_count: u32,
    /// Maximum number of surface or diffraction interactions per path.
    pub max_bounces: u32,
    /// Maximum completed or continued path length in world meters.
    pub maximum_path_length: f32,
    /// Absolute path-throughput threshold below which tracing stops.
    pub minimum_throughput: f32,
    /// Stable seed used to scramble listener directions and continuation samples.
    pub sampling_seed: u32,
}

/// Runtime policy for acoustic impulse-response reuse.
#[derive(Debug, Clone, Copy)]
pub struct IrCacheConfig {
    /// Position bucket size in world meters.
    ///
    /// Reuse remains governed by `reuse_distance_meters` across bucket boundaries.
    pub position_cell_meters: f32,
    /// Maximum source/listener movement that may reuse an IR.
    pub reuse_distance_meters: f32,
    /// Dot-product tolerance applied to both listener orientation axes during reuse.
    pub listener_orientation_dot_threshold: f32,
    /// Maximum cached IR snapshots retained per engine instance.
    pub max_entries: usize,
    /// Maximum GPU memory used by cached acoustic IR buffers.
    pub max_gpu_cache_bytes: usize,
}

impl Default for IrCacheConfig {
    fn default() -> Self {
        Self {
            position_cell_meters: 0.25,
            reuse_distance_meters: 0.2,
            listener_orientation_dot_threshold: 0.98,
            max_entries: 32,
            max_gpu_cache_bytes: 512 * 1024 * 1024,
        }
    }
}

/// Runtime configuration for the auralization engine.
#[derive(Debug, Clone, Copy)]
pub struct AuralizationConfig {
    /// Number of audio frames processed per callback block.
    pub block_size: usize,
}
