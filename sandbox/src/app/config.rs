//! Default audio and acoustic budgets for interactive playback.
use sound_engine::core::config::{
    AcousticsConfig, AuralizationConfig, EngineConfig, IrCacheConfig, OutputChannels, SoundConfig,
};

use sound_engine::core::error::{SoundError, SoundResult};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Duration;

const BLOCK_SIZE: usize = 1024;
const IR_SECONDS: u32 = 2;

/// Builds the sandbox audio and ray budgets at the output device sample rate.
fn engine_config(sample_rate: u32) -> EngineConfig {
    EngineConfig {
        sound: SoundConfig {
            output_channels: OutputChannels::Stereo,
            sample_rate,
            ir_num_samples: sample_rate * IR_SECONDS,
        },
        acoustics: AcousticsConfig {
            listener_ray_count: 1_000_000,
            max_bounces: 3,
            maximum_path_length: 343.0,
            minimum_throughput: 1.0e-6,
            sampling_seed: 0,
        },
        cache: IrCacheConfig::default(),
        binaural: Default::default(),
        auralization: AuralizationConfig {
            block_size: BLOCK_SIZE,
        },
    }
}

/// Validated startup settings for the playground.
pub(super) struct Options {
    /// Shows usage without creating device resources.
    pub help: bool,
    /// Prints the available scene identifiers.
    pub list_scenes: bool,
    /// Loads and validates the scene catalog without initializing audio.
    pub validate_scenes: bool,
    /// Root of optional measured benchmark fixtures.
    pub benchmark_root: PathBuf,
    /// Catalog identifier selected at startup.
    pub scene: Option<String>,
    /// WAV asset used by the playback controls.
    pub sound: PathBuf,
    /// Width of doorway diffraction strips, in meters.
    pub diffraction_radius: f32,
    /// Linear dry-sound gain, separate from acoustic pressure.
    pub volume: f32,
    /// Optional duration of automated camera movement.
    pub camera_stress: Option<Duration>,
    /// Optional destination for measured frame times.
    pub frame_report: Option<PathBuf>,
    rays: Option<u32>,
    bounces: Option<u32>,
}

impl Default for Options {
    fn default() -> Self {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        Self {
            help: false,
            list_scenes: false,
            validate_scenes: false,
            benchmark_root: root.join("../output"),
            scene: None,
            sound: root.join("../../circus.wav"),
            diffraction_radius: 1.25,
            volume: 0.6,
            camera_stress: None,
            frame_report: None,
            rays: None,
            bounces: None,
        }
    }
}

impl Options {
    /// Parses named options and rejects invalid values before device allocation.
    pub(super) fn parse(arguments: impl IntoIterator<Item = String>) -> SoundResult<Self> {
        let mut options = Self::default();
        for argument in arguments {
            match argument.as_str() {
                "--help" | "-h" => options.help = true,
                "--list-scenes" => options.list_scenes = true,
                "--validate-scenes" => options.validate_scenes = true,
                _ => {
                    let (name, value) = argument.split_once('=').ok_or_else(|| {
                        SoundError::invalid_argument(format!("unknown option: {argument}"))
                    })?;
                    if value.is_empty() {
                        return Err(SoundError::invalid_argument(format!("empty {name}")));
                    }
                    match name {
                        "--benchmark-root" => options.benchmark_root = value.into(),
                        "--scene" => options.scene = Some(value.into()),
                        "--sound" => options.sound = value.into(),
                        "--diffraction-radius" => options.diffraction_radius = parse(name, value)?,
                        "--volume" => options.volume = parse(name, value)?,
                        "--rays" => options.rays = Some(parse(name, value)?),
                        "--bounces" => options.bounces = Some(parse(name, value)?),
                        "--frame-report" => options.frame_report = Some(value.into()),
                        "--camera-stress-seconds" => {
                            let seconds: f32 = parse(name, value)?;
                            let duration = Duration::try_from_secs_f32(seconds).map_err(|_| {
                                SoundError::invalid_argument(format!("invalid {name}"))
                            })?;
                            if duration.is_zero() {
                                return Err(SoundError::invalid_argument(format!(
                                    "{name} must be positive"
                                )));
                            }
                            options.camera_stress = Some(duration);
                        }
                        _ => {
                            return Err(SoundError::invalid_argument(format!(
                                "unknown option: {name}"
                            )));
                        }
                    }
                }
            }
        }

        if !options.diffraction_radius.is_finite() || options.diffraction_radius <= 0.0 {
            return Err(SoundError::invalid_argument(
                "diffraction radius must be positive and finite",
            ));
        }
        if !options.volume.is_finite() || options.volume < 0.0 {
            return Err(SoundError::invalid_argument(
                "volume must be finite and non-negative",
            ));
        }
        if options.rays == Some(0) {
            return Err(SoundError::invalid_argument("ray count must be positive"));
        }
        Ok(options)
    }

    /// Applies explicit ray budgets to the default engine configuration.
    pub(super) fn engine_config(&self, sample_rate: u32) -> EngineConfig {
        let mut config = engine_config(sample_rate);
        if let Some(rays) = self.rays {
            config.acoustics.listener_ray_count = rays;
        }
        if let Some(bounces) = self.bounces {
            config.acoustics.max_bounces = bounces;
        }
        config
    }

    /// Prints supported options without requiring device resources.
    pub(super) fn print_help() {
        println!(concat!(
            "Spatial audio playground\n\n",
            "  --list-scenes                 List the scene catalog\n",
            "  --validate-scenes             Validate scene fixtures\n",
            "  --benchmark-root=PATH         Root of optional benchmark fixtures\n",
            "  --scene=ID                    Initial scene (default: playground)\n",
            "  --sound=PATH                  WAV asset to play\n",
            "  --volume=GAIN                 Dry-sound gain (default: 0.6)\n",
            "  --diffraction-radius=METERS   Edge-strip width (default: 1.25)\n",
            "  --rays=COUNT                  Listener rays (default: 1000000)\n",
            "  --bounces=COUNT               Path interaction limit (default: 3)\n",
            "  --camera-stress-seconds=N     Run automated camera motion\n",
            "  --frame-report=PATH           Export measured frame times\n",
            "  --help                        Show this help"
        ));
    }
}

fn parse<T: FromStr>(name: &str, value: &str) -> SoundResult<T> {
    value
        .parse()
        .map_err(|_| SoundError::invalid_argument(format!("invalid {name}: {value}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_options_fail_before_device_initialization() {
        for argument in [
            "--unknown",
            "--scene=",
            "--rays=0",
            "--volume=NaN",
            "--diffraction-radius=-1",
            "--camera-stress-seconds=no",
            "--camera-stress-seconds=NaN",
            "--camera-stress-seconds=0",
        ] {
            assert!(Options::parse([argument.to_owned()]).is_err(), "{argument}");
        }
    }

    #[test]
    fn options_control_scene_audio_and_motion() {
        let options = Options::parse(
            [
                "--scene=wedge",
                "--rays=64",
                "--bounces=0",
                "--sound=signal.wav",
                "--volume=0.25",
                "--camera-stress-seconds=2.5",
            ]
            .map(str::to_owned),
        )
        .unwrap();
        let config = options.engine_config(48_000);
        assert_eq!(options.scene.as_deref(), Some("wedge"));
        assert_eq!(options.sound, PathBuf::from("signal.wav"));
        assert_eq!(options.volume, 0.25);
        assert_eq!(options.camera_stress, Some(Duration::from_millis(2500)));
        assert_eq!(config.acoustics.listener_ray_count, 64);
        assert_eq!(config.acoustics.max_bounces, 0);
    }
}
