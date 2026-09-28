//! Small device-creation diagnostic; run the late-device mode with an external timeout.
//!
//! With this machine's driver, creating a second device on a worker thread while
//! a traced acoustic engine remains alive can stall inside vkCreateDevice. The default creates both
//! devices first; --late-device reproduces the ordering and --after-drop tests
//! whether releasing the acoustic engine changes the result. Add --worker-thread
//! to run outside the main thread or --pipeline-only to omit the initial trace.

use glam::Vec3;
use sound_engine::acoustics::AcousticsEngine;
use sound_engine::core::config::*;
use sound_engine::core::error::{SoundError, SoundResult};
use sound_engine::engine::{ListenerPose, PointSource};
use sound_engine::gpu::backend::VkBackend;

fn main() -> SoundResult<()> {
    let mode = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "--early-device".into());
    let trace = !std::env::args().any(|argument| argument == "--pipeline-only");
    if std::env::args().any(|argument| argument == "--worker-thread") {
        return std::thread::spawn(move || probe(&mode, trace))
            .join()
            .map_err(|_| SoundError::external("Device diagnostic worker panicked"))?;
    }
    probe(&mode, trace)
}

fn probe(mode: &str, trace: bool) -> SoundResult<()> {
    if !matches!(mode, "--early-device" | "--late-device" | "--after-drop") {
        return Err(SoundError::invalid_argument(
            "Expected --early-device, --late-device or --after-drop",
        ));
    }
    eprintln!("Creating first backend");
    let gpu = VkBackend::new()?;
    let mut second = if mode == "--early-device" {
        eprintln!("Creating second backend before tracing");
        Some(VkBackend::new()?)
    } else {
        None
    };
    let mut engine = Some(AcousticsEngine::new(
        &gpu,
        EngineConfig {
            sound: SoundConfig {
                sample_rate: 48_000,
                output_channels: OutputChannels::Stereo,
                ir_num_samples: 1024,
            },
            acoustics: AcousticsConfig {
                listener_ray_count: 1,
                max_bounces: 0,
                maximum_path_length: 100.0,
                minimum_throughput: 0.0,
                sampling_seed: 0,
            },
            cache: Default::default(),
            binaural: Default::default(),
            auralization: AuralizationConfig {
                block_size: 64,
            },
        },
    )?);
    let source = PointSource::new(Vec3::X);
    let listener = ListenerPose::default();
    if trace {
        eprintln!("Tracing one empty-scene query");
        engine
            .as_mut()
            .unwrap()
            .build_impulse_response_gpu(&gpu, source, listener)?;
    } else {
        eprintln!("Acoustic pipeline ready; initial trace omitted");
    }
    if mode == "--after-drop" {
        eprintln!("Dropping acoustic engine before second device");
        drop(engine.take());
    }
    if second.is_none() {
        eprintln!("Creating second backend after acoustic setup");
        second = Some(VkBackend::new()?);
    }
    eprintln!("Second backend ready");
    if let Some(engine) = &mut engine {
        if engine
            .build_impulse_response_gpu(second.as_ref().unwrap(), source, listener)
            .is_ok()
        {
            return Err(SoundError::invalid_state(
                "Foreign backend unexpectedly accepted",
            ));
        }
        engine.build_impulse_response_gpu(&gpu, source, listener)?;
    }
    drop(engine);
    drop(second);
    drop(gpu);
    eprintln!("Cleanup complete");
    Ok(())
}
