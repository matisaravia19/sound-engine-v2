use super::*;
use crate::core::config::*;
use crate::playback::TestPlaybackControl;
use glam::Vec3;
use std::time::{Duration, Instant};

#[test]
fn panicking_worker_publishes_failure_without_stale_voices() {
    let status = Arc::new(RuntimeStatus::new(3));
    let worker_status = status.clone();
    let worker = thread::spawn(move || {
        let _guard = PanicStatusGuard(worker_status);
        panic!("simulated playback panic");
    });
    assert!(worker.join().is_err());
    assert_eq!(status.active_voices.load(Ordering::Acquire), 0);
    assert!(
        status
            .check_error()
            .unwrap_err()
            .to_string()
            .contains("panicked")
    );
}

#[test]
#[ignore = "requires Vulkan; uses simulated audio callbacks; run serially"]
fn audit_runtime_handoff_status_failure_recovery_and_drop() {
    let config = EngineConfig {
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
    };
    let mut core = Box::new(EngineCore::new(config).unwrap());
    let device = core.device_lifetime();
    let short = core
        .insert_sound(SoundAsset::from_mono_samples(48_000, vec![0.01; 256]).unwrap())
        .unwrap();
    let long = core
        .insert_sound(SoundAsset::from_mono_samples(48_000, vec![0.01; 16384]).unwrap())
        .unwrap();

    let failed = EngineRuntime::start_with_player(core, || {
        Err(SoundError::backend_unavailable("simulated startup failure"))
    })
    .err()
    .unwrap();
    assert!(
        failed
            .error
            .to_string()
            .contains("simulated startup failure")
    );
    let mut core = failed.core;
    assert_eq!(core.sample_rate(), 48_000);

    // Test publication at the acknowledgment boundary, before any render can
    // independently publish the same count and hide a stale-command snapshot.
    let status = RuntimeStatus::new(0);
    let (reply, response) = mpsc::sync_channel(1);
    assert!(!handle_command(
        EngineCommand::PlaySound(PlaySpatialSoundRequest::new(short, Vec3::X), reply),
        &mut core,
        &status
    ));
    let voice = response.recv().unwrap().unwrap();
    assert_eq!(status.active_voices.load(Ordering::Acquire), 1);
    let (reply, response) = mpsc::sync_channel(1);
    assert!(!handle_command(
        EngineCommand::StopVoice(voice, reply),
        &mut core,
        &status
    ));
    response.recv().unwrap().unwrap();
    assert_eq!(status.active_voices.load(Ordering::Acquire), 0);

    let (runtime, control) = start_headless(core);
    assert_eq!(runtime.active_voice_count(), 0);
    let voice = runtime
        .play_sound(PlaySpatialSoundRequest::new(long, Vec3::X))
        .unwrap();
    assert_eq!(runtime.active_voice_count(), 1);
    wait_until(|| control.queued_samples() == 128);
    control.fail(SoundError::backend_unavailable(
        "simulated device disconnect",
    ));
    wait_until(|| !runtime.is_running());
    let error = runtime
        .create_sound_source(PointSource::new(Vec3::X))
        .unwrap_err();
    assert!(error.to_string().contains("simulated device disconnect"));
    let (core, result) = runtime.stop();
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("simulated device disconnect")
    );
    let mut core = core.unwrap();
    assert_eq!(core.active_voice_count(), 1);
    core.stop_voice(voice);

    let (runtime, _control) = start_headless(core);
    let mailbox = runtime.pending_listener.clone();
    assert!(
        thread::spawn(move || {
            let _guard = mailbox.lock().unwrap();
            panic!("simulated listener writer panic");
        })
        .join()
        .is_err()
    );
    let voice = runtime
        .play_sound(PlaySpatialSoundRequest::new(long, Vec3::X))
        .unwrap();
    wait_until(|| !runtime.is_running());
    let error = runtime
        .create_sound_source(PointSource::new(Vec3::X))
        .unwrap_err();
    assert_eq!(error.code(), crate::core::error::ErrorCode::PoisonedLock);
    let (core, result) = runtime.stop();
    assert_eq!(
        result.unwrap_err().code(),
        crate::core::error::ErrorCode::PoisonedLock
    );
    let mut core = core.unwrap();
    core.stop_voice(voice);

    // Restart with an existing voice, then drive real queue callbacks until its
    // dry samples and convolution tail finish. Status reads must not block.
    core.play_sound(PlaySpatialSoundRequest::new(short, Vec3::X))
        .unwrap();
    let (runtime, control) = start_headless(core);
    assert_eq!(runtime.active_voice_count(), 1);
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut output = [0.0; 128];
    while runtime.active_voice_count() != 0 {
        assert!(Instant::now() < deadline, "voice did not finish");
        control.drain(&mut output);
        assert!(output.iter().all(|sample| sample.is_finite()));
        thread::sleep(Duration::from_millis(1));
    }
    let (core, result) = runtime.stop();
    result.unwrap();
    let mut core = core.unwrap();
    assert_eq!(core.active_voice_count(), 0);
    core.play_sound(PlaySpatialSoundRequest::new(long, Vec3::X))
        .unwrap();
    let (runtime, control) = start_headless(core);
    wait_until(|| control.queued_samples() == 128);
    drop(runtime);
    assert!(
        device.upgrade().is_none(),
        "runtime drop retained the device owner"
    );
}

fn start_headless(core: Box<EngineCore>) -> (EngineRuntime, TestPlaybackControl) {
    let (sender, receiver) = mpsc::sync_channel(1);
    let runtime = EngineRuntime::start_with_player(core, move || {
        let (player, control) = SoundPlayer::test_player(128);
        sender.send(control).unwrap();
        Ok(player)
    })
    .unwrap_or_else(|error| panic!("{}", error.error));
    (runtime, receiver.recv().unwrap())
}

fn wait_until(mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !condition() {
        assert!(Instant::now() < deadline, "worker did not make progress");
        thread::sleep(Duration::from_millis(1));
    }
}
