use super::*;

#[test]
fn finished_player_rejects_late_blocks_without_losing_its_queued_tail() {
    for prefill_blocks in [0, 4] {
        let (player, control) = SoundPlayer::test_player_with_prefill(2, prefill_blocks);
        assert!(player.push_block(&[0.25, -0.25]).unwrap());
        player.finish().unwrap();
        player.finish().unwrap();
        let mut output = [1.0; 4];
        control.drain(&mut output);
        assert_eq!(output, [0.25, -0.25, 0.0, 0.0]);
        assert!(player.state.is_playback_done());
        player.wait_until_finished().unwrap();
        assert_eq!(
            player.push_block(&[0.5, -0.5]).unwrap_err().code(),
            ErrorCode::InvalidState
        );
        assert_eq!(control.queued_samples(), 0);
        player.stop().unwrap();
        assert!(!player.push_block(&[0.0; 2]).unwrap());
        control.fail(SoundError::external("retained device failure"));
        assert!(
            player
                .push_block(&[0.0; 2])
                .unwrap_err()
                .to_string()
                .contains("retained device failure")
        );
    }
}

#[test]
fn finish_releases_a_blocked_producer_without_draining() {
    let state = Arc::new(PlaybackState::new(1, 0).unwrap());
    state.push_block(&[0.25]).unwrap();
    let producer_state = state.clone();
    let (started_tx, started_rx) = std::sync::mpsc::channel();
    let (result_tx, result_rx) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        started_tx.send(()).unwrap();
        let result = producer_state.push_block(&[0.5]);
        result_tx.send(result).unwrap();
    });
    started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    state.mark_producer_done().unwrap();
    let finished = result_rx.recv_timeout(Duration::from_secs(1));
    // Always release and join the owned producer, even against the old bug.
    state.request_stop().unwrap();
    worker.join().unwrap();
    assert_eq!(
        finished.unwrap().unwrap_err().code(),
        ErrorCode::InvalidState
    );
    assert_eq!(state.queue().queued_samples().unwrap(), 1);
}

#[test]
fn device_failure_wakes_a_full_queue_and_remains_visible_to_the_runtime() {
    let state = Arc::new(PlaybackState::new(1, 0).unwrap());
    state.push_block(&[1.0]).unwrap();
    let handle = PlaybackStopHandle {
        state: state.clone(),
    };
    let worker_state = state.clone();
    let (sent, received) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        sent.send(worker_state.push_block(&[2.0])).unwrap();
    });
    state.report_failure(SoundError::external("device disconnected"));
    let result = received.recv_timeout(Duration::from_secs(1)).unwrap();
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("device disconnected")
    );
    assert!(handle.check_error().is_err());
    assert!(handle.request_stop().is_err());
    worker.join().unwrap();
}

#[test]
fn stop_handle_releases_a_producer_without_an_audio_callback() {
    let state = Arc::new(PlaybackState::new(1, 0).unwrap());
    state.push_block(&[1.0]).unwrap();
    let handle = PlaybackStopHandle {
        state: state.clone(),
    };
    let (sent, received) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        sent.send(state.push_block(&[2.0]).unwrap()).unwrap();
    });
    handle.request_stop().unwrap();
    assert!(!received.recv_timeout(Duration::from_secs(1)).unwrap());
    worker.join().unwrap();
}

#[test]
fn derives_queue_blocks_from_seconds() {
    let config = PlaybackConfig {
        prefill_blocks: 4,
        max_queue_seconds: 0.5,
    };

    assert_eq!(max_queued_blocks(config, 48_000, 1024), 24);
}

#[test]
fn queue_blocks_respect_prefill_floor() {
    let config = PlaybackConfig {
        prefill_blocks: 8,
        max_queue_seconds: 0.01,
    };

    assert_eq!(max_queued_blocks(config, 48_000, 1024), 8);
}

#[test]
fn rejects_invalid_queue_seconds() {
    let mut config = PlaybackConfig {
        max_queue_seconds: f32::NAN,
        ..Default::default()
    };
    assert!(validate_config(config).is_err());

    config.max_queue_seconds = 0.0;
    assert!(validate_config(config).is_err());
}
