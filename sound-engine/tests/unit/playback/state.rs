use super::*;

#[test]
fn construction_rejects_invalid_capacity_and_unreachable_prebuffer() {
    let error = PlaybackState::new(usize::MAX, 0).err().unwrap();
    assert_eq!(
        error.code(),
        crate::core::error::ErrorCode::ResourceAllocationFailed
    );
    assert_eq!(
        PlaybackState::new(2, 3).err().unwrap().code(),
        crate::core::error::ErrorCode::InvalidArgument
    );
    let state = PlaybackState::new(2, 0).unwrap();
    assert!(!state.is_playback_done());
    assert!(!state.is_producer_done());
    assert!(state.ready_to_play());
    assert_eq!(state.queue().queued_samples().unwrap(), 0);
}

#[test]
fn first_failure_survives_shutdown_and_preserves_its_cause() {
    use std::error::Error;
    let state = PlaybackState::new(2, 0).unwrap();
    state.report_failure(
        std::io::Error::new(std::io::ErrorKind::BrokenPipe, "device disconnected").into(),
    );
    state.report_failure(SoundError::external("later error"));
    for result in [
        state.check_error(),
        state.mark_producer_done(),
        state.request_stop(),
    ] {
        let error = result.unwrap_err();
        assert_eq!(error.code(), crate::core::error::ErrorCode::Io);
        assert!(error.to_string().contains("device disconnected"));
        assert_eq!(
            error
                .source()
                .unwrap()
                .downcast_ref::<std::io::Error>()
                .unwrap()
                .kind(),
            std::io::ErrorKind::BrokenPipe
        );
    }
    assert!(state.is_playback_done());
    assert!(state.is_stop_requested());
}

#[test]
fn counts_active_starvation_but_excludes_startup_idle_and_finished_audio() {
    let state = PlaybackState::new(1024, 0).unwrap();
    state.record_callback(0, 100);
    assert_eq!(state.starvation_counts(), (0, 0));
    assert_eq!(state.starvation_time_range_seconds(), None);
    state.record_callback(100, 100);
    state.record_callback(40, 100);
    assert_eq!(state.starvation_counts(), (1, 60));
    let (first, last) = state.starvation_time_range_seconds().unwrap();
    assert!(first >= 0.0 && first == last);
    state.set_expects_audio(false);
    state.record_callback(0, 100);
    assert_eq!(state.starvation_counts(), (1, 60));
    state.set_expects_audio(true);
    state.record_callback(0, 100);
    assert_eq!(state.starvation_counts(), (2, 160));
    let (retained_first, latest) = state.starvation_time_range_seconds().unwrap();
    assert_eq!(retained_first, first);
    assert!(latest >= last);
    state.mark_producer_done().unwrap();
    state.record_callback(0, 100);
    assert_eq!(state.starvation_counts(), (2, 160));
}

#[test]
fn explicit_stop_does_not_count_callback_silence_as_starvation() {
    let state = PlaybackState::new(1024, 0).unwrap();
    state.record_callback(100, 100);
    state.record_callback(40, 100);
    assert_eq!(state.starvation_counts(), (1, 60));
    state.request_stop().unwrap();
    // The device can deliver more callbacks before the stream is dropped.
    state.record_callback(0, 100);
    assert_eq!(state.starvation_counts(), (1, 60));
}

#[test]
fn prebuffer_releases_at_its_threshold_or_when_short_audio_finishes() {
    for finish_early in [false, true] {
        let state = PlaybackState::new(4, 4).unwrap();
        assert!(!state.ready_to_play());
        state.push_block(&[0.25, -0.25]).unwrap();
        assert!(!state.ready_to_play());
        if finish_early {
            state.mark_producer_done().unwrap();
        } else {
            state.push_block(&[0.5, -0.5]).unwrap();
        }
        assert!(state.ready_to_play());
    }
}
