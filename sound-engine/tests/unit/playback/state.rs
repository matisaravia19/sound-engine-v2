use super::*;

#[test]
fn allocation_failure_leaves_the_state_unprepared_and_retryable() {
    let state = PlaybackState::new();
    let error = state.prepare_queue(usize::MAX).err().unwrap();
    assert_eq!(
        error.code(),
        crate::core::error::ErrorCode::ResourceAllocationFailed
    );
    assert!(state.is_playback_done());
    assert!(state.queue.get().is_none());
    assert!(state.prepare_queue(2).is_ok());
}

#[test]
fn first_failure_survives_shutdown_and_preserves_its_cause() {
    use std::error::Error;
    let state = PlaybackState::new();
    state.prepare_queue(2).unwrap();
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
    assert!(state.prepare_queue(2).is_err());
}

#[test]
fn queue_cannot_be_replaced_or_cancellation_reset() {
    let state = PlaybackState::new();
    let queue = state.prepare_queue(2).unwrap();
    queue
        .push_block(&[0.5], state.stop_requested(), state.producer_done())
        .unwrap();
    assert!(state.prepare_queue(4).is_err());
    state.request_stop().unwrap();
    assert!(state.check_error().is_ok());
    assert!(state.prepare_queue(4).is_err());
    assert!(state.is_stop_requested());
    assert_eq!(queue.queued_samples().unwrap(), 1);
}

#[test]
fn counts_active_starvation_but_excludes_startup_idle_and_finished_audio() {
    let state = PlaybackState::new();
    state.prepare_queue(1024).unwrap();
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
    let state = PlaybackState::new();
    state.prepare_queue(1024).unwrap();
    state.record_callback(100, 100);
    state.record_callback(40, 100);
    assert_eq!(state.starvation_counts(), (1, 60));
    state.request_stop().unwrap();
    // The device can deliver more callbacks before the stream is dropped.
    state.record_callback(0, 100);
    assert_eq!(state.starvation_counts(), (1, 60));
}

#[test]
fn prebuffer_is_explicit_and_short_finished_sounds_are_released() {
    let state = PlaybackState::new();
    state.prepare_queue(1024).unwrap();
    state.begin_prebuffering();
    assert!(!state.ready_to_play());
    state.release_prebuffer();
    assert!(state.ready_to_play());
    state.begin_prebuffering();
    state.mark_producer_done().unwrap();
    assert!(state.ready_to_play());
}
