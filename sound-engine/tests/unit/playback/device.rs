use super::*;

#[test]
fn malformed_callback_reports_failure_and_silences_output() {
    let state = PlaybackState::new();
    let queue = state.prepare_queue(4).unwrap();
    let mut output = [1.0_f32; 3];
    write_output(&mut output, 2, 2, &queue, &state);
    assert_eq!(output, [0.0; 3]);
    assert_eq!(
        state.check_error().unwrap_err().code(),
        ErrorCode::InvalidArgument
    );
    assert!(state.is_stop_requested());
}

#[test]
fn write_output_silences_non_finite_samples() {
    let state = PlaybackState::new();
    let queue = state.prepare_queue(4).unwrap();
    queue
        .push_block(
            &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.25],
            state.stop_requested(),
            state.producer_done(),
        )
        .unwrap();
    let mut output = [1.0_f32; 4];
    write_output(&mut output, 2, 2, &queue, &state);
    assert_eq!(output, [0.0, 0.0, 0.0, 0.25]);
}

#[test]
fn write_output_maps_engine_channels_and_zeros_extra_device_channels() -> SoundResult<()> {
    let queue = SampleQueue::new(4)?;
    let state = PlaybackState::new();
    state.prepare_queue(4)?;
    assert!(queue.push_block(
        &[2.0, -2.0, 0.25, -0.25],
        state.stop_requested(),
        state.producer_done()
    )?);
    let mut output = [1.0f32; 6];

    write_output(&mut output, 3, 2, &queue, &state);

    assert_eq!(output, [1.0, -1.0, 0.0, 0.25, -0.25, 0.0]);
    assert!(!state.is_playback_done());
    Ok(())
}

#[test]
fn write_output_marks_done_after_final_underflow() -> SoundResult<()> {
    let queue = SampleQueue::new(2)?;
    let state = PlaybackState::new();
    state.prepare_queue(2)?;
    assert!(queue.push_block(&[0.5, -0.5], state.stop_requested(), state.producer_done())?);
    state.mark_producer_done()?;
    let mut output = [0.0f32; 4];

    write_output(&mut output, 2, 2, &queue, &state);

    assert_eq!(output, [0.5, -0.5, 0.0, 0.0]);
    assert!(state.is_playback_done());
    Ok(())
}
