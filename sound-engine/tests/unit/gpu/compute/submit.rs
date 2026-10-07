use super::*;
use crate::gpu::backend::VkBackend;

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn audit_failed_recording_recycles_command_buffer_and_fence() {
    let gpu = VkBackend::new().unwrap();
    for _ in 0..32 {
        assert!(
            // The deliberately failing closure records no resource access.
            unsafe {
                gpu.compute().submit_compute(|_| {
                    Err(SoundError::invalid_state("intentional recording failure"))
                })
            }
            .is_err()
        );
        assert!(gpu.compute().in_flight.lock().unwrap().is_empty());
        assert_eq!(
            gpu.compute()
                .available_command_buffers
                .lock()
                .unwrap()
                .len(),
            COMMAND_BUFFER_POOL_SIZE as usize
        );
    }
    gpu.compute().submit_compute_and_wait(|_| Ok(())).unwrap();
}

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn submissions_wait_on_their_originating_context_with_overlapping_ids() {
    let first = VkBackend::new().unwrap();
    let second = VkBackend::new().unwrap();
    let first_buffer = first.memory().create_storage_buffer(16).unwrap();
    let second_buffer = second.memory().create_storage_buffer(16).unwrap();
    first
        .memory()
        .upload_typed(&first_buffer, &[1_u32, 2, 3, 4])
        .unwrap();
    second
        .memory()
        .upload_typed(&second_buffer, &[5_u32, 6, 7, 8])
        .unwrap();

    // Each backend retains its own buffer through both completion waits below.
    let first_submission = unsafe {
        first.compute().submit_compute(|command_buffer| {
            first
                .memory()
                .record_clear_buffer(command_buffer, &first_buffer)
        })
    }
    .unwrap();
    let second_submission = unsafe {
        second.compute().submit_compute(|command_buffer| {
            second
                .memory()
                .record_clear_buffer(command_buffer, &second_buffer)
        })
    };
    let overlapping_ids = second_submission
        .as_ref()
        .is_ok_and(|submission| submission.id == first_submission.id);

    let second_wait = second_submission.and_then(Submission::wait);
    let first_still_tracked = first
        .compute()
        .in_flight
        .lock()
        .unwrap()
        .contains_key(&first_submission.id);
    let second_finished = second.compute().in_flight.lock().unwrap().is_empty();
    let first_wait = first_submission.wait();

    // Defer assertions until both waits finish, so a panic cannot release an in-flight buffer.
    first_wait.unwrap();
    second_wait.unwrap();
    assert!(overlapping_ids);
    assert!(first_still_tracked);
    assert!(second_finished);
    assert!(first.compute().in_flight.lock().unwrap().is_empty());
    for (gpu, buffer) in [(&first, &first_buffer), (&second, &second_buffer)] {
        let mut output = [u32::MAX; 4];
        gpu.memory().download_typed(buffer, &mut output).unwrap();
        assert_eq!(output, [0; 4]);
    }
}
