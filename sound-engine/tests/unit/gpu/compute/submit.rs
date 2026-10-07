use super::*;

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn audit_failed_recording_recycles_command_buffer_and_fence() {
    let gpu = crate::gpu::backend::VkBackend::new().unwrap();
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
