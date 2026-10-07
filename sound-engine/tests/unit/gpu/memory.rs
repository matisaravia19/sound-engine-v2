use super::*;
use crate::core::error::ErrorCode;
use crate::gpu::backend::VkBackend;

#[test]
fn buffer_usage_rejects_disabled_extensions_and_unknown_bits() {
    use vk::BufferUsageFlags as Usage;
    assert!(validate_buffer_usage(Usage::STORAGE_BUFFER | Usage::TRANSFER_DST).is_ok());
    assert!(validate_buffer_usage(Usage::ACCELERATION_STRUCTURE_STORAGE_KHR).is_ok());
    for usage in [
        Usage::empty(),
        Usage::VIDEO_DECODE_SRC_KHR,
        Usage::TRANSFORM_FEEDBACK_BUFFER_EXT,
        Usage::from_raw(u32::MAX),
    ] {
        assert!(validate_buffer_usage(usage).is_err());
    }
}

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn audit_buffer_retains_device_after_backend_drop() {
    let gpu = VkBackend::new().unwrap();
    let device = Arc::downgrade(&gpu.device_context());
    let buffer = gpu.memory().create_storage_buffer(16).unwrap();
    gpu.memory()
        .upload_typed(&buffer, &[1_u32, 2, 3, 4])
        .unwrap();
    drop(gpu);
    assert!(device.upgrade().is_some());
    drop(buffer);
    assert!(device.upgrade().is_none());
}

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn audit_empty_transfers_and_invalid_allocations() {
    let gpu = VkBackend::new().unwrap();
    assert!(gpu.memory().create_storage_buffer(0).is_err());
    assert!(
        gpu.memory()
            .create_device_address_buffer(16, vk::BufferUsageFlags::VIDEO_DECODE_SRC_KHR)
            .is_err()
    );
    let buffer = gpu.memory().create_storage_buffer(5).unwrap();
    assert!(gpu.memory().clear_buffer(&buffer).is_err());
    gpu.memory().upload_bytes(&buffer, &[]).unwrap();
    gpu.memory().download_bytes(&buffer, &mut []).unwrap();
    assert!(gpu.memory().upload_bytes(&buffer, &[0; 6]).is_err());
    let aligned = gpu.memory().create_storage_buffer(4).unwrap();
    gpu.compute()
        .submit_compute_and_wait(|command_buffer| {
            // The local allocation is retained until this synchronous submission finishes.
            unsafe {
                gpu.memory().record_clear_buffer(command_buffer, &aligned)?;
                gpu.compute().record_buffer_barrier(
                    command_buffer,
                    &BufferBarrierSpec::transfer_write_to_device_read(aligned.buffer),
                );
            }
            Ok(())
        })
        .unwrap();
    let mut output = [1_u32];
    gpu.memory().download_typed(&aligned, &mut output).unwrap();
    assert_eq!(output, [0]);
}

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn readback_fills_only_the_output_slice_and_rejects_oversized_reads() {
    let gpu = VkBackend::new().unwrap();
    let input = [0x1122_3344_u32, 0x5566_7788, 0x99aa_bbcc, 0xddee_ff00];
    let buffer = gpu
        .memory()
        .create_storage_buffer(size_of_val(&input) as u64)
        .unwrap();
    gpu.memory().upload_typed(&buffer, &input).unwrap();

    let mut output = [u32::MAX; 4];
    gpu.memory()
        .download_typed(&buffer, &mut output[1..3])
        .unwrap();
    assert_eq!(output, [u32::MAX, input[0], input[1], u32::MAX]);

    let mut oversized_words = [u32::MAX; 5];
    let error = gpu
        .memory()
        .download_typed(&buffer, &mut oversized_words)
        .unwrap_err();
    assert_eq!(error.code(), ErrorCode::InvalidArgument);
    assert_eq!(oversized_words, [u32::MAX; 5]);

    let mut oversized_bytes = [u8::MAX; 17];
    let error = gpu
        .memory()
        .download_bytes(&buffer, &mut oversized_bytes)
        .unwrap_err();
    assert_eq!(error.code(), ErrorCode::InvalidArgument);
    assert_eq!(oversized_bytes, [u8::MAX; 17]);

    let mut bytes = [u8::MAX; 20];
    gpu.memory()
        .download_bytes(&buffer, &mut bytes[2..18])
        .unwrap();
    assert_eq!(&bytes[2..18], bytemuck::cast_slice::<u32, u8>(&input));
    assert_eq!(&bytes[..2], &[u8::MAX; 2]);
    assert_eq!(&bytes[18..], &[u8::MAX; 2]);
}

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn staging_growth_and_explicit_release_preserve_transfers() {
    let gpu = VkBackend::new().unwrap();
    // Cross both the initial capacity and its first doubling. Round-trip
    // every word so a freed/reused allocation cannot pass by checking a prefix.
    for bytes in [
        STAGING_BUFFER_INITIAL_SIZE + 64,
        2 * STAGING_BUFFER_INITIAL_SIZE + 64,
    ] {
        let input: Vec<u32> = (0..bytes as usize / 4)
            .map(|i| i as u32 ^ 0xa55a_5aa5)
            .collect();
        let buffer = gpu.memory().create_storage_buffer(bytes).unwrap();
        gpu.memory().upload_typed(&buffer, &input).unwrap();
        let mut output = vec![0_u32; input.len()];
        gpu.memory().download_typed(&buffer, &mut output).unwrap();
        assert_eq!(input, output);
        drop(buffer);
    }
}

impl GpuAllocator {
    /// Counts live VMA allocations and requested bytes, excluding reserved pool capacity.
    pub(crate) fn live_allocation_stats(&self) -> SoundResult<(u32, u64)> {
        let stats = self.allocator.calculate_statistics()?.total.statistics;
        Ok((stats.allocationCount, stats.allocationBytes))
    }
}
