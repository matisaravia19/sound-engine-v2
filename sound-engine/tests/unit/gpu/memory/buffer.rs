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
fn mapped_buffer_retains_its_device_and_host_access_after_backend_drop() {
    let gpu = VkBackend::new().unwrap();
    let device = Arc::downgrade(&gpu.device_context());
    let buffer = gpu
        .memory()
        .create_host_device_address_buffer(16, vk::BufferUsageFlags::STORAGE_BUFFER)
        .unwrap();
    let address = buffer.device_address();
    assert_ne!(address, 0);

    drop(gpu);
    assert!(device.upgrade().is_some());
    assert_eq!(buffer.device_address(), address);
    let input = [1_u32, 2, 3, 4];
    buffer
        .write_mapped_bytes(bytemuck::cast_slice(&input))
        .unwrap();
    let mut output = [0_u32; 4];
    buffer
        .read_mapped_bytes(bytemuck::cast_slice_mut(&mut output))
        .unwrap();
    assert_eq!(output, input);
    drop(buffer);
    assert!(device.upgrade().is_none());
}

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn mapped_access_checks_bounds_and_mapping_state_before_touching_memory() {
    let gpu = VkBackend::new().unwrap();
    let unmapped = gpu.memory().create_storage_buffer(16).unwrap();
    assert_eq!(
        unmapped.write_mapped_bytes(&[0; 16]).unwrap_err().code(),
        ErrorCode::InvalidState
    );
    let mut output = [u8::MAX; 16];
    assert_eq!(
        unmapped.read_mapped_bytes(&mut output).unwrap_err().code(),
        ErrorCode::InvalidState
    );
    assert_eq!(output, [u8::MAX; 16]);

    let mapped = gpu
        .memory()
        .create_host_device_address_buffer(16, vk::BufferUsageFlags::STORAGE_BUFFER)
        .unwrap();
    let input = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    mapped.write_mapped_bytes(&input).unwrap();
    mapped.write_mapped_bytes(&[16, 17]).unwrap();
    mapped.write_mapped_bytes(&[]).unwrap();
    assert_eq!(
        mapped.write_mapped_bytes(&[0; 17]).unwrap_err().code(),
        ErrorCode::InvalidArgument
    );

    let mut oversized = [u8::MAX; 17];
    assert_eq!(
        mapped.read_mapped_bytes(&mut oversized).unwrap_err().code(),
        ErrorCode::InvalidArgument
    );
    assert_eq!(oversized, [u8::MAX; 17]);
    mapped.read_mapped_bytes(&mut []).unwrap();
    mapped.read_mapped_bytes(&mut output[1..5]).unwrap();
    assert_eq!(&output[1..5], &[16, 17, 2, 3]);
    assert_eq!(output[0], u8::MAX);
    assert_eq!(&output[5..], &[u8::MAX; 11]);

    // A device transfer also observes the flushed write and unchanged suffix.
    gpu.memory().download_bytes(&mapped, &mut output).unwrap();
    assert_eq!(&output[..2], &[16, 17]);
    assert_eq!(&output[2..], &input[2..]);
}
