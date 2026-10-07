use super::*;

fn builder() -> FFTPlanBuilder {
    // Validation-only sentinel handles: these tests never enter Vulkan or VkFFT.
    let handles = DeviceHandles {
        physical_device: ash::vk::PhysicalDevice::from_raw(1),
        device: ash::vk::Device::from_raw(1),
        queue: ash::vk::Queue::from_raw(1),
        command_pool: ash::vk::CommandPool::from_raw(1),
        fence: ash::vk::Fence::from_raw(1),
    };
    FFTPlanBuilder::new(&handles)
        .with_length(8)
        .with_buffer(ash::vk::Buffer::from_raw(1), 64)
}

#[test]
fn configuration_rejects_missing_handles_and_short_buffers() {
    assert_eq!(builder().validate(), Ok(()));
    let mut invalid = builder();
    invalid.handles.fence = 0;
    assert_eq!(invalid.validate(), Err(FftError::INVALID_FENCE));
    let mut invalid = builder();
    invalid.handles.buffer = 0;
    assert_eq!(invalid.validate(), Err(FftError::EMPTY_BUFFER));
    assert_eq!(
        builder()
            .with_buffer(ash::vk::Buffer::from_raw(1), 63)
            .validate(),
        Err(FftError::EMPTY_BUFFER_SIZE)
    );
}

#[test]
fn configuration_rejects_unsupported_lengths_before_entering_native_code() {
    assert_eq!(
        builder().with_length(0).validate(),
        Err(FftError::EMPTY_SIZE)
    );
    for length in [3, 6, 17, 65_537, u64::MAX] {
        assert_eq!(
            builder().with_length(length).validate(),
            Err(FftError::UNSUPPORTED_LENGTH)
        );
    }
    assert_eq!(
        builder().with_length(1 << 63).validate(),
        Err(FftError::INSUFFICIENT_STORAGE)
    );
}
