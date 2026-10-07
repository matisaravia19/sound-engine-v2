use super::*;

#[test]
fn push_writes_check_alignment_overflow_bounds_and_stage_visibility() {
    let ranges = [vk::PushConstantRange::default()
        .offset(4)
        .size(8)
        .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)];
    assert_eq!(
        push_constant_stages(&ranges, 4, 8).unwrap(),
        vk::ShaderStageFlags::RAYGEN_KHR
    );
    for (offset, size) in [
        (0, 8),
        (4, 12),
        (5, 4),
        (4, 3),
        (u32::MAX - 3, 8),
        (0, usize::MAX),
    ] {
        assert!(push_constant_stages(&ranges, offset, size).is_err());
    }
}

#[test]
fn descriptors_reject_missing_duplicate_wrong_type_and_uninitialized_arrays() {
    let binding = vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER);
    let write = (0, vk::DescriptorType::STORAGE_BUFFER);
    assert!(validate_descriptor_writes(&[binding], &[write]).is_ok());
    assert!(validate_descriptor_writes(&[binding], &[]).is_err());
    assert!(validate_descriptor_writes(&[binding], &[write, write]).is_err());
    assert!(
        validate_descriptor_writes(&[binding], &[(0, vk::DescriptorType::UNIFORM_BUFFER)]).is_err()
    );
    assert!(validate_descriptor_writes(&[binding.descriptor_count(2)], &[write]).is_err());
}
