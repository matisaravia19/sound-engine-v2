use super::*;
use glam::{Mat4, Vec3};

#[test]
#[ignore = "requires Vulkan ray tracing; run hardware tests serially"]
fn empty_tlas_builds_without_placeholder_blas_and_releases_storage() {
    let gpu = crate::gpu::backend::VkBackend::new().unwrap();
    for _ in 0..3 {
        let id = gpu
            .rt()
            .build_tlas(TlasBuildSpec {
                instances: &[],
                flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
            })
            .unwrap();
        assert_eq!(gpu.rt().resource_counts(), (0, 1));
        assert!(
            gpu.rt().tlas.lock().unwrap()[&id]
                ._referenced_blas
                .is_empty()
        );
        // This test owns the TLAS and has never submitted a trace referencing it.
        unsafe { gpu.rt().destroy_tlas(id) }.unwrap();
        assert_eq!(gpu.rt().resource_counts(), (0, 0));
    }
}

#[test]
fn build_flags_reject_unknown_features_and_conflicting_preferences() {
    use vk::BuildAccelerationStructureFlagsKHR as Flags;
    for flags in [
        Flags::empty(),
        Flags::PREFER_FAST_TRACE | Flags::ALLOW_UPDATE,
        Flags::PREFER_FAST_BUILD | Flags::LOW_MEMORY | Flags::ALLOW_COMPACTION,
    ] {
        assert!(validate_build_flags(flags).is_ok());
    }
    for flags in [
        Flags::MOTION_NV,
        Flags::from_raw(u32::MAX),
        Flags::PREFER_FAST_TRACE | Flags::PREFER_FAST_BUILD,
    ] {
        assert!(validate_build_flags(flags).is_err());
    }
}

#[test]
fn scratch_device_address_alignment_handles_unaligned_allocations() {
    assert_eq!(align_device_address(0x1000, 128), Some(0x1000));
    assert_eq!(align_device_address(0x1001, 128), Some(0x1080));
    assert_eq!(align_device_address(0x107f, 128), Some(0x1080));
    assert_eq!(align_device_address(u64::MAX, 128), None);
    assert_eq!(align_device_address(0x1001, 0), None);
    assert_eq!(align_device_address(0x1001, 96), None);
}

#[test]
fn translation_transform_is_converted_correctly() {
    let matrix = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));

    let result = matrix_to_tlas_transform(matrix);

    assert_eq!(
        result,
        [
            1.0, 0.0, 0.0, 1.0, // row 0
            0.0, 1.0, 0.0, 2.0, // row 1
            0.0, 0.0, 1.0, 3.0, // row 2
        ]
    );
}
