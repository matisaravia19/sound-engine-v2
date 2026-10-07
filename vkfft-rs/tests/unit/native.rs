//! Regression coverage for the patched dependency, including its internal failure paths.

#[test]
fn native_codegen_rader_temporaries_release_nested_and_partial_allocations() {
    unsafe extern "C" {
        fn vkfft_test_codegen_rader_storage() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_codegen_rader_storage() }, 0);
}

#[test]
fn native_codegen_permutations_preserve_names_and_reject_invalid_maps() {
    unsafe extern "C" {
        fn vkfft_test_codegen_permutations() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_codegen_permutations() }, 0);
}

#[test]
fn native_lut_storage_releases_temporary_tables_and_skips_borrowed_r2c_work() {
    unsafe extern "C" {
        fn vkfft_test_lut_ownership() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_lut_ownership() }, 0);
}

#[test]
fn native_lut_layout_rejects_invalid_offsets_before_allocation() {
    unsafe extern "C" {
        fn vkfft_test_lut_layout_bounds() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_lut_layout_bounds() }, 0);
}

#[test]
fn native_lut_reuse_requires_compatible_initialized_storage() {
    unsafe extern "C" {
        fn vkfft_test_lut_sharing_compatibility() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_lut_sharing_compatibility() }, 0);
}

#[test]
fn native_lut_borrowers_clear_handles_without_freeing_shared_storage() {
    unsafe extern "C" {
        fn vkfft_test_lut_shared_teardown() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_lut_shared_teardown() }, 0);
}

#[test]
fn native_lut_rader_offsets_match_values_and_cached_table_extent() {
    unsafe extern "C" {
        fn vkfft_test_lut_rader_layout_and_reuse() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_lut_rader_layout_and_reuse() }, 0);
}

#[test]
fn native_lut_uploads_initialized_values_and_rejects_byte_size_overflow() {
    unsafe extern "C" {
        fn vkfft_test_lut_values_and_bounds() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_lut_values_and_bounds() }, 0);
}

#[test]
fn native_double_double_conversion_stops_after_allocation_failure() {
    unsafe extern "C" {
        fn vkfft_test_double_double_conversion_failure() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_double_double_conversion_failure() }, 0);
}

#[test]
fn native_codegen_radix_kernels_need_no_temporary_allocations() {
    unsafe extern "C" {
        fn vkfft_test_codegen_radix_allocations() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_codegen_radix_allocations() }, 0);
}

#[test]
fn native_codegen_radix_stages_release_storage_on_failures_and_exceptions() {
    unsafe extern "C" {
        fn vkfft_test_codegen_radix_stage_storage() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_codegen_radix_stage_storage() }, 0);
}

#[test]
fn native_codegen_radix_stages_reject_invalid_register_layouts() {
    unsafe extern "C" {
        fn vkfft_test_codegen_radix_stage_layout() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_codegen_radix_stage_layout() }, 0);
}

#[test]
fn native_stage_policy_checks_normalization_and_dispatch() {
    unsafe extern "C" {
        fn vkfft_test_codegen_stage_policy() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_codegen_stage_policy() }, 0);
}

#[test]
fn native_direct_rader_avoids_allocations_and_uses_integer_radices() {
    unsafe extern "C" {
        fn vkfft_test_codegen_direct_rader() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_codegen_direct_rader() }, 0);
}

#[test]
fn native_rader_rejects_invalid_subgroups_and_register_layouts() {
    unsafe extern "C" {
        fn vkfft_test_codegen_rader_layout() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_codegen_rader_layout() }, 0);
}

#[test]
fn native_rader_accepts_batched_and_boosted_register_layouts() {
    unsafe extern "C" {
        fn vkfft_test_codegen_rader_layout_valid() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_codegen_rader_layout_valid() }, 0);
}

#[test]
fn native_scheduler_bounds_stages_and_resolves_nested_registers() {
    unsafe extern "C" {
        fn vkfft_test_scheduler_metadata() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_scheduler_metadata() }, 0);
}

#[test]
fn native_scheduler_plans_preserve_dimensions_and_release_failed_allocations() {
    unsafe extern "C" {
        fn vkfft_test_scheduler_plans() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_scheduler_plans() }, 0);
}

#[test]
fn native_scheduler_retry_preserves_completed_axes_and_releases_partial_state() {
    unsafe extern "C" {
        fn vkfft_test_scheduler_retry_exits() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_scheduler_retry_exits() }, 0);
}

#[test]
fn native_scheduler_retry_handles_real_bluestein_fallback_without_leaking() {
    unsafe extern "C" {
        fn vkfft_test_scheduler_retry_real() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_scheduler_retry_real() }, 0);
}

#[test]
fn native_scheduler_primitive_roots_match_independent_orders() {
    unsafe extern "C" {
        fn vkfft_test_scheduler_primitive_roots() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_scheduler_primitive_roots() }, 0);
}

#[test]
fn native_scheduler_rader_counts_do_not_overflow_or_lose_ownership() {
    unsafe extern "C" {
        fn vkfft_test_scheduler_rader_counts() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_scheduler_rader_counts() }, 0);
}

#[test]
fn native_scheduler_rejects_invalid_factor_limits_before_allocation() {
    unsafe extern "C" {
        fn vkfft_test_scheduler_factor_limits() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_scheduler_factor_limits() }, 0);
}

#[test]
fn native_scheduler_preserves_full_width_batches_and_workspace_sizes() {
    unsafe extern "C" {
        fn vkfft_test_scheduler_wide_sizes() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_scheduler_wide_sizes() }, 0);
}

#[test]
fn native_scheduler_rejects_overflow_before_publishing_buffer_sizes() {
    unsafe extern "C" {
        fn vkfft_test_scheduler_size_failures() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_scheduler_size_failures() }, 0);
}

#[test]
fn native_scheduler_checks_real_transform_extents_before_factorization() {
    unsafe extern "C" {
        fn vkfft_test_scheduler_transform_extents() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_scheduler_transform_extents() }, 0);
}

#[test]
fn native_scheduler_validates_bluestein_padding_and_real_retry() {
    unsafe extern "C" {
        fn vkfft_test_scheduler_padding_extents() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_scheduler_padding_extents() }, 0);
}

#[test]
fn native_scheduler_padding_search_matches_reference_and_checks_boundaries() {
    unsafe extern "C" {
        fn vkfft_test_padding_search() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_padding_search() }, 0);
}

#[test]
fn native_scheduler_padding_preserves_register_policy_and_errors() {
    unsafe extern "C" {
        fn vkfft_test_padding_register_policy() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_padding_register_policy() }, 0);
}

#[test]
fn native_scheduler_large_bluestein_plans_preserve_width_and_factors() {
    unsafe extern "C" {
        fn vkfft_test_scheduler_wide_padding() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_scheduler_wide_padding() }, 0);
}

#[test]
fn native_scheduler_rejects_invalid_resources_before_mutation() {
    unsafe extern "C" {
        fn vkfft_test_scheduler_invalid_resources() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_scheduler_invalid_resources() }, 0);
}

#[test]
fn native_scheduler_normalizes_coalescing_and_falls_back_within_resource_limits() {
    unsafe extern "C" {
        fn vkfft_test_scheduler_coalescing_limits() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_scheduler_coalescing_limits() }, 0);
}

#[test]
fn native_scheduler_bounds_boost_hints_and_keeps_pass_counts_positive() {
    unsafe extern "C" {
        fn vkfft_test_scheduler_boost_hints() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_scheduler_boost_hints() }, 0);
}

#[test]
fn native_codegen_names_respect_capacity_and_propagate_errors() {
    unsafe extern "C" {
        fn vkfft_test_codegen_names() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_codegen_names() }, 0);
}

#[test]
fn native_codegen_copy_preserves_nested_allocation_ownership() {
    unsafe extern "C" {
        fn vkfft_test_codegen_copy_ownership() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_codegen_copy_ownership() }, 0);
}

#[test]
fn native_codegen_registers_clean_up_after_errors() {
    unsafe extern "C" {
        fn vkfft_test_codegen_registers() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_codegen_registers() }, 0);
}

#[test]
fn native_codegen_formatting_respects_temporary_capacity() {
    unsafe extern "C" {
        fn vkfft_test_codegen_formatting() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_codegen_formatting() }, 0);
}

#[test]
fn native_codegen_parameters_clean_up_after_each_allocation_failure() {
    unsafe extern "C" {
        fn vkfft_test_codegen_parameters() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_codegen_parameters() }, 0);
}

#[test]
fn native_codegen_containers_release_nested_and_partial_allocations() {
    unsafe extern "C" {
        fn vkfft_test_codegen_containers() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_codegen_containers() }, 0);
}

#[test]
fn native_codegen_text_checks_capacity_before_writing() {
    unsafe extern "C" {
        fn vkfft_test_codegen_bounds() -> i32;
    }
    assert_eq!(unsafe { vkfft_test_codegen_bounds() }, 0);
}

#[test]
fn native_compiler_failures_release_resources_and_contain_exceptions() {
    unsafe extern "C" {
        fn vkfft_test_compiler_failures() -> i32;
    }
    assert_eq!(
        unsafe { vkfft_test_compiler_failures() },
        0,
        "native compiler regression failed at this C++ source line"
    );
}

#[test]
fn recursive_native_failures_preserve_parent_kernel_ownership() {
    unsafe extern "C" {
        fn vkfft_test_recursive_failures() -> i32;
    }
    assert_eq!(
        unsafe { vkfft_test_recursive_failures() },
        0,
        "recursive initialization regression failed at this C++ source line"
    );
}

#[test]
fn native_transfer_failures_release_resources_before_returning() {
    unsafe extern "C" {
        fn vkfft_test_memory_failures() -> i32;
    }
    assert_eq!(
        unsafe { vkfft_test_memory_failures() },
        0,
        "native transfer regression failed at this C++ source line"
    );
}
