use crate::ffi::*;
use ash::vk::Handle;
use std::sync::Mutex;

// VkFFT initializes/finalizes process-global glslang state during plan creation.
static COMPILER_ACCESS: Mutex<()> = Mutex::new(());

/// Owns a VkFFT application and stable storage for the handle pointers it retains.
/// The caller must keep the Vulkan device and buffers alive, serialize recording,
/// and finish submitted work before dropping the plan.
pub struct FFTPlan {
    app: VkFFTApp,
    // VkFFT retains pointers into this allocation for the entire plan lifetime.
    _handles: Box<HandleStorage>,
}

// The plan owns an opaque VkFFT application and copied Vulkan handle values.
// Moving it to another thread does not invalidate those handles; callers still
// need external synchronization for command recording and device use.
unsafe impl Send for FFTPlan {}

/// Owns configuration storage until it is transferred into an FFT plan.
/// Referenced Vulkan resources remain externally owned.
pub struct FFTPlanBuilder {
    config: VkFFTConfiguration,
    handles: Box<HandleStorage>,
}

#[derive(Default)]
struct HandleStorage {
    physical_device: VkPhysicalDevice,
    device: VkDevice,
    queue: VkQueue,
    command_pool: VkCommandPool,
    fence: VkFence,

    buffer: VkBuffer,
    buffer_size: u64,

    kernel: VkBuffer,
    kernel_size: u64,
}

/// Borrowed Vulkan handle values; the caller retains ownership and synchronization.
pub struct DeviceHandles {
    /// Physical device associated with the logical device.
    pub physical_device: ash::vk::PhysicalDevice,
    /// Logical device used by the FFT plan.
    pub device: ash::vk::Device,
    /// Queue used for VkFFT initialization work.
    pub queue: ash::vk::Queue,
    /// Command pool compatible with the queue family.
    pub command_pool: ash::vk::CommandPool,
    /// Fence available for VkFFT initialization synchronization.
    pub fence: ash::vk::Fence,
}

impl FFTPlanBuilder {
    /// Copies handle values into stable storage; does not take ownership of Vulkan resources.
    pub fn new(handles: &DeviceHandles) -> Self {
        let mut handle_storage = Box::new(HandleStorage {
            physical_device: handles.physical_device.as_raw() as VkPhysicalDevice,
            device: handles.device.as_raw() as VkDevice,
            queue: handles.queue.as_raw() as VkQueue,
            command_pool: handles.command_pool.as_raw() as VkCommandPool,
            fence: handles.fence.as_raw() as VkFence,
            ..Default::default()
        });

        let config = VkFFTConfiguration {
            physical_device: std::ptr::from_mut(&mut handle_storage.physical_device),
            device: std::ptr::from_mut(&mut handle_storage.device),
            queue: std::ptr::from_mut(&mut handle_storage.queue),
            command_pool: std::ptr::from_mut(&mut handle_storage.command_pool),
            fence: std::ptr::from_mut(&mut handle_storage.fence),
            ..Default::default()
        };

        Self {
            config,
            handles: handle_storage,
        }
    }

    /// Initializes VkFFT and transfers configuration storage into the owned plan.
    ///
    /// # Safety
    /// All handles must be valid and belong to the same live device. The caller
    /// must exclusively synchronize queue/pool/fence access and keep the device,
    /// loader, buffers and initialization handles alive until the plan is dropped.
    pub unsafe fn build(self) -> Result<FFTPlan, VkFFTResult> {
        self.validate()?;
        let _compiler = COMPILER_ACCESS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let app = unsafe { vkfft_app_create() };
        if app.is_null() {
            return Err(VkFFTResult::ErrorEmptyApp);
        }

        let result = unsafe { vkfft_init(app, &self.config) };
        if result != VkFFTResult::Success {
            unsafe { vkfft_app_destroy(app) };
            return Err(result);
        }

        Ok(FFTPlan {
            app,
            _handles: self.handles,
        })
    }

    fn validate(&self) -> Result<(), VkFFTResult> {
        if self.handles.physical_device.is_null() {
            return Err(VkFFTResult::ErrorInvalidPhysicalDevice);
        }
        if self.handles.device.is_null() {
            return Err(VkFFTResult::ErrorInvalidDevice);
        }
        if self.handles.queue.is_null() {
            return Err(VkFFTResult::ErrorInvalidQueue);
        }
        if self.handles.command_pool == 0 {
            return Err(VkFFTResult::ErrorInvalidCommandPool);
        }
        if self.handles.fence == 0 {
            return Err(VkFFTResult::ErrorInvalidFence);
        }
        if self.config.dimensions == 0 {
            return Err(VkFFTResult::ErrorEmptyFftdim);
        }
        let sizes = &self.config.size[..self.config.dimensions as usize];
        if sizes.contains(&0) {
            return Err(VkFFTResult::ErrorEmptySize);
        }
        let bytes = sizes
            .iter()
            .try_fold(8_u64, |bytes, size| bytes.checked_mul(*size))
            .and_then(|bytes| bytes.checked_mul(self.config.number_batches.max(1)))
            .ok_or(VkFFTResult::ErrorInsufficientTempBuffer)?;
        if self.handles.buffer == 0 {
            return Err(VkFFTResult::ErrorEmptyBuffer);
        }
        if self.handles.buffer_size < bytes {
            return Err(VkFFTResult::ErrorEmptyBufferSize);
        }
        if self.config.perform_convolution != 0 {
            if self.handles.kernel == 0 {
                return Err(VkFFTResult::ErrorEmptyKernel);
            }
            if self.handles.kernel_size < bytes {
                return Err(VkFFTResult::ErrorEmptyKernelSize);
            }
        }
        Ok(())
    }

    /// Sets transform dimensions; panics if the dimension count is outside 1..=4.
    pub fn with_dimensions(mut self, dimensions: &[u64]) -> Self {
        assert!(dimensions.len() > 0 && dimensions.len() <= VKFFT_MAX_FFT_DIMENSIONS);

        self.config.dimensions = dimensions.len() as u64;
        for (i, &dim) in dimensions.iter().enumerate() {
            self.config.size[i] = dim;
        }

        self
    }

    /// Configures a one-dimensional transform of the given length.
    pub fn with_single_dimension(self, size: u64) -> Self {
        self.with_dimensions(&[size])
    }

    /// Borrows the transform buffer and records its available size in bytes.
    pub fn with_buffer(mut self, buffer: ash::vk::Buffer, buffer_size: u64) -> Self {
        self.handles.buffer = buffer.as_raw() as VkBuffer;
        self.handles.buffer_size = buffer_size;
        self.config.buffer = std::ptr::from_mut(&mut self.handles.buffer);
        self.config.buffer_size = std::ptr::from_mut(&mut self.handles.buffer_size);
        self
    }

    /// Transforms `count` contiguous arrays with the configured dimensions.
    /// The supplied buffer must hold every batch. Panics if `count` is zero.
    pub fn with_batch_count(mut self, count: u64) -> Self {
        assert!(count > 0, "FFT batch count must be positive");
        self.config.number_batches = count;
        self
    }

    /// Requests inverse-transform normalization from VkFFT.
    pub fn with_normalization(mut self) -> Self {
        self.config.normalize = 1;
        self
    }

    /// Configures the transform for a convolution kernel.
    pub fn for_kernel(mut self) -> Self {
        self.config.kernel_convolution = 1;
        self
    }

    /// Borrows a frequency-domain kernel buffer for convolution.
    pub fn for_convolution(mut self, kernel: ash::vk::Buffer, kernel_size: u64) -> Self {
        self.handles.kernel = kernel.as_raw() as VkBuffer;
        self.handles.kernel_size = kernel_size;
        self.config.kernel = std::ptr::from_mut(&mut self.handles.kernel);
        self.config.kernel_size = std::ptr::from_mut(&mut self.handles.kernel_size);
        self.config.perform_convolution = 1;
        self
    }
}

impl FFTPlan {
    /// Records a forward FFT into a caller-owned command buffer; does not submit or wait.
    ///
    /// # Safety
    /// The command buffer must be recording on this plan's device. Synchronize
    /// buffer access and retain the plan and its resources until execution ends.
    pub unsafe fn append(&self, command_buffer: ash::vk::CommandBuffer) -> Result<(), VkFFTResult> {
        self.append_internal(command_buffer, 0)
    }

    /// Records an inverse FFT; the caller synchronizes buffer access and completion.
    ///
    /// # Safety
    /// The same recording, lifetime and synchronization requirements as `append` apply.
    pub unsafe fn append_inverse(
        &self,
        command_buffer: ash::vk::CommandBuffer,
    ) -> Result<(), VkFFTResult> {
        self.append_internal(command_buffer, 1)
    }

    fn append_internal(
        &self,
        command_buffer: ash::vk::CommandBuffer,
        inverse: i32,
    ) -> Result<(), VkFFTResult> {
        let mut command_buffer_handle = command_buffer.as_raw() as VkCommandBuffer;
        let params = VkFFTLaunchParams {
            command_buffer: std::ptr::from_mut(&mut command_buffer_handle),
        };

        let result = unsafe { vkfft_exec(self.app, inverse, &params) };
        if result != VkFFTResult::Success {
            return Err(result);
        }

        Ok(())
    }
}

impl Drop for FFTPlan {
    fn drop(&mut self) {
        let _compiler = COMPILER_ACCESS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        unsafe {
            vkfft_delete_plan(self.app);
            vkfft_app_destroy(self.app);
        }
    }
}

#[cfg(test)]
mod tests {
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
            .with_single_dimension(8)
            .with_buffer(ash::vk::Buffer::from_raw(1), 64)
    }

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

    #[test]
    fn configuration_rejects_null_fence_short_buffer_and_dimension_overflow() {
        assert_eq!(builder().validate(), Ok(()));
        let mut invalid = builder();
        invalid.handles.fence = 0;
        assert_eq!(invalid.validate(), Err(VkFFTResult::ErrorInvalidFence));
        assert_eq!(
            builder().with_batch_count(2).validate(),
            Err(VkFFTResult::ErrorEmptyBufferSize)
        );
        assert_eq!(
            builder().with_single_dimension(0).validate(),
            Err(VkFFTResult::ErrorEmptySize)
        );
        assert_eq!(
            builder().with_single_dimension(u64::MAX).validate(),
            Err(VkFFTResult::ErrorInsufficientTempBuffer)
        );
    }
}
