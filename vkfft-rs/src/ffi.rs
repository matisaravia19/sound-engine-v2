//! C ABI shared with `ffi.cpp`; retained pointers borrow plan-owned handle storage.

use std::ffi::c_void;

/// Opaque application owned and destroyed by an FFT plan.
pub(crate) type VkFFTApp = *mut c_void;

/// Borrowed dispatchable physical-device handle.
pub(crate) type VkPhysicalDevice = *mut c_void;
/// Borrowed dispatchable logical-device handle.
pub(crate) type VkDevice = *mut c_void;
/// Borrowed initialization queue handle.
pub(crate) type VkQueue = *mut c_void;
/// Borrowed command buffer receiving FFT commands.
pub(crate) type VkCommandBuffer = *mut c_void;
/// Non-dispatchable Vulkan command pool handle, always 64 bits.
pub(crate) type VkCommandPool = u64;
/// Non-dispatchable Vulkan fence handle, always 64 bits.
pub(crate) type VkFence = u64;
/// Non-dispatchable Vulkan buffer handle, always 64 bits.
pub(crate) type VkBuffer = u64;

/// Native plan inputs; every handle pointer remains valid until plan destruction.
#[repr(C)]
#[derive(Default)]
pub(crate) struct VkFFTConfiguration {
    /// Power-of-two complex sample count for a single one-dimensional transform.
    pub(crate) length: u64,

    /// Stable pointer to the borrowed physical device.
    pub(crate) physical_device: *mut VkPhysicalDevice,
    /// Stable pointer to the borrowed logical device.
    pub(crate) device: *mut VkDevice,
    /// Stable pointer to the externally synchronized initialization queue.
    pub(crate) queue: *mut VkQueue,
    /// Stable pointer to the externally synchronized initialization command pool.
    pub(crate) command_pool: *mut VkCommandPool,
    /// Stable pointer to the externally synchronized initialization fence.
    pub(crate) fence: *mut VkFence,

    /// Stable pointer to the borrowed transform buffer.
    pub(crate) buffer: *mut VkBuffer,
    /// Stable pointer to the transform buffer capacity in bytes.
    pub(crate) buffer_size: *mut u64,

    /// Nonzero requests normalized inverse transforms.
    pub(crate) normalize: u64,
}

/// Borrowed command buffer pointer used only during command recording.
#[repr(C)]
pub(crate) struct VkFFTLaunchParams {
    /// Pointer to a caller-owned command buffer valid for this call.
    pub(crate) command_buffer: *mut VkCommandBuffer,
}

unsafe extern "C" {
    /// Allocates an empty application; returns null on allocation failure.
    pub(crate) fn vkfft_app_create() -> VkFFTApp;
    /// Deletes an application after its native plan has been released.
    pub(crate) fn vkfft_app_destroy(app: VkFFTApp);

    /// Initializes an application using borrowed handles; returns a native status code.
    pub(crate) fn vkfft_init(app: VkFFTApp, config: *const VkFFTConfiguration) -> i32;
    /// Records a transform without submitting it; returns a native status code.
    pub(crate) fn vkfft_exec(app: VkFFTApp, inverse: i32, params: *const VkFFTLaunchParams) -> i32;
    /// Releases native plan resources after all submitted work has completed.
    pub(crate) fn vkfft_delete_plan(app: VkFFTApp);
}
