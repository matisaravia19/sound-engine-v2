use crate::FftError;
use crate::ffi::*;
use ash::vk::Handle;
use std::sync::Mutex;

// VkFFT initializes/finalizes process-global glslang state during plan creation.
static COMPILER_ACCESS: Mutex<()> = Mutex::new(());

/// Owns a single one-dimensional complex FFT and the handle storage it retains.
/// Transform lengths are powers of two, matching the audio convolution pipeline.
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
    pub unsafe fn build(self) -> Result<FFTPlan, FftError> {
        self.validate()?;
        let _compiler = COMPILER_ACCESS
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let app = unsafe { vkfft_app_create() };
        if app.is_null() {
            return Err(FftError::EMPTY_APP);
        }

        let result = unsafe { vkfft_init(app, &self.config) };
        if let Err(error) = FftError::check(result) {
            unsafe { vkfft_app_destroy(app) };
            return Err(error);
        }

        Ok(FFTPlan {
            app,
            _handles: self.handles,
        })
    }

    fn validate(&self) -> Result<(), FftError> {
        if self.handles.physical_device.is_null() {
            return Err(FftError::INVALID_PHYSICAL_DEVICE);
        }
        if self.handles.device.is_null() {
            return Err(FftError::INVALID_DEVICE);
        }
        if self.handles.queue.is_null() {
            return Err(FftError::INVALID_QUEUE);
        }
        if self.handles.command_pool == 0 {
            return Err(FftError::INVALID_COMMAND_POOL);
        }
        if self.handles.fence == 0 {
            return Err(FftError::INVALID_FENCE);
        }
        if self.config.length == 0 {
            return Err(FftError::EMPTY_SIZE);
        }
        if !self.config.length.is_power_of_two() {
            return Err(FftError::UNSUPPORTED_LENGTH);
        }
        let bytes = self
            .config
            .length
            .checked_mul(std::mem::size_of::<[f32; 2]>() as u64)
            .ok_or(FftError::INSUFFICIENT_STORAGE)?;
        if self.handles.buffer == 0 {
            return Err(FftError::EMPTY_BUFFER);
        }
        if self.handles.buffer_size < bytes {
            return Err(FftError::EMPTY_BUFFER_SIZE);
        }
        Ok(())
    }

    /// Sets the complex sample count; `build` rejects zero and non-power-of-two lengths.
    pub fn with_length(mut self, length: u64) -> Self {
        self.config.length = length;
        self
    }

    /// Borrows the transform buffer and records its available size in bytes.
    pub fn with_buffer(mut self, buffer: ash::vk::Buffer, buffer_size: u64) -> Self {
        self.handles.buffer = buffer.as_raw() as VkBuffer;
        self.handles.buffer_size = buffer_size;
        self.config.buffer = std::ptr::from_mut(&mut self.handles.buffer);
        self.config.buffer_size = std::ptr::from_mut(&mut self.handles.buffer_size);
        self
    }

    /// Requests inverse-transform normalization from VkFFT.
    pub fn with_normalization(mut self) -> Self {
        self.config.normalize = 1;
        self
    }
}

impl FFTPlan {
    /// Records a forward FFT into a caller-owned command buffer; does not submit or wait.
    ///
    /// # Safety
    /// The command buffer must be recording on this plan's device. Synchronize
    /// buffer access and retain the plan and its resources until execution ends.
    pub unsafe fn append(&self, command_buffer: ash::vk::CommandBuffer) -> Result<(), FftError> {
        self.append_internal(command_buffer, 0)
    }

    /// Records an inverse FFT; the caller synchronizes buffer access and completion.
    ///
    /// # Safety
    /// The same recording, lifetime and synchronization requirements as `append` apply.
    pub unsafe fn append_inverse(
        &self,
        command_buffer: ash::vk::CommandBuffer,
    ) -> Result<(), FftError> {
        self.append_internal(command_buffer, 1)
    }

    fn append_internal(
        &self,
        command_buffer: ash::vk::CommandBuffer,
        inverse: i32,
    ) -> Result<(), FftError> {
        let mut command_buffer_handle = command_buffer.as_raw() as VkCommandBuffer;
        let params = VkFFTLaunchParams {
            command_buffer: std::ptr::from_mut(&mut command_buffer_handle),
        };

        let result = unsafe { vkfft_exec(self.app, inverse, &params) };
        FftError::check(result)
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
#[path = "../tests/unit/plan.rs"]
mod tests;
