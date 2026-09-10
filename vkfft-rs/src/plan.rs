use crate::ffi::*;
use ash::vk::Handle;

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
    pub fn build(self) -> Result<FFTPlan, VkFFTResult> {
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
    pub fn append(&self, command_buffer: ash::vk::CommandBuffer) -> Result<(), VkFFTResult> {
        self.append_internal(command_buffer, 0)
    }

    /// Records an inverse FFT; the caller synchronizes buffer access and completion.
    pub fn append_inverse(
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
        unsafe {
            vkfft_delete_plan(self.app);
            vkfft_app_destroy(self.app);
        }
    }
}
