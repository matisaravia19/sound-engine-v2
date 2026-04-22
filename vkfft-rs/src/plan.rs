use crate::ffi::*;
use ash::vk::Handle;

pub struct FFTPlan {
    app: VkFFTApp,
}

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

pub struct DeviceHandles {
    pub physical_device: ash::vk::PhysicalDevice,
    pub device: ash::vk::Device,
    pub queue: ash::vk::Queue,
    pub command_pool: ash::vk::CommandPool,
    pub fence: ash::vk::Fence,
}

impl FFTPlanBuilder {
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

    pub fn build(&mut self) -> Result<FFTPlan, VkFFTResult> {
        let app = unsafe { vkfft_app_create() };
        if app.is_null() {
            return Err(VkFFTResult::ErrorEmptyApp);
        }

        let result = unsafe { vkfft_init(app, &self.config) };
        if result != VkFFTResult::Success {
            unsafe { vkfft_app_destroy(app) };
            return Err(result);
        }

        Ok(FFTPlan { app })
    }

    pub fn with_dimensions(&mut self, dimensions: &[u64]) -> &mut Self {
        assert!(dimensions.len() > 0 && dimensions.len() <= VKFFT_MAX_FFT_DIMENSIONS);

        self.config.dimensions = dimensions.len() as u64;
        for (i, &dim) in dimensions.iter().enumerate() {
            self.config.size[i] = dim;
        }

        self
    }

    pub fn with_single_dimension(&mut self, size: u64) -> &mut Self {
        self.with_dimensions(&[size])
    }

    pub fn with_buffer(&mut self, buffer: ash::vk::Buffer, buffer_size: u64) -> &mut Self {
        self.handles.buffer = buffer.as_raw() as VkBuffer;
        self.handles.buffer_size = buffer_size;
        self.config.buffer = std::ptr::from_mut(&mut self.handles.buffer);
        self.config.buffer_size = std::ptr::from_mut(&mut self.handles.buffer_size);
        self
    }

    pub fn with_normalization(&mut self) -> &mut Self {
        self.config.normalize = 1;
        self
    }

    pub fn for_kernel(&mut self) -> &mut Self {
        self.config.kernel_convolution = 1;
        self
    }

    pub fn for_convolution(&mut self, kernel: ash::vk::Buffer, kernel_size: u64) -> &mut Self {
        self.handles.kernel = kernel.as_raw() as VkBuffer;
        self.handles.kernel_size = kernel_size;
        self.config.kernel = std::ptr::from_mut(&mut self.handles.kernel);
        self.config.kernel_size = std::ptr::from_mut(&mut self.handles.kernel_size);
        self.config.perform_convolution = 1;
        self
    }
}

impl FFTPlan {
    pub fn append(&self, command_buffer: ash::vk::CommandBuffer) -> Result<(), VkFFTResult> {
        self.append_internal(command_buffer, 0)
    }

    pub fn append_inverse(&self, command_buffer: ash::vk::CommandBuffer) -> Result<(), VkFFTResult> {
        self.append_internal(command_buffer, 1)
    }

    fn append_internal(&self, command_buffer: ash::vk::CommandBuffer, inverse: i32) -> Result<(), VkFFTResult> {
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
