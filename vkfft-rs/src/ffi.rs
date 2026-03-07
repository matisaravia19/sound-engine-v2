#![allow(dead_code)]

use std::ffi::c_void;

pub const VKFFT_MAX_FFT_DIMENSIONS: usize = 4;

pub type VkFFTApp = *mut c_void;

pub type VkPhysicalDevice = *mut c_void;
pub type VkDevice = *mut c_void;
pub type VkQueue = *mut c_void;
pub type VkCommandBuffer = *mut c_void;
pub type VkCommandPool = usize;
pub type VkFence = usize;
pub type VkBuffer = usize;

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum VkFFTResult {
    Success = 0,
    ErrorMallocFailed = 1,
    ErrorInsufficientCodeBuffer = 2,
    ErrorInsufficientTempBuffer = 3,
    ErrorPlanNotInitialized = 4,
    ErrorNullTempPassed = 5,
    ErrorMathFailed = 6,
    ErrorFftdimGtMaxFftDimensions = 7,
    ErrorNonzeroAppInitialization = 8,
    ErrorInvalidPhysicalDevice = 1001,
    ErrorInvalidDevice = 1002,
    ErrorInvalidQueue = 1003,
    ErrorInvalidCommandPool = 1004,
    ErrorInvalidFence = 1005,
    ErrorOnlyForwardFftInitialized = 1006,
    ErrorOnlyInverseFftInitialized = 1007,
    ErrorInvalidContext = 1008,
    ErrorInvalidPlatform = 1009,
    ErrorEnabledSaveApplicationToString = 1010,
    ErrorEmptyFile = 1011,
    ErrorEmptyFftdim = 2001,
    ErrorEmptySize = 2002,
    ErrorEmptyBufferSize = 2003,
    ErrorEmptyBuffer = 2004,
    ErrorEmptyTempBufferSize = 2005,
    ErrorEmptyTempBuffer = 2006,
    ErrorEmptyInputBufferSize = 2007,
    ErrorEmptyInputBuffer = 2008,
    ErrorEmptyOutputBufferSize = 2009,
    ErrorEmptyOutputBuffer = 2010,
    ErrorEmptyKernelSize = 2011,
    ErrorEmptyKernel = 2012,
    ErrorEmptyApplicationString = 2013,
    ErrorEmptyUseCustomBluesteinPaddingPatternArrays = 2014,
    ErrorEmptyApp = 2015,
    ErrorInvalidUserTempBufferTooSmall = 2016,
    ErrorUnsupportedRadix = 3001,
    ErrorUnsupportedFftLength = 3002,
    ErrorUnsupportedFftLengthR2c = 3003,
    ErrorUnsupportedFftLengthR2r = 3004,
    ErrorUnsupportedFftOmit = 3005,
    ErrorFailedToAllocate = 4001,
    ErrorFailedToMapMemory = 4002,
    ErrorFailedToAllocateCommandBuffers = 4003,
    ErrorFailedToBeginCommandBuffer = 4004,
    ErrorFailedToEndCommandBuffer = 4005,
    ErrorFailedToSubmitQueue = 4006,
    ErrorFailedToWaitForFences = 4007,
    ErrorFailedToResetFences = 4008,
    ErrorFailedToCreateDescriptorPool = 4009,
    ErrorFailedToCreateDescriptorSetLayout = 4010,
    ErrorFailedToAllocateDescriptorSets = 4011,
    ErrorFailedToCreatePipelineLayout = 4012,
    ErrorFailedShaderPreprocess = 4013,
    ErrorFailedShaderParse = 4014,
    ErrorFailedShaderLink = 4015,
    ErrorFailedSpirvGenerate = 4016,
    ErrorFailedToCreateShaderModule = 4017,
    ErrorFailedToCreateInstance = 4018,
    ErrorFailedToSetupDebugMessenger = 4019,
    ErrorFailedToFindPhysicalDevice = 4020,
    ErrorFailedToCreateDevice = 4021,
    ErrorFailedToCreateFence = 4022,
    ErrorFailedToCreateCommandPool = 4023,
    ErrorFailedToCreateBuffer = 4024,
    ErrorFailedToAllocateMemory = 4025,
    ErrorFailedToBindBufferMemory = 4026,
    ErrorFailedToFindMemory = 4027,
    ErrorFailedToSynchronize = 4028,
    ErrorFailedToCopy = 4029,
    ErrorFailedToCreateProgram = 4030,
    ErrorFailedToCompileProgram = 4031,
    ErrorFailedToGetCodeSize = 4032,
    ErrorFailedToGetCode = 4033,
    ErrorFailedToDestroyProgram = 4034,
    ErrorFailedToLoadModule = 4035,
    ErrorFailedToGetFunction = 4036,
    ErrorFailedToSetDynamicSharedMemory = 4037,
    ErrorFailedToModuleGetGlobal = 4038,
    ErrorFailedToLaunchKernel = 4039,
    ErrorFailedToEventRecord = 4040,
    ErrorFailedToAddNameExpression = 4041,
    ErrorFailedToInitialize = 4042,
    ErrorFailedToSetDeviceId = 4043,
    ErrorFailedToGetDevice = 4044,
    ErrorFailedToCreateContext = 4045,
    ErrorFailedToCreatePipeline = 4046,
    ErrorFailedToSetKernelArg = 4047,
    ErrorFailedToCreateCommandQueue = 4048,
    ErrorFailedToReleaseCommandQueue = 4049,
    ErrorFailedToEnumerateDevices = 4050,
    ErrorFailedToGetAttribute = 4051,
    ErrorFailedToCreateEvent = 4052,
    ErrorFailedToCreateCommandList = 4053,
    ErrorFailedToDestroyCommandList = 4054,
    ErrorFailedToSubmitBarrier = 4055,
}

#[repr(C)]
#[derive(Default)]
pub struct VkFFTConfiguration {
    pub dimensions: u64,
    pub size: [u64; VKFFT_MAX_FFT_DIMENSIONS],

    pub physical_device: *mut VkPhysicalDevice,
    pub device: *mut VkDevice,
    pub queue: *mut VkQueue,
    pub command_pool: *mut VkCommandPool,
    pub fence: *mut VkFence,
    pub is_compiler_initialized: u64,

    pub buffer: *mut VkBuffer,
    pub buffer_size: *mut u64,

    pub kernel: *mut VkBuffer,
    pub kernel_size: *mut u64,

    pub normalize: u64,
    pub perform_convolution: u64,
    pub kernel_convolution: u64,
}

#[repr(C)]
pub struct VkFFTLaunchParams {
    pub command_buffer: *mut VkCommandBuffer,
}

unsafe extern "C" {
    pub fn vkfft_app_create() -> VkFFTApp;
    pub fn vkfft_app_destroy(app: VkFFTApp);

    pub fn vkfft_init(app: VkFFTApp, config: *const VkFFTConfiguration) -> VkFFTResult;
    pub fn vkfft_exec(app: VkFFTApp, inverse: i32, params: *const VkFFTLaunchParams) -> VkFFTResult;
    pub fn vkfft_delete_plan(app: VkFFTApp);
}
