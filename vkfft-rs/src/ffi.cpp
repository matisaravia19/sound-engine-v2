#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <new>

#define VKFFT_BACKEND 0
// Load the scoped Vulkan helpers before VkFFT's include guard selects its own.
// These scalar aliases match the default precision configuration in vkFFT.h.
#define VKFFT_MAX_FFT_DIMENSIONS 4
#define pfUINT uint64_t
#define pfINT int64_t
#define pfLD long double
#include "vulkan_memory.hpp"
#include "vulkan_init.hpp"
#include "vulkan_compiler.hpp"
#include "ffi_guard.hpp"

#include "vkFFT.h"

static_assert(VKFFT_MAX_FFT_DIMENSIONS == 4, "Rust VkFFT FFI dimensions must match the bundled library");

struct FFIVkFFTConfiguration
{
    uint64_t dimensions;
    uint64_t size[VKFFT_MAX_FFT_DIMENSIONS];

    VkPhysicalDevice *physicalDevice;
    VkDevice *device;
    VkQueue *queue;
    VkCommandPool *commandPool;
    VkFence *fence;
    uint64_t isCompilerInitialized;

    VkBuffer *buffer;
    uint64_t *bufferSize;

    VkBuffer *kernel;
    uint64_t *kernelSize;

    uint64_t normalize;
    uint64_t performConvolution;
    uint64_t kernelConvolution;
    uint64_t numberBatches;
};

struct FFIVkFFTLaunchParams
{
    VkCommandBuffer *commandBuffer;
};

extern "C"
{
    void *vkfft_app_create()
    {
        return new (std::nothrow) VkFFTApplication{};
    }

    void vkfft_app_destroy(void *app)
    {
        delete reinterpret_cast<VkFFTApplication *>(app);
    }

    VkFFTResult vkfft_init(void *app, const FFIVkFFTConfiguration *config_in) noexcept
    {
        VkFFTConfiguration config = {};

        config.FFTdim = config_in->dimensions;
        memcpy(config.size, config_in->size, sizeof(config.size));

        config.physicalDevice = config_in->physicalDevice;
        config.device = config_in->device;
        config.queue = config_in->queue;
        config.commandPool = config_in->commandPool;
        config.fence = config_in->fence;
        config.isCompilerInitialized = config_in->isCompilerInitialized;

        config.bufferSize = config_in->bufferSize;
        config.kernelSize = config_in->kernelSize;

        config.buffer = config_in->buffer;
        config.kernel = config_in->kernel;

        config.normalize = config_in->normalize;
        config.performConvolution = config_in->performConvolution;
        config.kernelConvolution = config_in->kernelConvolution;
        config.numberBatches = config_in->numberBatches;

        auto* application = reinterpret_cast<VkFFTApplication *>(app);
        return vkfft_ffi::invoke(
            [&] { return initializeVkFFT(application, config); },
            [&]() noexcept { deleteVkFFT(application); },
            VKFFT_ERROR_FAILED_TO_INITIALIZE);
    }

    VkFFTResult vkfft_exec(void *app, int inverse, const FFIVkFFTLaunchParams *launch_in) noexcept
    {
        VkFFTLaunchParams launch_params = {};

        launch_params.commandBuffer = launch_in->commandBuffer;

        // The caller owns the plan and will discard failed command recording.
        return vkfft_ffi::invoke(
            [&] { return VkFFTAppend(reinterpret_cast<VkFFTApplication *>(app), inverse, &launch_params); },
            []() noexcept {}, VKFFT_ERROR_FAILED_TO_LAUNCH_KERNEL);
    }

    void vkfft_delete_plan(void *app)
    {
        deleteVkFFT(reinterpret_cast<VkFFTApplication *>(app));
    }
}
