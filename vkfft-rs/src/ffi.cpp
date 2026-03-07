#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define VKFFT_BACKEND 0

#include "vkfft.h"

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
};

struct FFIVkFFTLaunchParams
{
    VkCommandBuffer *commandBuffer;
};

extern "C"
{
    void *vkfft_app_create()
    {
        return new VkFFTApplication{};
    }

    void vkfft_app_destroy(void *app)
    {
        delete reinterpret_cast<VkFFTApplication *>(app);
    }

    VkFFTResult vkfft_init(void *app, const FFIVkFFTConfiguration *config_in)
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

        return initializeVkFFT(reinterpret_cast<VkFFTApplication *>(app), config);
    }

    VkFFTResult vkfft_exec(void *app, int inverse, const FFIVkFFTLaunchParams *launch_in)
    {
        VkFFTLaunchParams launch_params = {};

        launch_params.commandBuffer = launch_in->commandBuffer;

        return VkFFTAppend(reinterpret_cast<VkFFTApplication *>(app), inverse, &launch_params);
    }

    void vkfft_delete_plan(void *app)
    {
        deleteVkFFT(reinterpret_cast<VkFFTApplication *>(app));
    }
}