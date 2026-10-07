#include <stdint.h>
#include <stdlib.h>
#include <new>

#define VKFFT_BACKEND 0
// Load the scoped Vulkan helpers before VkFFT's include guard selects its own.
// These scalar aliases match the default precision configuration in vkFFT.h.
#define VKFFT_MAX_FFT_DIMENSIONS 4
#define pfUINT uint64_t
#define pfINT int64_t
#define pfLD long double
#include "vulkan/memory.hpp"
#include "vulkan/init.hpp"
#include "vulkan/compiler.hpp"
#include "ffi_guard.hpp"

#include "vkFFT.h"

// Must match the borrowed handle layout in src/ffi.rs.
struct FFIVkFFTConfiguration {
    uint64_t length;

    VkPhysicalDevice* physicalDevice;
    VkDevice* device;
    VkQueue* queue;
    VkCommandPool* commandPool;
    VkFence* fence;

    VkBuffer* buffer;
    uint64_t* bufferSize;

    uint64_t normalize;
};

struct FFIVkFFTLaunchParams {
    VkCommandBuffer* commandBuffer;
};

extern "C" {
    void* vkfft_app_create() {
        return new (std::nothrow) VkFFTApplication{};
    }

    void vkfft_app_destroy(void* app) {
        delete reinterpret_cast<VkFFTApplication*>(app);
    }

    int32_t vkfft_init(void* app, const FFIVkFFTConfiguration* config_in) noexcept {
        VkFFTConfiguration config = {};

        config.FFTdim = 1;
        config.size[0] = config_in->length;
        config.numberBatches = 1;

        config.physicalDevice = config_in->physicalDevice;
        config.device = config_in->device;
        config.queue = config_in->queue;
        config.commandPool = config_in->commandPool;
        config.fence = config_in->fence;

        config.bufferSize = config_in->bufferSize;

        config.buffer = config_in->buffer;

        config.normalize = config_in->normalize;

        auto* application = reinterpret_cast<VkFFTApplication*>(app);
        return vkfft_ffi::invoke([&] { return initializeVkFFT(application, config); },
                                 [&]() noexcept { deleteVkFFT(application); },
                                 VKFFT_ERROR_FAILED_TO_INITIALIZE);
    }

    int32_t vkfft_exec(void* app, int inverse, const FFIVkFFTLaunchParams* launch_in) noexcept {
        VkFFTLaunchParams launch_params = {};

        launch_params.commandBuffer = launch_in->commandBuffer;

        // The caller owns the plan and will discard failed command recording.
        return vkfft_ffi::invoke(
            [&] {
                return VkFFTAppend(reinterpret_cast<VkFFTApplication*>(app), inverse,
                                   &launch_params);
            },
            []() noexcept {}, VKFFT_ERROR_FAILED_TO_LAUNCH_KERNEL);
    }

    void vkfft_delete_plan(void* app) {
        deleteVkFFT(reinterpret_cast<VkFFTApplication*>(app));
    }
}
