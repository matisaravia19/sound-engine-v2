// Included after the fake Vulkan API so tests execute the actual patched generator.
// Nested compilation/FFT execution and transfers are simulated; ownership is real.
#include <unordered_set>
#include <cmath>

namespace recursive_fake {
enum Failure { None, Initialize, Upload, Record, Readback };
struct State {
    Failure failure = None;
    int applications = 0;
    bool invalid_cleanup = false;
    std::unordered_set<void*> allocations;
};
thread_local State state;

void* allocate(size_t size) {
    void* pointer = malloc(size);
    if (pointer)
        state.allocations.insert(pointer);
    return pointer;
}
void release(void* pointer) {
    if (!pointer)
        return;
    if (state.allocations.erase(pointer) != 1)
        state.invalid_cleanup = true;
    else
        free(pointer);
}
VkFFTResult upload(VkFFTApplication*, void*, void*, pfUINT) {
    return state.failure == Upload ? VKFFT_ERROR_FAILED_TO_MAP_MEMORY : VKFFT_SUCCESS;
}
VkFFTResult readback(VkFFTApplication*, void*, void*, pfUINT) {
    return state.failure == Readback ? VKFFT_ERROR_FAILED_TO_MAP_MEMORY : VKFFT_SUCCESS;
}
} // namespace recursive_fake

static inline VkFFTResult initializeVkFFT(VkFFTApplication* app, VkFFTConfiguration config) {
    if (recursive_fake::state.failure == recursive_fake::Initialize)
        return VKFFT_ERROR_FAILED_TO_INITIALIZE;
    app->configuration = config;
    ++recursive_fake::state.applications;
    return VKFFT_SUCCESS;
}
static inline void deleteVkFFT(VkFFTApplication* app) {
    if (!app->configuration.device)
        return;
    if (fake::state.pending || fake::state.commands || --recursive_fake::state.applications < 0)
        recursive_fake::state.invalid_cleanup = true;
    app->configuration = {};
}
static inline VkFFTResult VkFFTAppend(VkFFTApplication*, int, VkFFTLaunchParams*) {
    return recursive_fake::state.failure == recursive_fake::Record
               ? VKFFT_ERROR_PLAN_NOT_INITIALIZED
               : VKFFT_SUCCESS;
}

#define VKFFT_ZERO_INIT                                                                            \
    {}
#define pfsin sin
#define pfcos cos
#define pfceil ceil
#define pffloor floor
#define pfsqrt sqrt
#define pfFPinit(x) strtold(x, 0)
#define VKFFT_INITIALIZEAPP_H
#define VKFFT_DELETEAPP_H
#define malloc recursive_fake::allocate
#define free recursive_fake::release
#include "vulkan/init.hpp"
#define VkFFT_TransferDataFromCPU recursive_fake::upload
#define VkFFT_TransferDataToCPU recursive_fake::readback
#include "vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_RecursiveFFTGenerators.h"
#undef malloc
#undef free

extern "C" int vkfft_test_recursive_failures() {
    using fake::handle;
    const fake::Failure failures[] = {
        fake::None,           fake::Create,         fake::Find, fake::Allocate, fake::Bind,
        fake::Command,        fake::Begin,          fake::End,  fake::Submit,   fake::WaitTimeout,
        fake::WaitHostMemory, fake::WaitDeviceLost, fake::Reset};
    for (auto local_failure :
         {recursive_fake::None, recursive_fake::Initialize, recursive_fake::Upload,
          recursive_fake::Record, recursive_fake::Readback}) {
        for (auto native_failure : failures) {
            if (local_failure != recursive_fake::None && native_failure != fake::None)
                continue;
            fake::state = {};
            fake::state.failure = native_failure;
            CHECK(recursive_fake::state.allocations.empty());
            recursive_fake::state = {};
            recursive_fake::state.failure = local_failure;
            VkDevice device = handle<VkDevice>(10);
            VkPhysicalDevice physical = handle<VkPhysicalDevice>(11);
            VkQueue queue = handle<VkQueue>(12);
            VkCommandPool pool = handle<VkCommandPool>(13);
            VkFence fence = handle<VkFence>(14);
            VkFFTApplication app{};
            app.configuration.device = &device;
            app.configuration.physicalDevice = &physical;
            app.configuration.queue = &queue;
            app.configuration.commandPool = &pool;
            app.configuration.fence = &fence;
            VkFFTRaderContainer container{};
            container.prime = 17;
            container.generator = 3;
            VkFFTAxis axis{};
            axis.specializationConstants.useRader = 1;
            axis.specializationConstants.numRaderPrimes = 1;
            axis.specializationConstants.raderContainer = &container;

            const VkFFTResult result = VkFFTGenerateRaderFFTKernel(&app, &axis);
            CHECK((result == VKFFT_SUCCESS) ==
                  (local_failure == recursive_fake::None && native_failure == fake::None));
            CHECK(!fake::state.pending && !fake::state.invalid_cleanup);
            CHECK(!fake::state.buffers && !fake::state.memories && !fake::state.commands);
            CHECK(!recursive_fake::state.applications && !recursive_fake::state.invalid_cleanup);
            // Error rollback still belongs to the parent, which must free the CPU kernel exactly once.
            CHECK(app.numRaderFFTPrimes == 1 && app.raderFFTkernel[0] == container.raderFFTkernel);
            CHECK(recursive_fake::state.allocations.size() == 1);
            recursive_fake::release(app.raderFFTkernel[0]);
            CHECK(recursive_fake::state.allocations.empty() &&
                  !recursive_fake::state.invalid_cleanup);
        }
    }
    return 0;
}
