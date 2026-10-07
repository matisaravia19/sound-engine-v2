// CPU-only failure injection. These functions never call the Vulkan loader.
#include <array>
#include <cstdint>
#include <cstring>
#include <type_traits>

#define VKFFT_BACKEND 0
#define VKFFT_MAX_FFT_DIMENSIONS 4
#define pfUINT uint64_t
#define pfINT int64_t
#define pfLD long double
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"

namespace fake {

enum Failure { None, Create, Find, Allocate, Bind, Map, Command, Begin, End, Submit,
    WaitTimeout, WaitHostMemory, WaitDeviceLost, Reset };

struct State {
    Failure failure = None;
    int buffers = 0;
    int memories = 0;
    int commands = 0;
    int mapped = 0;
    int waits = 0;
    int barriers = 0;
    bool pending = false;
    bool invalid_cleanup = false;
    VkBuffer source = VK_NULL_HANDLE;
    VkBuffer destination = VK_NULL_HANDLE;
    std::array<unsigned char, 16> staging{};
    std::array<unsigned char, 16> target{};
};

thread_local State state;
template<class T> T handle(uintptr_t value) {
    if constexpr (std::is_pointer_v<T>) return reinterpret_cast<T>(value);
    else return static_cast<T>(value);
}

VkResult create_buffer(VkDevice, const VkBufferCreateInfo* info, const VkAllocationCallbacks*, VkBuffer* out) {
    if (info->queueFamilyIndexCount != 0 || info->pQueueFamilyIndices != nullptr)
        state.invalid_cleanup = true;
    *out = handle<VkBuffer>(1);
    if (state.failure == Create) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    ++state.buffers;
    return VK_SUCCESS;
}
void destroy_buffer(VkDevice, VkBuffer buffer, const VkAllocationCallbacks*) {
    if (state.pending || buffer != handle<VkBuffer>(1) || --state.buffers < 0)
        state.invalid_cleanup = true;
}
void requirements(VkDevice, VkBuffer, VkMemoryRequirements* out) {
    *out = {};
    out->size = 16;
    out->memoryTypeBits = 1;
}
void memory_properties(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties* out) {
    *out = {};
    out->memoryTypeCount = 1;
    out->memoryHeapCount = 1;
    out->memoryHeaps[0].size = 1024;
    out->memoryTypes[0].propertyFlags = state.failure == Find ? 0 :
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
}
VkResult allocate_memory(VkDevice, const VkMemoryAllocateInfo*, const VkAllocationCallbacks*, VkDeviceMemory* out) {
    *out = handle<VkDeviceMemory>(2);
    if (state.failure == Allocate) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    ++state.memories;
    return VK_SUCCESS;
}
void free_memory(VkDevice, VkDeviceMemory memory, const VkAllocationCallbacks*) {
    if (state.pending || state.mapped || state.buffers || memory != handle<VkDeviceMemory>(2)
        || --state.memories < 0) state.invalid_cleanup = true;
}
VkResult bind(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) {
    return state.failure == Bind ? VK_ERROR_OUT_OF_DEVICE_MEMORY : VK_SUCCESS;
}
VkResult map(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkMemoryMapFlags, void** out) {
    *out = reinterpret_cast<void*>(1); // Failure must not dereference or unmap this output.
    if (state.failure == Map) return VK_ERROR_MEMORY_MAP_FAILED;
    *out = state.staging.data();
    ++state.mapped;
    return VK_SUCCESS;
}
void unmap(VkDevice, VkDeviceMemory) {
    if (--state.mapped < 0) state.invalid_cleanup = true;
}
VkResult allocate_command(VkDevice, const VkCommandBufferAllocateInfo*, VkCommandBuffer* out) {
    *out = handle<VkCommandBuffer>(3);
    if (state.failure == Command) return VK_ERROR_OUT_OF_HOST_MEMORY;
    ++state.commands;
    return VK_SUCCESS;
}
void free_command(VkDevice, VkCommandPool, uint32_t, const VkCommandBuffer* command) {
    if (state.pending || *command != handle<VkCommandBuffer>(3) || --state.commands < 0)
        state.invalid_cleanup = true;
}
VkResult begin(VkCommandBuffer, const VkCommandBufferBeginInfo*) {
    return state.failure == Begin ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_SUCCESS;
}
VkResult end(VkCommandBuffer) {
    return state.failure == End ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_SUCCESS;
}
void barrier(VkCommandBuffer, VkPipelineStageFlags source, VkPipelineStageFlags destination,
    VkDependencyFlags, uint32_t count, const VkMemoryBarrier* memory,
    uint32_t, const VkBufferMemoryBarrier*, uint32_t, const VkImageMemoryBarrier*) {
    ++state.barriers;
    if (count != 1 || !memory->srcAccessMask || !memory->dstAccessMask || !source || !destination)
        state.invalid_cleanup = true;
}
void copy(VkCommandBuffer, VkBuffer source, VkBuffer destination, uint32_t count, const VkBufferCopy* region) {
    if (count != 1 || region->size != 16 || region->srcOffset || region->dstOffset)
        state.invalid_cleanup = true;
    state.source = source;
    state.destination = destination;
}
VkResult submit(VkQueue, uint32_t, const VkSubmitInfo*, VkFence) {
    if (state.failure == Submit) return VK_ERROR_OUT_OF_HOST_MEMORY;
    state.pending = true;
    if (state.destination == handle<VkBuffer>(100)) state.target = state.staging;
    else state.staging = state.target;
    return VK_SUCCESS;
}
VkResult wait(VkDevice, uint32_t, const VkFence*, VkBool32, uint64_t timeout) {
    if (timeout != UINT64_MAX) state.invalid_cleanup = true;
    ++state.waits;
    if (state.waits == 1 && state.failure == WaitTimeout) return VK_TIMEOUT;
    if (state.waits == 1 && state.failure == WaitHostMemory) return VK_ERROR_OUT_OF_HOST_MEMORY;
    state.pending = false;
    return state.failure == WaitDeviceLost ? VK_ERROR_DEVICE_LOST : VK_SUCCESS;
}
VkResult reset(VkDevice, uint32_t, const VkFence*) {
    if (state.pending) state.invalid_cleanup = true;
    return state.failure == Reset ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_SUCCESS;
}

} // namespace fake

#define vkfft_memory vkfft_test_memory
#define vkCreateBuffer fake::create_buffer
#define vkDestroyBuffer fake::destroy_buffer
#define vkGetBufferMemoryRequirements fake::requirements
#define vkGetPhysicalDeviceMemoryProperties fake::memory_properties
#define vkAllocateMemory fake::allocate_memory
#define vkFreeMemory fake::free_memory
#define vkBindBufferMemory fake::bind
#define vkMapMemory fake::map
#define vkUnmapMemory fake::unmap
#define vkAllocateCommandBuffers fake::allocate_command
#define vkFreeCommandBuffers fake::free_command
#define vkBeginCommandBuffer fake::begin
#define vkEndCommandBuffer fake::end
#define vkCmdPipelineBarrier fake::barrier
#define vkCmdCopyBuffer fake::copy
#define vkQueueSubmit fake::submit
#define vkWaitForFences fake::wait
#define vkResetFences fake::reset
#include "vulkan/memory.hpp"

// Return a source line on failure; ordinary assert is disabled in release builds.
#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

extern "C" int vkfft_test_memory_failures() {
    using namespace fake;
    const VkFFTResult expected[] = {
        VKFFT_SUCCESS, VKFFT_ERROR_FAILED_TO_CREATE_BUFFER, VKFFT_ERROR_FAILED_TO_FIND_MEMORY,
        VKFFT_ERROR_FAILED_TO_ALLOCATE_MEMORY, VKFFT_ERROR_FAILED_TO_BIND_BUFFER_MEMORY,
        VKFFT_ERROR_FAILED_TO_MAP_MEMORY, VKFFT_ERROR_FAILED_TO_ALLOCATE_COMMAND_BUFFERS,
        VKFFT_ERROR_FAILED_TO_BEGIN_COMMAND_BUFFER, VKFFT_ERROR_FAILED_TO_END_COMMAND_BUFFER,
        VKFFT_ERROR_FAILED_TO_SUBMIT_QUEUE, VKFFT_ERROR_FAILED_TO_WAIT_FOR_FENCES,
        VKFFT_ERROR_FAILED_TO_WAIT_FOR_FENCES, VKFFT_ERROR_FAILED_TO_WAIT_FOR_FENCES,
        VKFFT_ERROR_FAILED_TO_RESET_FENCES
    };
    VkDevice device = handle<VkDevice>(10);
    VkPhysicalDevice physical = handle<VkPhysicalDevice>(11);
    VkQueue queue = handle<VkQueue>(12);
    VkCommandPool pool = handle<VkCommandPool>(13);
    VkFence fence = handle<VkFence>(14);
    VkBuffer target = handle<VkBuffer>(100);
    VkBuffer borrowed_staging = handle<VkBuffer>(101);
    VkDeviceMemory borrowed_memory = handle<VkDeviceMemory>(102);
    VkFFTApplication app{};
    app.configuration.device = &device;
    app.configuration.physicalDevice = &physical;
    app.configuration.queue = &queue;
    app.configuration.commandPool = &pool;
    app.configuration.fence = &fence;

    for (bool upload : {false, true}) {
        for (bool borrowed : {false, true}) {
            app.configuration.stagingBuffer = borrowed ? &borrowed_staging : nullptr;
            app.configuration.stagingBufferMemory = borrowed ? &borrowed_memory : nullptr;
            for (int failure = None; failure <= Reset; ++failure) {
                if (borrowed && failure >= Create && failure <= Bind) continue;
                state = {};
                state.failure = static_cast<Failure>(failure);
                state.target.fill(7);
                std::array<unsigned char, 16> cpu;
                cpu.fill(42);
                const VkFFTResult result = upload
                    ? VkFFT_TransferDataFromCPU(&app, cpu.data(), &target, cpu.size())
                    : VkFFT_TransferDataToCPU(&app, cpu.data(), &target, cpu.size());
                CHECK(result == expected[failure]);
                CHECK(!state.pending && !state.invalid_cleanup);
                CHECK(state.buffers == 0 && state.memories == 0 && state.commands == 0 && state.mapped == 0);
                if (failure == WaitTimeout || failure == WaitHostMemory) CHECK(state.waits == 2);
                if (failure == None) {
                    CHECK(state.barriers == 2);
                    CHECK(state.source == (upload ? (borrowed ? borrowed_staging : handle<VkBuffer>(1)) : target));
                    CHECK(state.destination == (upload ? target : (borrowed ? borrowed_staging : handle<VkBuffer>(1))));
                    for (auto value : (upload ? state.target : cpu)) CHECK(value == (upload ? 42 : 7));
                } else if (!upload) {
                    for (auto value : cpu) CHECK(value == 42);
                }
            }
        }
    }
    // Failed native allocations must not leave handles for upstream cleanup to destroy twice.
    app.configuration.stagingBuffer = nullptr;
    app.configuration.stagingBufferMemory = nullptr;
    for (int failure = Create; failure <= Bind; ++failure) {
        state = {};
        state.failure = static_cast<Failure>(failure);
        VkBuffer buffer = handle<VkBuffer>(999);
        VkDeviceMemory memory = handle<VkDeviceMemory>(999);
        CHECK(allocateBufferVulkan(&app, &buffer, &memory, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 16) == expected[failure]);
        CHECK(!buffer && !memory && !state.invalid_cleanup && !state.buffers && !state.memories);
    }
    return 0;
}

#include "recursive_init_tests.hpp"
