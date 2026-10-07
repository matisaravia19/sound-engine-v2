// Temporary ownership used by the patched recursive VkFFT initialization paths.
#ifndef VKFFT_RS_VULKAN_INIT_HPP
#define VKFFT_RS_VULKAN_INIT_HPP

#include "vulkan/memory.hpp"

static inline void deleteVkFFT(VkFFTApplication* app);

namespace vkfft_memory {

// Nested plans borrow their parent's device/compiler and release only their own resources.
struct Application {
    VkFFTApplication value{};
    Application() = default;
    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;
    ~Application() {
        deleteVkFFT(&value);
    }
};

// Phase-vector storage is local; cached Rader kernels remain owned by the parent application.
struct HeapBuffer {
    void* data;
    explicit HeapBuffer(size_t size) : data(malloc(size)) {}
    HeapBuffer(const HeapBuffer&) = delete;
    HeapBuffer& operator=(const HeapBuffer&) = delete;
    ~HeapBuffer() {
        free(data);
    }
};

// Complete recursive FFT work before any temporary plan or buffer can unwind.
template <class Record> VkFFTResult run_commands(VkFFTApplication* app, Record record) {
    const VkDevice device = app->configuration.device[0];
    Command command(device, app->configuration.commandPool[0]);
    VkCommandBufferAllocateInfo allocate = {};
    allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocate.commandPool = command.pool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device, &allocate, &command.handle) != VK_SUCCESS) {
        command.handle = VK_NULL_HANDLE;
        return VKFFT_ERROR_FAILED_TO_ALLOCATE_COMMAND_BUFFERS;
    }
    VkCommandBufferBeginInfo begin = {};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(command.handle, &begin) != VK_SUCCESS)
        return VKFFT_ERROR_FAILED_TO_BEGIN_COMMAND_BUFFER;

    VkMemoryBarrier before = {};
    before.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    before.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(command.handle, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &before, 0, nullptr, 0,
                         nullptr);
    const VkFFTResult recorded = record(command.handle);
    if (recorded != VKFFT_SUCCESS)
        return recorded;
    VkMemoryBarrier after = {};
    after.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    after.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(command.handle, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
    if (vkEndCommandBuffer(command.handle) != VK_SUCCESS)
        return VKFFT_ERROR_FAILED_TO_END_COMMAND_BUFFER;
    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command.handle;
    if (vkQueueSubmit(app->configuration.queue[0], 1, &submit, app->configuration.fence[0]) !=
        VK_SUCCESS)
        return VKFFT_ERROR_FAILED_TO_SUBMIT_QUEUE;
    return wait_for_transfer(app);
}

} // namespace vkfft_memory
#endif
