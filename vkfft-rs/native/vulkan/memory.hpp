// Adapted from VkFFT's Vulkan memory helpers.
// Copyright (C) 2021 - present Dmitrii Tolmachev <dtolm96@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// Vulkan-only replacement for VkFFT's allocation and initialization transfers.
// Keep resources scoped on every error path; the upstream helpers leak locals
// and can return after a failed wait while their commands are still executing.
#ifndef VKFFT_MANAGEMEMORY_H
#define VKFFT_MANAGEMEMORY_H

#if VKFFT_BACKEND != 0
#error This override supports only the Vulkan backend
#endif

#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include <chrono>
#include <cstring>
#include <thread>

namespace vkfft_memory {

// Owns a buffer and its dedicated allocation until explicitly transferred.
struct Buffer {
    VkDevice device;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;

    explicit Buffer(VkDevice device) : device(device) {}
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() {
        if (buffer) vkDestroyBuffer(device, buffer, nullptr);
        if (memory) vkFreeMemory(device, memory, nullptr);
    }
};

// The caller completes submission before this scope releases the command.
struct Command {
    VkDevice device;
    VkCommandPool pool;
    VkCommandBuffer handle = VK_NULL_HANDLE;

    Command(VkDevice device, VkCommandPool pool) : device(device), pool(pool) {}
    Command(const Command&) = delete;
    Command& operator=(const Command&) = delete;
    ~Command() {
        if (handle) vkFreeCommandBuffers(device, pool, 1, &handle);
    }
};

// Timeout/host allocation failure does not permit destruction of pending work.
// As in the Rust submission path, retain everything until completion or device loss.
inline VkFFTResult wait_for_transfer(VkFFTApplication* app) {
    VkFFTResult first_error = VKFFT_SUCCESS;
    for (;;) {
        const VkResult result = vkWaitForFences(
            app->configuration.device[0], 1, app->configuration.fence, VK_TRUE, UINT64_MAX);
        if (result == VK_ERROR_DEVICE_LOST) return VKFFT_ERROR_FAILED_TO_WAIT_FOR_FENCES;
        if (result == VK_SUCCESS) break;
        first_error = VKFFT_ERROR_FAILED_TO_WAIT_FOR_FENCES;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const VkResult reset = vkResetFences(
        app->configuration.device[0], 1, app->configuration.fence);
    if (first_error != VKFFT_SUCCESS) return first_error;
    return reset == VK_SUCCESS ? VKFFT_SUCCESS : VKFFT_ERROR_FAILED_TO_RESET_FENCES;
}

} // namespace vkfft_memory

static inline VkFFTResult findMemoryType(
    VkFFTApplication* app, pfUINT bits, pfUINT size,
    VkMemoryPropertyFlags properties, uint32_t* index) {
    VkPhysicalDeviceMemoryProperties memory = {};
    vkGetPhysicalDeviceMemoryProperties(app->configuration.physicalDevice[0], &memory);
    for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
        const auto& type = memory.memoryTypes[i];
        if ((bits & (uint64_t(1) << i)) && (type.propertyFlags & properties) == properties
            && memory.memoryHeaps[type.heapIndex].size >= size) {
            *index = i;
            return VKFFT_SUCCESS;
        }
    }
    return VKFFT_ERROR_FAILED_TO_FIND_MEMORY;
}

// Failed allocation always leaves null outputs, including callers that provide
// uninitialized heap storage. Ownership transfers only after binding succeeds.
static inline VkFFTResult allocateBufferVulkan(
    VkFFTApplication* app, VkBuffer* buffer, VkDeviceMemory* memory,
    VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkDeviceSize size) {
    if (!buffer || !memory) return VKFFT_ERROR_MALLOC_FAILED;
    *buffer = VK_NULL_HANDLE;
    *memory = VK_NULL_HANDLE;
    vkfft_memory::Buffer owned(app->configuration.device[0]);
    VkBufferCreateInfo create = {};
    create.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    create.size = size;
    create.usage = usage;
    if (vkCreateBuffer(owned.device, &create, nullptr, &owned.buffer) != VK_SUCCESS) {
        owned.buffer = VK_NULL_HANDLE;
        return VKFFT_ERROR_FAILED_TO_CREATE_BUFFER;
    }
    VkMemoryRequirements requirements = {};
    vkGetBufferMemoryRequirements(owned.device, owned.buffer, &requirements);
    VkMemoryAllocateInfo allocate = {};
    allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate.allocationSize = requirements.size;
    const VkFFTResult found = findMemoryType(
        app, requirements.memoryTypeBits, requirements.size, properties, &allocate.memoryTypeIndex);
    if (found != VKFFT_SUCCESS) return found;
    if (vkAllocateMemory(owned.device, &allocate, nullptr, &owned.memory) != VK_SUCCESS) {
        owned.memory = VK_NULL_HANDLE;
        return VKFFT_ERROR_FAILED_TO_ALLOCATE_MEMORY;
    }
    if (vkBindBufferMemory(owned.device, owned.buffer, owned.memory, 0) != VK_SUCCESS)
        return VKFFT_ERROR_FAILED_TO_BIND_BUFFER_MEMORY;
    *buffer = owned.buffer;
    *memory = owned.memory;
    owned.buffer = VK_NULL_HANDLE;
    owned.memory = VK_NULL_HANDLE;
    return VKFFT_SUCCESS;
}

namespace vkfft_memory {

// Borrowed staging remains caller-owned; internal staging is scoped to this call.
inline VkFFTResult transfer(VkFFTApplication* app, void* cpu, void* gpu, pfUINT size, bool upload) {
    if (size == 0) return VKFFT_SUCCESS;
    if (!cpu || !gpu || !static_cast<VkBuffer*>(gpu)[0]) return VKFFT_ERROR_EMPTY_buffer;
    if (size > SIZE_MAX) return VKFFT_ERROR_INSUFFICIENT_TEMP_BUFFER;
    const VkDevice device = app->configuration.device[0];
    Buffer owned(device);
    VkBuffer staging;
    VkDeviceMemory memory;
    if (app->configuration.stagingBuffer) {
        if (!app->configuration.stagingBufferMemory) return VKFFT_ERROR_EMPTY_buffer;
        staging = app->configuration.stagingBuffer[0];
        memory = app->configuration.stagingBufferMemory[0];
        if (!staging || !memory) return VKFFT_ERROR_EMPTY_buffer;
    } else {
        const VkFFTResult result = allocateBufferVulkan(
            app, &owned.buffer, &owned.memory,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, size);
        if (result != VKFFT_SUCCESS) return result;
        staging = owned.buffer;
        memory = owned.memory;
    }
    if (upload) {
        void* mapped = nullptr;
        if (vkMapMemory(device, memory, 0, size, 0, &mapped) != VK_SUCCESS)
            return VKFFT_ERROR_FAILED_TO_MAP_MEMORY;
        std::memcpy(mapped, cpu, static_cast<size_t>(size));
        vkUnmapMemory(device, memory);
    }

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

    // LUT and recursive FFT transfers can follow compute work on this queue.
    VkMemoryBarrier before = {};
    before.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    before.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(command.handle, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
    const VkBuffer target = static_cast<VkBuffer*>(gpu)[0];
    VkBufferCopy copy = {};
    copy.size = size;
    vkCmdCopyBuffer(command.handle, upload ? staging : target, upload ? target : staging, 1, &copy);
    VkMemoryBarrier after = {};
    after.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    after.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(command.handle, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        0, 1, &after, 0, nullptr, 0, nullptr);
    if (vkEndCommandBuffer(command.handle) != VK_SUCCESS)
        return VKFFT_ERROR_FAILED_TO_END_COMMAND_BUFFER;
    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command.handle;
    if (vkQueueSubmit(app->configuration.queue[0], 1, &submit, app->configuration.fence[0]) != VK_SUCCESS)
        return VKFFT_ERROR_FAILED_TO_SUBMIT_QUEUE;
    const VkFFTResult waited = wait_for_transfer(app);
    if (waited != VKFFT_SUCCESS) return waited;

    if (!upload) {
        void* mapped = nullptr;
        if (vkMapMemory(device, memory, 0, size, 0, &mapped) != VK_SUCCESS)
            return VKFFT_ERROR_FAILED_TO_MAP_MEMORY;
        std::memcpy(cpu, mapped, static_cast<size_t>(size));
        vkUnmapMemory(device, memory);
    }
    return VKFFT_SUCCESS;
}

} // namespace vkfft_memory

static inline VkFFTResult VkFFT_TransferDataFromCPU(
    VkFFTApplication* app, void* cpu, void* gpu, pfUINT size) {
    return vkfft_memory::transfer(app, cpu, gpu, size, true);
}

static inline VkFFTResult VkFFT_TransferDataToCPU(
    VkFFTApplication* app, void* cpu, void* gpu, pfUINT size) {
    return vkfft_memory::transfer(app, cpu, gpu, size, false);
}

#endif
