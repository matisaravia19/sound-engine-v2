// Scoped resources used by the Vulkan branch of VkFFT's compiler.
#ifndef VKFFT_RS_VULKAN_COMPILER_HPP
#define VKFFT_RS_VULKAN_COMPILER_HPP

#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include <memory>

namespace vkfft_compiler {

using Shader = std::unique_ptr<glslang_shader_t, decltype(&glslang_shader_delete)>;
using Program = std::unique_ptr<glslang_program_t, decltype(&glslang_program_delete)>;

// Compilation owns SPIR-V until a successful plan explicitly retains its binary.
struct Binary {
    uint32_t* data = nullptr;
    Binary() = default;
    Binary(const Binary&) = delete;
    Binary& operator=(const Binary&) = delete;
    ~Binary() {
        free(data);
    }
    uint32_t* release() {
        auto* result = data;
        data = nullptr;
        return result;
    }
};

// Shader modules are temporary even when a pipeline layout or pipeline fails.
struct ShaderModule {
    VkDevice device;
    VkShaderModule handle = VK_NULL_HANDLE;
    explicit ShaderModule(VkDevice device) : device(device) {}
    ShaderModule(const ShaderModule&) = delete;
    ShaderModule& operator=(const ShaderModule&) = delete;
    ~ShaderModule() {
        if (handle)
            vkDestroyShaderModule(device, handle, nullptr);
    }
};

} // namespace vkfft_compiler
#endif
