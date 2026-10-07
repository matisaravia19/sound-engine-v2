// CPU-only tests of the patched compiler and axis cleanup; no Vulkan loader calls.
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>

#define VKFFT_BACKEND 0
#define VKFFT_MAX_FFT_DIMENSIONS 4
#define VKFFT_ZERO_INIT {}
#define pfUINT uint64_t
#define pfINT int64_t
#define pfLD long double
#define pfsin sin
#define pfcos cos
#define pfceil ceil
#define pffloor floor
#define pfsqrt sqrt
#define pfFPinit(x) strtold(x, 0)
#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"

namespace compiler_fake {

enum Failure { None, Shader, Preprocess, Parse, Program, Link, SpirvMessage,
    SpirvEmpty, SpirvNull, SpirvOverflow, Allocate, Module, Layout, Pipeline,
    PartialPipeline, ThrowAllocation, ThrowUnexpected };

struct State {
    Failure failure = None;
    int shaders = 0;
    int programs = 0;
    int modules = 0;
    int layouts = 0;
    int pipelines = 0;
    bool invalid = false;
    std::array<void*, 4> allocations{};
};
thread_local State state;
uint32_t spirv[] = {0x07230203, 0x00010000, 0, 1, 0};

template<class T> T handle(uintptr_t value) {
    if constexpr (std::is_pointer_v<T>) return reinterpret_cast<T>(value);
    else return static_cast<T>(value);
}
void* allocate(size_t size) {
    if (state.failure == Allocate) return nullptr;
    for (auto& allocation : state.allocations) {
        if (!allocation) return allocation = std::malloc(size);
    }
    state.invalid = true;
    return nullptr;
}
void release(void* pointer) {
    if (!pointer) return;
    for (auto& allocation : state.allocations) {
        if (allocation == pointer) {
            std::free(pointer);
            allocation = nullptr;
            return;
        }
    }
    state.invalid = true; // Do not perform a real double free in the test process.
}
int allocations() {
    int count = 0;
    for (auto* allocation : state.allocations) count += allocation != nullptr;
    return count;
}
glslang_shader_t* shader_create(const glslang_input_t*) {
    if (state.failure == Shader) return nullptr;
    ++state.shaders;
    return handle<glslang_shader_t*>(1);
}
void shader_delete(glslang_shader_t*) {
    if (state.programs || --state.shaders < 0) state.invalid = true;
}
int preprocess(glslang_shader_t*, const glslang_input_t*) { return state.failure != Preprocess; }
int parse(glslang_shader_t*, const glslang_input_t*) { return state.failure != Parse; }
const char* shader_log(glslang_shader_t*) { return "simulated shader failure"; }
glslang_program_t* program_create() {
    if (state.failure == Program) return nullptr;
    ++state.programs;
    return handle<glslang_program_t*>(2);
}
void program_delete(glslang_program_t*) { if (--state.programs < 0) state.invalid = true; }
void add_shader(glslang_program_t*, glslang_shader_t*) {}
extern "C" int vkfft_test_glslang_link(glslang_program_t*, int) {
    if (state.failure == ThrowAllocation) throw std::bad_alloc();
    return state.failure != Link;
}
const char* program_log(glslang_program_t*) { return "simulated program failure"; }
void generate(glslang_program_t*, glslang_stage_t) {}
const char* messages(glslang_program_t*) {
    return state.failure == SpirvMessage ? "simulated SPIR-V failure\n" : nullptr;
}
uint32_t* spirv_pointer(glslang_program_t*) { return state.failure == SpirvNull ? nullptr : spirv; }
size_t spirv_size(glslang_program_t*) {
    if (state.failure == SpirvEmpty) return 0;
    if (state.failure == SpirvOverflow) return SIZE_MAX;
    return sizeof(spirv) / sizeof(uint32_t);
}
VkResult create_module(VkDevice, const VkShaderModuleCreateInfo* info,
    const VkAllocationCallbacks*, VkShaderModule* out) {
    *out = handle<VkShaderModule>(3);
    if (state.failure == Module) return VK_ERROR_OUT_OF_HOST_MEMORY;
    if (info->codeSize != sizeof(spirv) || std::memcmp(info->pCode, spirv, sizeof(spirv)))
        state.invalid = true;
    ++state.modules;
    return VK_SUCCESS;
}
void destroy_module(VkDevice, VkShaderModule, const VkAllocationCallbacks*) {
    if (--state.modules < 0) state.invalid = true;
}
VkResult create_layout(VkDevice, const VkPipelineLayoutCreateInfo*,
    const VkAllocationCallbacks*, VkPipelineLayout* out) {
    *out = handle<VkPipelineLayout>(4);
    if (state.failure == Layout) return VK_ERROR_OUT_OF_HOST_MEMORY;
    ++state.layouts;
    return VK_SUCCESS;
}
void destroy_layout(VkDevice, VkPipelineLayout, const VkAllocationCallbacks*) {
    if (--state.layouts < 0) state.invalid = true;
}
extern "C" VkResult vkfft_test_create_pipeline(VkDevice, VkPipelineCache, uint32_t,
    const VkComputePipelineCreateInfo*, const VkAllocationCallbacks*, VkPipeline* out) {
    if (state.failure == ThrowUnexpected) throw 17;
    *out = VK_NULL_HANDLE;
    if (state.failure == Pipeline) return VK_ERROR_OUT_OF_HOST_MEMORY;
    ++state.pipelines;
    *out = handle<VkPipeline>(5);
    return state.failure == PartialPipeline ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_SUCCESS;
}
void destroy_pipeline(VkDevice, VkPipeline, const VkAllocationCallbacks*) {
    if (--state.pipelines < 0) state.invalid = true;
}
// These resources are absent from the minimal axis fixture.
void destroy_buffer(VkDevice, VkBuffer, const VkAllocationCallbacks*) { state.invalid = true; }
void free_memory(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*) { state.invalid = true; }
void destroy_pool(VkDevice, VkDescriptorPool, const VkAllocationCallbacks*) { state.invalid = true; }
void destroy_set_layout(VkDevice, VkDescriptorSetLayout, const VkAllocationCallbacks*) { state.invalid = true; }

} // namespace compiler_fake

// Distinct owner types avoid mixing mock and production destructors in one binary.
#define vkfft_compiler vkfft_test_compiler
#define malloc compiler_fake::allocate
#define free compiler_fake::release
#define glslang_shader_create compiler_fake::shader_create
#define glslang_shader_delete compiler_fake::shader_delete
#define glslang_shader_preprocess compiler_fake::preprocess
#define glslang_shader_parse compiler_fake::parse
#define glslang_shader_get_info_log compiler_fake::shader_log
#define glslang_program_create compiler_fake::program_create
#define glslang_program_delete compiler_fake::program_delete
#define glslang_program_add_shader compiler_fake::add_shader
#define glslang_program_link compiler_fake::vkfft_test_glslang_link
#define glslang_program_get_info_log compiler_fake::program_log
#define glslang_program_SPIRV_generate compiler_fake::generate
#define glslang_program_SPIRV_get_messages compiler_fake::messages
#define glslang_program_SPIRV_get_ptr compiler_fake::spirv_pointer
#define glslang_program_SPIRV_get_size compiler_fake::spirv_size
#define vkCreateShaderModule compiler_fake::create_module
#define vkDestroyShaderModule compiler_fake::destroy_module
#define vkCreatePipelineLayout compiler_fake::create_layout
#define vkDestroyPipelineLayout compiler_fake::destroy_layout
#define vkCreateComputePipelines compiler_fake::vkfft_test_create_pipeline
#define vkDestroyPipeline compiler_fake::destroy_pipeline
#define vkDestroyBuffer compiler_fake::destroy_buffer
#define vkFreeMemory compiler_fake::free_memory
#define vkDestroyDescriptorPool compiler_fake::destroy_pool
#define vkDestroyDescriptorSetLayout compiler_fake::destroy_set_layout
#include "vulkan/compiler.hpp"
#include "ffi_guard.hpp"
#include "vkFFT/vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_CompileKernel.h"
#include "vkFFT/vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_DeletePlan.h"

namespace compiler_fake {

int check_case(Failure failure, bool save, bool keep, int serialized) {
    state = {};
    VkDevice device = handle<VkDevice>(6);
    VkFFTApplication app{};
    VkFFTAxis axis{};
    app.configuration.device = &device;
    app.configuration.saveApplicationToString = save;
    app.configuration.keepShaderCode = keep;
    axis.specializationConstants.code0 = static_cast<char*>(allocate(32));
    if (!axis.specializationConstants.code0) return __LINE__;
    std::strcpy(axis.specializationConstants.code0, "simulated GLSL");

    // Serialized cases cover valid input and length checks before pointer arithmetic.
    std::array<unsigned char, sizeof(pfUINT) + sizeof(spirv)> blob{};
    pfUINT bytes = sizeof(spirv);
    if (serialized) {
        app.configuration.loadApplicationFromString = 1;
        app.configuration.loadApplicationString = blob.data();
        app.applicationStringSize = blob.size();
        if (serialized == 2) app.currentApplicationStringPos = blob.size() + 1;
        if (serialized == 3) app.applicationStringSize = sizeof(pfUINT) - 1;
        if (serialized == 4) bytes = 0;
        if (serialized == 5) bytes = 3;
        if (serialized == 6) bytes = UINT64_MAX - 3;
        if (serialized == 7) bytes = sizeof(spirv) + sizeof(uint32_t);
        std::memcpy(blob.data(), &bytes, sizeof(bytes));
        std::memcpy(blob.data() + sizeof(bytes), spirv, sizeof(spirv));
    }
    state.failure = failure;
    int rollbacks = 0;
    auto result = vkfft_ffi::invoke([&] { return VkFFT_CompileKernel(&app, &axis); },
        [&]() noexcept {
            ++rollbacks;
            if (state.shaders || state.programs || state.modules) state.invalid = true;
            deleteAxis(&app, &axis, 0);
        }, VKFFT_ERROR_FAILED_TO_INITIALIZE);

    if (state.shaders || state.programs || state.modules || state.invalid) return __LINE__;
    if (failure == ThrowAllocation && result != VKFFT_ERROR_MALLOC_FAILED) return __LINE__;
    if (failure == ThrowUnexpected && result != VKFFT_ERROR_FAILED_TO_INITIALIZE) return __LINE__;
    if (rollbacks != int(failure == ThrowAllocation || failure == ThrowUnexpected)) return __LINE__;
    const bool success = failure == None && serialized <= 1;
    if ((result == VKFFT_SUCCESS) != success) return __LINE__;
    if (success) {
        if (allocations() != 1 + int(save) || bool(axis.binary) != save) return __LINE__;
        if (axis.binarySize != sizeof(spirv)) return __LINE__;
        if (serialized && app.currentApplicationStringPos != blob.size()) return __LINE__;
        // Match the plan caller: source is released immediately unless retained.
        if (!keep) {
            release(axis.specializationConstants.code0);
            axis.specializationConstants.code0 = nullptr;
        }
    }
    deleteAxis(&app, &axis, 0);
    deleteAxis(&app, &axis, 0); // Cleanup remains safe after an exception rollback.
    if (allocations() || state.layouts || state.pipelines || state.invalid) return __LINE__;
    return 0;
}

} // namespace compiler_fake

extern "C" int vkfft_test_compiler_failures() {
    using namespace compiler_fake;
    for (bool save : {false, true}) {
        for (bool keep : {false, true}) {
            for (int failure = None; failure <= ThrowUnexpected; ++failure) {
                if (int line = check_case(static_cast<Failure>(failure), save, keep, 0)) return line;
            }
            for (int serialized = 1; serialized <= 7; ++serialized) {
                if (int line = check_case(None, save, keep, serialized)) return line;
            }
        }
    }
    return 0;
}
