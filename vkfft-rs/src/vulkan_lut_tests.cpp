// CPU-only tests of the real LUT builders; no calls reach the Vulkan loader.
#include <array>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <memory>
#include <limits>
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

namespace lut_fake {
enum Failure { None, Host, Device, Transfer, ThrowTransfer };
struct Allocation { void* data = nullptr; size_t size = 0; };
struct State {
    std::array<Allocation, 16> allocations{};
    std::array<unsigned char, 65536> upload{};
    size_t uploaded = 0;
    int attempts = 0;
    int deleted = 0;
    int device_calls = 0;
    int transfer_calls = 0;
    int destroyed_buffers = 0;
    int freed_memories = 0;
    bool borrowed = false;
    Failure failure = None;
    bool invalid = false;
};
thread_local State state;

void* allocate(size_t size, bool zero) {
    ++state.attempts;
    if (state.failure == Host) return nullptr;
    if (size > state.upload.size()) { state.invalid = true; return nullptr; }
    for (auto& allocation : state.allocations) {
        if (allocation.data) continue;
        allocation = {std::malloc(size + 16), size};
        if (!allocation.data) return nullptr;
        std::memset(allocation.data, zero ? 0 : 0xa5, size);
        std::memset(static_cast<unsigned char*>(allocation.data) + size, 0x5a, 16);
        return allocation.data;
    }
    state.invalid = true;
    return nullptr;
}
void* allocate_raw(size_t size) { return allocate(size, false); }
void* allocate_zero(size_t count, size_t size) {
    if (size && count > SIZE_MAX / size) { state.invalid = true; return nullptr; }
    return allocate(count * size, true);
}
void release(void* pointer) {
    if (!pointer) return;
    for (auto& allocation : state.allocations) {
        if (allocation.data != pointer) continue;
        const auto* guard = static_cast<unsigned char*>(pointer) + allocation.size;
        for (int i = 0; i < 16; ++i) if (guard[i] != 0x5a) state.invalid = true;
        std::free(pointer);
        allocation = {};
        return;
    }
    state.invalid = true;
}
int live() {
    int count = 0;
    for (const auto& allocation : state.allocations) count += allocation.data != nullptr;
    return count;
}
template<class Handle> Handle handle(uintptr_t value) {
    if constexpr (std::is_pointer_v<Handle>) return reinterpret_cast<Handle>(value);
    else return static_cast<Handle>(value);
}
void destroy_buffer(VkDevice, VkBuffer, const VkAllocationCallbacks*) { ++state.destroyed_buffers; }
void free_memory(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*) { ++state.freed_memories; }
template<class Handle> void unexpected_destroy(VkDevice, Handle, const VkAllocationCallbacks*) {
    state.invalid = true;
}
} // namespace lut_fake

static inline void deleteVkFFT(VkFFTApplication*) { ++lut_fake::state.deleted; }
static inline VkFFTResult allocateBufferVulkan(VkFFTApplication*, VkBuffer* buffer,
        VkDeviceMemory* memory, VkBufferUsageFlags, VkMemoryPropertyFlags, VkDeviceSize) {
    auto& state = lut_fake::state;
    ++state.device_calls;
    if (state.failure == lut_fake::Device) return VKFFT_ERROR_FAILED_TO_ALLOCATE_MEMORY;
    *buffer = lut_fake::handle<VkBuffer>(1);
    *memory = lut_fake::handle<VkDeviceMemory>(2);
    return VKFFT_SUCCESS;
}
static inline VkFFTResult VkFFT_TransferDataFromCPU(VkFFTApplication*, void* source,
        VkBuffer*, uint64_t size) {
    auto& state = lut_fake::state;
    ++state.transfer_calls;
    if (state.failure == lut_fake::ThrowTransfer) throw 1;
    if (state.failure == lut_fake::Transfer) return VKFFT_ERROR_FAILED_TO_MAP_MEMORY;
    if (size > state.upload.size()) { state.invalid = true; return VKFFT_ERROR_MATH_FAILED; }
    state.uploaded = static_cast<size_t>(size);
    std::memcpy(state.upload.data(), source, state.uploaded);
    return VKFFT_SUCCESS;
}

#define VKFFT_MANAGEMEMORY_H
#define malloc lut_fake::allocate_raw
#define calloc lut_fake::allocate_zero
#define free lut_fake::release
#include "vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_ManageLUT.h"
#define vkDestroyBuffer lut_fake::destroy_buffer
#define vkFreeMemory lut_fake::free_memory
#define vkDestroyDescriptorPool lut_fake::unexpected_destroy
#define vkDestroyDescriptorSetLayout lut_fake::unexpected_destroy
#define vkDestroyPipelineLayout lut_fake::unexpected_destroy
#define vkDestroyPipeline lut_fake::unexpected_destroy
#include "vkFFT/vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_DeletePlan.h"
#undef vkDestroyBuffer
#undef vkFreeMemory
#undef vkDestroyDescriptorPool
#undef vkDestroyDescriptorSetLayout
#undef vkDestroyPipelineLayout
#undef vkDestroyPipeline
#undef malloc
#undef calloc
#undef free

namespace lut_fake {
// Separate plan storage keeps these large native structs off the test stack.
VkFFTResult run_case(int kind, int precision, bool shared, Failure failure, bool empty = false,
                    uint64_t size = 16, bool oversized_dct = false,
                    VkFFTResult prior = VKFFT_SUCCESS) {
    state = {};
    state.failure = failure;
    VkFFTApplication app{};
    auto plan = std::make_unique<VkFFTPlan>();
    auto inverse = std::make_unique<VkFFTPlan>();
    auto& axis = plan->axes[0][0];
    auto& sc = axis.specializationConstants;
    app.configuration.useLUT = 1;
    app.configuration.makeForwardPlanOnly = !shared;
    app.configuration.doublePrecision = precision == 1;
    app.configuration.doublePrecisionFloatMemory = precision == 2;
    app.configuration.quadDoubleDoublePrecision = precision == 3;
    app.configuration.quadDoubleDoublePrecisionDoubleMemory = precision == 4;
    app.configuration.size[0] = size;
    app.localFFTPlan_inverse = inverse.get();
    auto& borrowed = kind == 1 ? inverse->R2Cdecomposition : inverse->axes[0][0];
    borrowed.bufferLUT = handle<VkBuffer>(3);
    borrowed.bufferLUTDeviceMemory = handle<VkDeviceMemory>(4);
    borrowed.bufferLUTSize = 8 * (precision >= 3 ? 4 : 2) * (precision ? sizeof(double) : sizeof(float));
    sc.numStages = empty ? 1 : 2;
    sc.stageRadix[0] = 4;
    sc.stageRadix[1] = 4;
    sc.performDCT = oversized_dct ? 4 : 0;
    sc.res = prior;
    sc.numAxisUploads = 1;
    sc.registerBoost = 1;
    VkFFTRaderContainer rader[2]{};
    rader[0].prime = 17;
    rader[0].generator = 3;
    rader[1].prime = 19;
    rader[1].generator = 2;
    if (kind == 2) {
        sc.raderUintLUT = 1;
        sc.numRaderPrimes = 2;
        sc.raderContainer = rader;
    }
    borrowed.specializationConstants = sc;
    borrowed.specializationConstants.maxStageSumLUT = 8;
    const auto result = kind == 1 ? VkFFT_AllocateLUT_R2C(&app, plan.get(), &axis, 0)
        : kind == 2 ? VkFFT_AllocateRaderUintLUT(&app, &axis)
        : VkFFT_AllocateLUT(&app, plan.get(), &axis, 0);
    state.borrowed = axis.referenceLUT == 1 && axis.bufferLUT == borrowed.bufferLUT &&
        axis.bufferLUTDeviceMemory == borrowed.bufferLUTDeviceMemory &&
        axis.bufferLUTSize == borrowed.bufferLUTSize;
    return result;
}
} // namespace lut_fake

#define LUT_CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

extern "C" int vkfft_test_lut_ownership() {
    using namespace lut_fake;
    for (int kind = 0; kind < 3; ++kind) {
        for (int precision = 0; precision < 5; ++precision) {
            for (auto failure : {None, Host, Device, Transfer, ThrowTransfer}) {
                bool threw = false;
                VkFFTResult result = VKFFT_SUCCESS;
                try { result = run_case(kind, precision, false, failure); }
                catch (int) { threw = true; }
                const auto expected = failure == Host ? VKFFT_ERROR_MALLOC_FAILED
                    : failure == Device ? VKFFT_ERROR_FAILED_TO_ALLOCATE_MEMORY
                    : failure == Transfer ? VKFFT_ERROR_FAILED_TO_MAP_MEMORY : VKFFT_SUCCESS;
                LUT_CHECK(result == expected && threw == (failure == ThrowTransfer));
                LUT_CHECK(live() == 0 && !state.invalid);
                LUT_CHECK(state.attempts == 1);
                LUT_CHECK(state.deleted == (failure == Host || failure == Device || failure == Transfer));
            }
            if (kind == 2) continue;
            LUT_CHECK(run_case(kind, precision, false, None, false, 16, false,
                VKFFT_ERROR_INSUFFICIENT_CODE_BUFFER) == VKFFT_ERROR_INSUFFICIENT_CODE_BUFFER);
            LUT_CHECK(!state.attempts && !state.device_calls && !state.transfer_calls && state.deleted == 1);
            LUT_CHECK(run_case(kind, precision, true, None) == VKFFT_SUCCESS);
            LUT_CHECK(state.borrowed && !state.device_calls && !state.transfer_calls && !live() && !state.invalid);
            // Reuse must also work when no host allocation could succeed.
            LUT_CHECK(!state.attempts);
            LUT_CHECK(run_case(kind, precision, true, Host) == VKFFT_SUCCESS);
            LUT_CHECK(state.borrowed && !state.attempts && !live());
        }
    }
    return 0;
}

extern "C" int vkfft_test_lut_values_and_bounds() {
    using namespace lut_fake;
    const double pi = 3.1415926535897932384626433832795;
    for (int precision = 0; precision < 5; ++precision) {
        const size_t scalar_bytes = precision ? sizeof(double) : sizeof(float);
        const size_t lanes = precision >= 3 ? 4 : 2;
        LUT_CHECK(run_case(0, precision, false, None, true) == VKFFT_SUCCESS);
        LUT_CHECK(state.uploaded == scalar_bytes * lanes / 2 && !live() && !state.invalid);
        for (size_t i = 0; i < state.uploaded; ++i) LUT_CHECK(state.upload[i] == 0);
        for (int kind : {0, 1}) {
            LUT_CHECK(run_case(kind, precision, false, None) == VKFFT_SUCCESS);
            LUT_CHECK(state.uploaded == 8 * lanes * scalar_bytes && !live() && !state.invalid);
            for (size_t i = 0; i < 8; ++i) {
                const double angle = kind ? pi * i / 8 : pi * (i % 4) / 4 / (i < 4 ? 1 : 2);
                for (size_t component = 0; component < 2; ++component) {
                    const auto* input = state.upload.data() + scalar_bytes * (i * lanes + component * lanes / 2);
                    double value;
                    if (!precision) { float scalar; std::memcpy(&scalar, input, sizeof(scalar)); value = scalar; }
                    else {
                        std::memcpy(&value, input, sizeof(value));
                        if (lanes == 4) { double low; std::memcpy(&low, input + sizeof(double), sizeof(low)); value += low; }
                    }
                    const double expected = component ? std::sin(angle) : std::cos(angle);
                    LUT_CHECK(std::abs(value - expected) < (precision ? 1e-14 : 2e-7));
                }
            }
        }
        // Reject byte multiplication overflow before allocating or writing the table.
        LUT_CHECK(run_case(0, precision, false, None, true, UINT64_MAX - 1, true) == VKFFT_ERROR_MATH_FAILED);
        LUT_CHECK(!state.attempts && !state.device_calls && !state.transfer_calls && !state.invalid);
        for (uint64_t size : {uint64_t(0), uint64_t(1), UINT64_MAX}) {
            LUT_CHECK(run_case(1, precision, false, None, false, size) == VKFFT_ERROR_MATH_FAILED);
            LUT_CHECK(!state.attempts && !state.device_calls && !state.transfer_calls && !state.invalid);
        }
    }
    LUT_CHECK(run_case(2, 0, false, None) == VKFFT_SUCCESS);
    LUT_CHECK(state.uploaded == 36 * sizeof(uint32_t) && !live() && !state.invalid);
    size_t offset = 0;
    for (uint32_t prime : {17u, 19u}) {
        uint32_t expected = 1;
        for (uint32_t i = 0; i < prime; ++i) {
            uint32_t actual;
            std::memcpy(&actual, state.upload.data() + offset, sizeof(actual));
            LUT_CHECK(actual == expected);
            expected = expected * (prime == 17 ? 3 : 2) % prime;
            offset += sizeof(actual);
        }
    }
    return 0;
}

extern "C" int vkfft_test_double_double_conversion_failure() {
    using namespace lut_fake;
    for (int type : {22, 112}) {
        state = {};
        VkFFTSpecializationConstantsLayout sc{};
        PfContainer input{}, output{};
        input.type = type;
        input.data.d = 1.234567890123456789L;
        if (type > 100) {
            PfAllocateContainerFlexible(&sc, &input, 50);
            PfSetContainerName(&sc, &input, "input");
        }
        const int initial = live();
        state.failure = Host;
        const int attempts = state.attempts;
        PfConvToDoubleDouble(&sc, &output, &input);
        LUT_CHECK(sc.res == VKFFT_ERROR_MALLOC_FAILED && state.attempts == attempts + 1);
        LUT_CHECK(live() == initial && !state.invalid);
        PfDeallocateContainer(&sc, &output);
        PfDeallocateContainer(&sc, &input);
        LUT_CHECK(!live() && !state.invalid);
    }
    return 0;
}

#undef LUT_CHECK

#include "lut_layout_tests.hpp"
#include "lut_sharing_tests.hpp"
