// CPU-only allocation and bounds regressions against the actual patched headers.
#include <array>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>
#include <new>
#include <memory>

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

namespace codegen_fake {

struct State {
    std::array<void*, 4096> allocations{};
    int attempts = 0;
    int fail_at = 0;
    int throw_at = 0;
    bool invalid = false;
};
thread_local State state;

void* allocate(size_t count, size_t size) {
    ++state.attempts;
    if (state.attempts == state.throw_at) throw std::bad_alloc();
    if (state.attempts == state.fail_at) return nullptr;
    for (auto& allocation : state.allocations) {
        if (!allocation) return allocation = std::calloc(count, size);
    }
    state.invalid = true;
    return nullptr;
}
void release(void* pointer) {
    if (!pointer) return;
    for (auto& allocation : state.allocations) {
        if (allocation == pointer) {
            allocation = nullptr;
            std::free(pointer);
            return;
        }
    }
    state.invalid = true;
}
int live() {
    int count = 0;
    for (auto* allocation : state.allocations) count += allocation != nullptr;
    return count;
}

template<class Handle>
void unexpected_destroy(VkDevice, Handle, const VkAllocationCallbacks*) {
    state.invalid = true;
}

} // namespace codegen_fake

#define calloc codegen_fake::allocate
#define free codegen_fake::release
#define vkDestroyBuffer codegen_fake::unexpected_destroy
#define vkFreeMemory codegen_fake::unexpected_destroy
#define vkDestroyDescriptorPool codegen_fake::unexpected_destroy
#define vkDestroyDescriptorSetLayout codegen_fake::unexpected_destroy
#define vkDestroyPipelineLayout codegen_fake::unexpected_destroy
#define vkDestroyPipeline codegen_fake::unexpected_destroy
#include "vkFFT/vkFFT_CodeGen/vkFFT_MathUtils/vkFFT_MathUtils.h"
#include "vkFFT/vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_DeletePlan.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel2/vkFFT_FFT.h"
#include "vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel2/vkFFT_R2C_even_decomposition.h"

namespace codegen_fake {

int container_case(int type, int fail_at, int* attempts) {
    state = {};
    state.fail_at = fail_at;
    VkFFTSpecializationConstantsLayout sc{};
    PfContainer container{};
    container.type = type;
    PfAllocateContainerFlexible(&sc, &container, 50);
    *attempts = state.attempts;
    if (sc.res != (fail_at ? VKFFT_ERROR_MALLOC_FAILED : VKFFT_SUCCESS)) return __LINE__;
    if (!fail_at && type > 100) PfSetContainerName(&sc, &container, "value");
    PfDeallocateContainer(&sc, &container);
    PfDeallocateContainer(&sc, &container);
    if (live() || state.invalid || container.type || container.size || container.name) return __LINE__;
    return 0;
}

int copy_and_swap_case() {
    state = {};
    VkFFTSpecializationConstantsLayout sc{};
    PfContainer left{}, right{};
    left.type = right.type = 133;
    PfAllocateContainerFlexible(&sc, &left, 50);
    PfAllocateContainerFlexible(&sc, &right, 50);
    PfSetContainerName(&sc, &left, "left");
    PfSetContainerName(&sc, &right, "right");
    PfCopyContainer(&sc, &left, &right);
    PfSwapContainers(&sc, &left, &right);
    if (sc.res != VKFFT_SUCCESS || std::strcmp(left.name, "right")) return __LINE__;
    PfDeallocateContainer(&sc, &left);
    PfDeallocateContainer(&sc, &right);
    return live() || state.invalid ? __LINE__ : 0;
}

VkFFTResult initialize_parameters(VkFFTApplication* app, VkFFTAxis* axis) {
    auto* sc = &axis->specializationConstants;
    sc->maxTempLength = 5000;
    sc->inline_rader_g_pow = 1;
    sc->inline_rader_kernel = 1;
    sc->performPostCompilationInputOffset = 1;
    sc->performPostCompilationOutputOffset = 1;
    sc->performPostCompilationKernelOffset = 1;
    for (int i = 0; i < 3; ++i) {
        sc->performWorkGroupShift[i] = 1;
        sc->localSize[i].type = 31;
        sc->localSize[i].data.i = 64;
    }
    auto result = initMemoryParametersAPI(app, sc);
    if (result != VKFFT_SUCCESS) return result;
    sc->inputMemoryCode = sc->vecTypeInputMemoryCode;
    sc->outputMemoryCode = sc->vecTypeOutputMemoryCode;
    return initParametersAPI(app, sc);
}

int parameter_case(int fail_at, bool release_before_drop, int* attempts) {
    state = {};
    VkFFTApplication app{};
    VkFFTAxis axis{};
    auto& sc = axis.specializationConstants;
    sc.raderContainer = static_cast<VkFFTRaderContainer*>(allocate(1, sizeof(VkFFTRaderContainer)));
    if (!sc.raderContainer) return __LINE__;
    sc.raderContainer[0].prime = 17;
    sc.numRaderPrimes = 1;
    state.attempts = 0;
    state.fail_at = fail_at;
    auto result = initialize_parameters(&app, &axis);
    *attempts = state.attempts;
    if (result != (fail_at ? VKFFT_ERROR_MALLOC_FAILED : VKFFT_SUCCESS)) return __LINE__;
    if (release_before_drop) {
        freeMemoryParametersAPI(&app, &sc);
        freeParametersAPI(&app, &sc);
    }
    deleteAxis(&app, &axis, 0);
    deleteAxis(&app, &axis, 0);
    return live() || state.invalid ? __LINE__ : 0;
}

int borrowed_rader_case() {
    state = {};
    VkFFTApplication app{};
    VkFFTAxis owner{}, borrower{};
    auto& sc = owner.specializationConstants;
    sc.raderContainer = static_cast<VkFFTRaderContainer*>(allocate(1, sizeof(VkFFTRaderContainer)));
    if (!sc.raderContainer) return __LINE__;
    sc.raderContainer[0].prime = 17;
    sc.numRaderPrimes = 1;
    if (initialize_parameters(&app, &owner) != VKFFT_SUCCESS) return __LINE__;
    freeMemoryParametersAPI(&app, &sc);
    freeParametersAPI(&app, &sc);
    borrower.specializationConstants.raderContainer = sc.raderContainer;
    borrower.specializationConstants.numRaderPrimes = 1;
    if (initialize_parameters(&app, &borrower) != VKFFT_SUCCESS) return __LINE__;
    // Match application teardown: the owner's Rader array dies before the borrower.
    deleteAxis(&app, &owner, 0);
    deleteAxis(&app, &borrower, 1);
    return live() || state.invalid ? __LINE__ : 0;
}

int register_case(int mode, int fail_at, bool fail_after_initialization, int* attempts) {
    state = {};
    VkFFTApplication app{};
    VkFFTAxis axis{};
    auto& sc = axis.specializationConstants;
    if (mode == 3) app.configuration.quadDoubleDoublePrecision = 1;
    if (initialize_parameters(&app, &axis) != VKFFT_SUCCESS) return __LINE__;
    sc.maxCodeLength = 65536;
    sc.code0 = static_cast<char*>(allocate(sc.maxCodeLength, 1));
    if (!sc.code0) return __LINE__;
    sc.registers_per_thread = 4;
    sc.registerBoost = 1;
    sc.matrixConvolution = 1;
    sc.usedLocRegs = 4;
    sc.fftDim.type = sc.fft_dim_full.type = sc.stageStartSize.type = 31;
    sc.fftDim.data.i = 64;
    sc.fft_dim_full.data.i = 128; // Allocates disableThreads without explicit zero-padding.
    sc.stageStartSize.data.i = 2;
    sc.mergeSequencesR2C = 1;
    if (mode == 1 || mode == 3) {
        sc.convolutionStep = 1;
        sc.matrixConvolution = 2;
        sc.numKernels.data.i = 2;
        sc.useRader = sc.useRaderFFT = sc.useRaderMult = 1;
        sc.raderRegisters = 8;
        sc.numStages = 1;
        sc.stageRadix[0] = 8;
        sc.maxNonPow2Radix = 1;
        sc.LUT = 1;
        sc.numAxisUploads = 2;
    }
    const int before = live();
    state.attempts = 0;
    state.fail_at = fail_at;
    if (mode == 2) appendRegisterInitialization_R2C(&sc, 500);
    else appendRegisterInitialization(&sc, 500);
    *attempts = state.attempts;
    if (sc.res != (fail_at ? VKFFT_ERROR_MALLOC_FAILED : VKFFT_SUCCESS)) return __LINE__;
    if (fail_after_initialization) sc.res = VKFFT_ERROR_INSUFFICIENT_CODE_BUFFER;
    const auto expected = sc.res;
    for (int repeat = 0; repeat < 2; ++repeat) {
        if (mode == 2) freeRegisterInitialization_R2C(&sc, 500);
        else freeRegisterInitialization(&sc, 500);
    }
    if (live() != before || state.invalid || sc.res != expected || sc.regIDs || sc.regIDs_copy || sc.temp_conv)
        return __LINE__;
    deleteAxis(&app, &axis, 0);
    return live() || state.invalid ? __LINE__ : 0;
}

int generator_register_failure(bool r2c) {
    state = {};
    VkFFTApplication app{};
    VkFFTAxis axis{};
    auto& sc = axis.specializationConstants;
    if (initialize_parameters(&app, &axis) != VKFFT_SUCCESS) return __LINE__;
    sc.maxCodeLength = 65536;
    sc.code0 = static_cast<char*>(allocate(sc.maxCodeLength, 1));
    if (!sc.code0) return __LINE__;
    sc.registers_per_thread = 4;
    sc.registerBoost = 1;
    sc.matrixConvolution = 1;
    sc.fftDim.type = sc.fft_dim_full.type = 31;
    sc.fftDim.data.i = sc.fft_dim_full.data.i = 64;
    sc.numAxisUploads = 1;
    sc.numStages = 2;
    sc.stageRadix[0] = sc.stageRadix[1] = 8;
    sc.registers_per_thread_per_radix[8] = 4;
    sc.complexSize = 8;
    sc.inputBufferBlockNum = sc.outputBufferBlockNum = 1;
    sc.inputBufferBlockSize = sc.outputBufferBlockSize = 64 * sc.complexSize;
    sc.numSharedBanks = 32;
    sc.sharedMemSize = 32768;
    const int before = live();
    state.attempts = 0;
    state.fail_at = 1;
    const auto result = r2c ? shaderGen_R2C_even_decomposition(&sc, 500) : shaderGen_FFT(&sc, 500);
    if (result != VKFFT_ERROR_MALLOC_FAILED || state.attempts != 1 || live() != before) return __LINE__;
    deleteAxis(&app, &axis, 0);
    return live() || state.invalid ? __LINE__ : 0;
}

int format_case(int capacity, const char* input, VkFFTResult expected) {
    std::array<char, 64> temporary;
    temporary.fill('!');
    VkFFTSpecializationConstantsLayout sc{};
    sc.tempStr = temporary.data();
    sc.maxTempLength = capacity;
    const int result = PfFormatTemp(&sc, "%s", input);
    if (sc.res != expected) return __LINE__;
    if (capacity > 0 && result != static_cast<int>(std::strlen(input))) return __LINE__;
    const size_t boundary = capacity > 0 ? static_cast<size_t>(capacity) : 0;
    for (size_t i = boundary; i < temporary.size(); ++i)
        if (temporary[i] != '!') return __LINE__;
    if (expected == VKFFT_SUCCESS && std::strcmp(temporary.data(), input)) return __LINE__;
    const auto before = temporary;
    sc.res = VKFFT_ERROR_MATH_FAILED;
    PfFormatTemp(&sc, "%s", "ignored");
    return temporary == before && sc.res == VKFFT_ERROR_MATH_FAILED ? 0 : __LINE__;
}

int rewritten_format_case() {
    state = {};
    VkFFTSpecializationConstantsLayout sc{};
    PfContainer value{};
    value.type = 112;
    PfAllocateContainerFlexible(&sc, &value, 50);
    std::array<char, 64> temporary;
    temporary.fill('!');
    char output[128] = "unchanged";
    char float_name[] = "float";
    sc.floatDef.name = float_name;
    sc.tempStr = temporary.data();
    sc.maxTempLength = 8;
    sc.code0 = output;
    sc.maxCodeLength = sizeof(output);
    PfDefine(&sc, &value, "abcdefghi");
    if (sc.res != VKFFT_ERROR_INSUFFICIENT_TEMP_BUFFER || std::strcmp(output, "unchanged")) return __LINE__;
    for (size_t i = 8; i < temporary.size(); ++i) if (temporary[i] != '!') return __LINE__;
    PfDeallocateContainer(&sc, &value);
    return live() || state.invalid ? __LINE__ : 0;
}

struct AppendCase {
    pfINT current;
    pfINT length;
    pfINT capacity;
    pfINT temporary_capacity;
    VkFFTResult expected;
};
int append_case(const AppendCase& test) {
    std::array<char, 24> output;
    output.fill('!');
    output[0] = 'a';
    output[1] = 'b';
    output[2] = 0;
    const auto before = output;
    char temporary[] = "123456789";
    VkFFTSpecializationConstantsLayout sc{};
    sc.code0 = output.data();
    sc.tempStr = temporary;
    sc.tempLen = test.length;
    sc.currentLen = test.current;
    sc.maxCodeLength = test.capacity;
    sc.maxTempLength = test.temporary_capacity;
    PfAppendLine(&sc);
    if (sc.res != test.expected) return __LINE__;
    if (test.expected != VKFFT_SUCCESS) {
        if (output != before || sc.currentLen != test.current) return __LINE__;
    } else {
        const auto end = static_cast<size_t>(test.current + test.length);
        if (sc.currentLen != static_cast<pfINT>(end) || output[end] != 0) return __LINE__;
        if (std::memcmp(output.data() + test.current, temporary, static_cast<size_t>(test.length)))
            return __LINE__;
        for (size_t i = end + 1; i < output.size(); ++i) {
            if (output[i] != before[i]) return __LINE__;
        }
    }
    return 0;
}

} // namespace codegen_fake

extern "C" int vkfft_test_codegen_containers() {
    using namespace codegen_fake;
    for (int type : {0, 1, 2, 3, 11, 12, 13, 22, 23, 31, 32, 33,
            101, 112, 113, 122, 123, 132, 133, 201, 213, 300, 313, 323}) {
        int attempts = 0;
        if (int line = container_case(type, 0, &attempts)) return line;
        for (int fail_at = 1; fail_at <= attempts; ++fail_at) {
            int ignored = 0;
            if (int line = container_case(type, fail_at, &ignored)) return line;
        }
    }
    return copy_and_swap_case();
}

extern "C" int vkfft_test_codegen_bounds() {
    using namespace codegen_fake;
    constexpr auto code_error = VKFFT_ERROR_INSUFFICIENT_CODE_BUFFER;
    constexpr auto temp_error = VKFFT_ERROR_INSUFFICIENT_TEMP_BUFFER;
    constexpr auto maximum = std::numeric_limits<pfINT>::max();
    const AppendCase cases[] = {
        {0, 0, 8, 10, VKFFT_SUCCESS}, {0, 7, 8, 10, VKFFT_SUCCESS},
        {2, 5, 8, 10, VKFFT_SUCCESS}, {2, 6, 8, 10, code_error},
        {0, 8, 8, 10, code_error}, {0, 9, 8, 10, code_error},
        {0, 0, 0, 10, code_error}, {-1, 1, 8, 10, code_error},
        {8, 0, 8, 10, code_error}, {9, 0, 8, 10, code_error},
        {maximum - 2, 8, maximum, 10, code_error},
        {0, -1, 8, 10, temp_error}, {0, 7, 8, 7, temp_error},
        {0, 0, 8, 0, temp_error}, {0, 1, 8, -1, temp_error},
    };
    for (const auto& test : cases) if (int line = append_case(test)) return line;
    VkFFTSpecializationConstantsLayout sc{};
    PfAppendLine(&sc);
    if (sc.res != temp_error) return __LINE__;
    sc.res = VKFFT_ERROR_MATH_FAILED;
    PfAppendLine(&sc);
    return sc.res == VKFFT_ERROR_MATH_FAILED ? 0 : __LINE__;
}

extern "C" int vkfft_test_codegen_parameters() {
    using namespace codegen_fake;
    for (bool release_before_drop : {false, true}) {
        int attempts = 0;
        if (int line = parameter_case(0, release_before_drop, &attempts)) return line;
        for (int fail_at = 1; fail_at <= attempts; ++fail_at) {
            int ignored = 0;
            if (int line = parameter_case(fail_at, release_before_drop, &ignored)) return line;
        }
    }
    return borrowed_rader_case();
}

extern "C" int vkfft_test_codegen_registers() {
    using namespace codegen_fake;
    for (int mode = 0; mode < 4; ++mode) {
        int attempts = 0;
        if (int line = register_case(mode, 0, false, &attempts)) return line;
        for (int fail_at = 1; fail_at <= attempts; ++fail_at) {
            int ignored = 0;
            if (int line = register_case(mode, fail_at, false, &ignored)) return line;
        }
        if (int line = register_case(mode, 0, true, &attempts)) return line;
    }
    for (bool r2c : {false, true}) {
        if (int line = generator_register_failure(r2c)) return line;
    }
    VkFFTSpecializationConstantsLayout sc{};
    sc.registers_per_thread = INT_MAX;
    sc.registerBoost = 2;
    if (vkfft_codegen::register_count(&sc, 500, false)) return __LINE__;
    sc.registers_per_thread = 3;
    if (vkfft_codegen::register_count(&sc, 500, true)) return __LINE__;
    sc.usedLocRegs = 34;
    if (vkfft_codegen::register_layout_valid(&sc)) return __LINE__;
    sc.usedLocRegs = 0;
    sc.useRaderMult = 1;
    return vkfft_codegen::register_layout_valid(&sc) ? __LINE__ : 0;
}

extern "C" int vkfft_test_codegen_formatting() {
    using namespace codegen_fake;
    for (const char* input : {"", "1234567", "12345678", "123456789012345678901234567890"}) {
        const auto expected = std::strlen(input) < 8 ? VKFFT_SUCCESS : VKFFT_ERROR_INSUFFICIENT_TEMP_BUFFER;
        if (int line = format_case(8, input, expected)) return line;
    }
    for (int capacity : {0, -1})
        if (int line = format_case(capacity, "x", VKFFT_ERROR_INSUFFICIENT_TEMP_BUFFER)) return line;
    VkFFTSpecializationConstantsLayout sc{};
    sc.maxTempLength = 8;
    PfFormatTemp(&sc, "%s", "x");
    if (sc.res != VKFFT_ERROR_INSUFFICIENT_TEMP_BUFFER) return __LINE__;
    return rewritten_format_case();
}

#include "codegen_name_tests.hpp"
#include "codegen_permutation_tests.hpp"
#include "codegen_stage_tests.hpp"
#include "codegen_radix_stage_tests.hpp"
#include "codegen_stage_policy_tests.hpp"
#include "codegen_rader_layout_tests.hpp"
#include "scheduler_tests.hpp"
