// CPU-only stage generation: fixtures own all parameters and tracked allocations.
#include "codegen/storage.hpp"

namespace codegen_fake {

int initialize_rader_fixture(VkFFTApplication* app, VkFFTAxis* axis, int mode) {
    auto& sc = axis->specializationConstants;
    if (mode == 2)
        app->configuration.doublePrecision = 1;
    sc.raderContainer = static_cast<VkFFTRaderContainer*>(allocate(1, sizeof(VkFFTRaderContainer)));
    if (!sc.raderContainer)
        return __LINE__;
    sc.numRaderPrimes = 1;
    auto& rader = sc.raderContainer[0];
    rader.prime = 17;
    rader.generator = 3;
    rader.numStages = mode == 0 ? 1 : 2;
    rader.stageRadix[0] = mode == 0 ? 16 : 4;
    rader.stageRadix[1] = 4;
    rader.registers_per_thread_per_radix[4] = rader.registers_per_thread_per_radix[16] = 16;
    rader.containerFFTDim = 16;
    rader.containerFFTNum = 1;
    if (initialize_parameters(app, axis) != VKFFT_SUCCESS)
        return __LINE__;
    sc.maxCodeLength = 262144;
    sc.code0 = static_cast<char*>(allocate(sc.maxCodeLength, 1));
    if (!sc.code0)
        return __LINE__;
    sc.registers_per_thread = 16;
    sc.registerBoost = 1;
    sc.matrixConvolution = 1;
    sc.usedLocRegs = 33;
    sc.useRader = sc.useRaderFFT = sc.useRaderMult = 1;
    sc.raderRegisters = 4;
    sc.LUT = 1;
    sc.numSharedBanks = 32;
    sc.numStages = 1;
    sc.stageRadix[0] = 17;
    sc.maxNonPow2Radix = 13;
    sc.currentRaderContainer = &rader;
    sc.stridedSharedLayout = mode == 3;
    sc.double_PI = 3.141592653589793238462643383279502884L;
    sc.fftDim.type = sc.fft_dim_full.type = sc.sourceFFTSize.type = 31;
    sc.fftDim.data.i = sc.fft_dim_full.data.i = sc.sourceFFTSize.data.i = 17;
    sc.sharedStride.type = sc.sharedStrideRaderFFT.type = 31;
    sc.sharedStride.data.i = sc.sharedStrideRaderFFT.data.i = 17;
    appendRegisterInitialization(&sc, 500);
    return sc.res == VKFFT_SUCCESS ? 0 : __LINE__;
}

int rader_storage_case(int mode, int fail_at, bool exhaust_code, int* attempts) {
    state = {};
    VkFFTApplication app{};
    VkFFTAxis axis{};
    if (int line = initialize_rader_fixture(&app, &axis, mode))
        return line;
    auto& sc = axis.specializationConstants;
    PfContainer stage{}, sum{}, angle{}, radix{};
    stage.type = sum.type = radix.type = 31;
    stage.data.i = 1;
    radix.data.i = 17;
    angle.type = 22;
    angle.data.d = -sc.double_PI;
    const int baseline = live();
    state.attempts = 0;
    state.fail_at = fail_at;
    if (exhaust_code)
        sc.maxCodeLength = sc.currentLen + 256;
    appendFFTRaderStage(&sc, &stage, &sum, &angle, &radix, 0);
    *attempts = state.attempts;
    const auto expected = fail_at        ? VKFFT_ERROR_MALLOC_FAILED
                          : exhaust_code ? VKFFT_ERROR_INSUFFICIENT_CODE_BUFFER
                                         : VKFFT_SUCCESS;
    if (sc.res != expected || live() != baseline || state.invalid)
        return __LINE__;
    if (fail_at && state.attempts != fail_at)
        return __LINE__;
    freeRegisterInitialization(&sc, 500);
    deleteAxis(&app, &axis, 0);
    return live() || state.invalid ? __LINE__ : 0;
}

int scoped_containers_case() {
    state = {};
    VkFFTSpecializationConstantsLayout sc{};
    const pfINT invalid[][2] = {{0, 1},  {-1, 1},      {1, 0},
                                {1, -1}, {INT_MAX, 2}, {INT64_MAX, INT64_MAX}};
    for (const auto& pair : invalid) {
        sc.res = VKFFT_SUCCESS;
        vkfft_codegen::Containers values(&sc, pair[0], pair[1]);
        if (values.get() || sc.res != VKFFT_ERROR_MATH_FAILED || state.attempts)
            return __LINE__;
    }
    sc.res = VKFFT_ERROR_MALLOC_FAILED;
    {
        vkfft_codegen::Containers values(&sc, 2);
        if (values.get() || state.attempts || sc.res != VKFFT_ERROR_MALLOC_FAILED)
            return __LINE__;
    }
    sc.res = VKFFT_SUCCESS;
    {
        vkfft_codegen::Containers unused(&sc, 16, 1, false);
        if (unused.get() || sc.res != VKFFT_SUCCESS || state.attempts)
            return __LINE__;
    }
    state.fail_at = 1;
    {
        vkfft_codegen::Containers values(&sc, 2);
        if (values.get() || sc.res != VKFFT_ERROR_MALLOC_FAILED)
            return __LINE__;
    }
    sc.res = VKFFT_SUCCESS;
    state.fail_at = 0;
    struct Stop {};
    try {
        vkfft_codegen::Containers values(&sc, 2, 3);
        if (!values.get())
            return __LINE__;
        for (int i = 0; i < 6; ++i) {
            values.get()[i].type = 133;
            PfAllocateContainerFlexible(&sc, &values.get()[i], 50);
        }
        if (sc.res != VKFFT_SUCCESS)
            return __LINE__;
        sc.res = VKFFT_ERROR_INSUFFICIENT_CODE_BUFFER;
        throw Stop{};
    } catch (const Stop&) {}
    return live() || state.invalid || sc.res != VKFFT_ERROR_INSUFFICIENT_CODE_BUFFER ? __LINE__ : 0;
}

} // namespace codegen_fake

extern "C" int vkfft_test_codegen_rader_storage() {
    using namespace codegen_fake;
    for (int mode = 0; mode < 4; ++mode) {
        int attempts = 0;
        if (int line = rader_storage_case(mode, 0, false, &attempts))
            return line;
        for (int fail_at = 1; fail_at <= attempts; ++fail_at) {
            int ignored = 0;
            if (int line = rader_storage_case(mode, fail_at, false, &ignored))
                return line;
        }
        if (int line = rader_storage_case(mode, 0, true, &attempts))
            return line;
    }
    return scoped_containers_case();
}
