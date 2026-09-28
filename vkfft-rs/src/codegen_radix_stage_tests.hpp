// CPU fixtures for both memory layouts and the ordinary/boosted shuffle branches.
namespace codegen_fake {

int initialize_radix_stage_fixture(VkFFTApplication* app, VkFFTAxis* axis, int mode, int precision, bool strided) {
    app->configuration.doublePrecision = precision == 1;
    app->configuration.quadDoubleDoublePrecision = precision == 2;
    if (initialize_parameters(app, axis) != VKFFT_SUCCESS) return __LINE__;
    auto& sc = axis->specializationConstants;
    sc.maxCodeLength = 262144;
    sc.code0 = static_cast<char*>(allocate(sc.maxCodeLength, 1));
    if (!sc.code0) return __LINE__;
    sc.registers_per_thread = mode == 3 ? 18 : 16;
    sc.registerBoost = mode >= 2 ? 2 : 1;
    sc.registers_per_thread_per_radix[2] = sc.registers_per_thread_per_radix[4] = 16;
    sc.matrixConvolution = 1;
    sc.usedLocRegs = 33;
    sc.LUT = 1;
    sc.warpSize = sc.numSharedBanks = 32;
    sc.numStages = 3;
    sc.stageRadix[0] = sc.stageRadix[1] = 4;
    sc.stageRadix[2] = mode == 3 ? 2 : 4;
    sc.maxNonPow2Radix = 13;
    sc.stridedSharedLayout = strided;
    sc.readToRegisters = 1;
    sc.double_PI = 3.141592653589793238462643383279502884L;
    sc.fftDim.type = sc.fft_dim_full.type = sc.sourceFFTSize.type = 31;
    sc.fftDim.data.i = sc.fft_dim_full.data.i = sc.sourceFFTSize.data.i = 64;
    sc.sharedStride.type = sc.sharedStrideBankConflictFirstStages.type = sc.sharedStrideReadWriteConflict.type = 31;
    sc.sharedStride.data.i = sc.sharedStrideReadWriteConflict.data.i = 64;
    sc.sharedStrideBankConflictFirstStages.data.i = 68;
    sc.localSize[0].data.i = strided ? 1 : 4;
    sc.localSize[1].data.i = strided ? 4 : 1;
    sc.localSize[2].data.i = 1;
    appendRegisterInitialization(&sc, 500);
    return sc.res == VKFFT_SUCCESS ? 0 : __LINE__;
}

void generate_radix_stage(VkFFTSpecializationConstantsLayout* sc, int mode, bool strided, bool shuffle) {
    PfContainer stage{}, sum{}, angle{}, radix{}, next{};
    stage.type = sum.type = radix.type = next.type = 31;
    stage.data.i = mode == 1 ? 4 : mode == 3 ? 8 : 1;
    radix.data.i = 4;
    next.data.i = mode == 3 ? 2 : 4;
    angle.type = 22;
    angle.data.d = -sc->double_PI;
    if (shuffle) appendRadixShuffle(sc, &stage, &sum, &angle, &radix, &next, 0, strided ? 1 : 0);
    else appendRadixStage(sc, &stage, &sum, &angle, &radix, 0, strided ? 1 : 0);
}

int radix_stage_storage_case(int mode, int precision, bool strided, bool shuffle,
                             int fail_at, bool throws, bool exhaust_code, int* attempts) {
    state = {};
    VkFFTApplication app{};
    VkFFTAxis axis{};
    if (int line = initialize_radix_stage_fixture(&app, &axis, mode, precision, strided)) return line;
    auto& sc = axis.specializationConstants;
    const int baseline = live();
    state.attempts = 0;
    state.fail_at = throws ? 0 : fail_at;
    state.throw_at = throws ? fail_at : 0;
    if (exhaust_code) sc.maxCodeLength = sc.currentLen + 256;
    bool caught = false;
    try { generate_radix_stage(&sc, mode, strided, shuffle); }
    catch (const std::bad_alloc&) { caught = true; }
    *attempts = state.attempts;
    const auto expected = !throws && fail_at ? VKFFT_ERROR_MALLOC_FAILED :
        exhaust_code ? VKFFT_ERROR_INSUFFICIENT_CODE_BUFFER : VKFFT_SUCCESS;
    if (caught != throws || sc.res != expected || live() != baseline || state.invalid) {
        std::fprintf(stderr, "radix storage: mode=%d precision=%d strided=%d shuffle=%d fail=%d throws=%d code=%d result=%d expected=%d live=%d/%d caught=%d invalid=%d\n",
            mode, precision, strided, shuffle, fail_at, throws, exhaust_code, sc.res, expected, live(), baseline, caught, state.invalid);
        return __LINE__;
    }
    if (fail_at && state.attempts != fail_at) return __LINE__;
    freeRegisterInitialization(&sc, 500);
    deleteAxis(&app, &axis, 0);
    return live() || state.invalid ? __LINE__ : 0;
}

int radix_stage_invalid_layout_case() {
    state = {};
    VkFFTApplication app{};
    VkFFTAxis axis{};
    if (int line = initialize_radix_stage_fixture(&app, &axis, 0, 0, false)) return line;
    auto& sc = axis.specializationConstants;
    const int baseline = live();
    auto* registers = sc.regIDs;
    state.attempts = 0;
    for (int mode = 0; mode < 10; ++mode) {
        sc.res = VKFFT_SUCCESS;
        sc.regIDs = registers;
        sc.registers_per_thread = 16;
        sc.registerBoost = 1;
        sc.registers_per_thread_per_radix[4] = 16;
        pfINT radix = 4;
        switch (mode) {
            case 0: radix = -1; break;
            case 1: radix = 33; break;
            case 2: sc.regIDs = nullptr; break;
            case 3: sc.registers_per_thread = 0; break;
            case 4: sc.registerBoost = 0; break;
            case 5: sc.registers_per_thread = INT_MAX; sc.registerBoost = 2; break;
            case 6: sc.registers_per_thread_per_radix[4] = 0; break;
            case 7: sc.registers_per_thread_per_radix[4] = 20; break;
            case 8: sc.registers_per_thread_per_radix[4] = 15; break;
            case 9: radix = 0; break;
        }
        if (vkfft_codegen::stage_register_layout(&sc, radix) || sc.res != VKFFT_ERROR_MATH_FAILED ||
            state.attempts || live() != baseline) return __LINE__;
    }
    sc.regIDs = registers;
    sc.registers_per_thread = sc.registers_per_thread_per_radix[4] = 16;
    sc.registerBoost = 1;
    for (bool strided : {false, true}) for (bool shuffle : {false, true}) {
        for (int mode = 0; mode < 5; ++mode) {
            sc.res = VKFFT_SUCCESS;
            sc.fftDim.data.i = mode == 0 ? 0 : mode == 1 ? 1 : 64;
            sc.localSize[0].data.i = mode == 2 ? 0 : 4;
            sc.localSize[1].data.i = mode == 3 ? 0 : 1;
            sc.numSharedBanks = mode == 4 ? 1 : 32;
            generate_radix_stage(&sc, 0, strided, shuffle);
            if (sc.res != VKFFT_ERROR_MATH_FAILED || state.attempts || live() != baseline) return __LINE__;
        }
    }
    sc.fftDim.data.i = 64;
    sc.localSize[0].data.i = 4;
    sc.localSize[1].data.i = 1;
    sc.numSharedBanks = 32;
    for (bool strided : {false, true}) for (int mode = 0; mode < 3; ++mode) {
        sc.res = VKFFT_SUCCESS;
        sc.numStages = mode == 0 ? 0 : mode == 1 ? 34 : 3;
        sc.stageRadix[2] = mode == 2 ? 0 : 4;
        generate_radix_stage(&sc, 0, strided, true);
        if (sc.res != VKFFT_ERROR_MATH_FAILED || state.attempts || live() != baseline) return __LINE__;
    }
    sc.numStages = 3;
    sc.stageRadix[2] = 4;
    sc.res = VKFFT_ERROR_MALLOC_FAILED;
    for (bool strided : {false, true}) for (bool shuffle : {false, true})
        generate_radix_stage(&sc, 0, strided, shuffle);
    if (sc.res != VKFFT_ERROR_MALLOC_FAILED || state.attempts || live() != baseline) return __LINE__;
    freeRegisterInitialization(&sc, 500);
    deleteAxis(&app, &axis, 0);
    return live() || state.invalid ? __LINE__ : 0;
}
} // namespace codegen_fake

extern "C" int vkfft_test_codegen_radix_stage_storage() {
    using namespace codegen_fake;
    for (bool shuffle : {false, true}) for (bool strided : {false, true})
    for (int precision = 0; precision < 3; ++precision) for (int mode = 0; mode < 4; ++mode) {
        int attempts = 0;
        if (int line = radix_stage_storage_case(mode, precision, strided, shuffle, 0, false, false, &attempts)) return line;
        for (bool throws : {false, true}) for (int fail_at = 1; fail_at <= attempts; ++fail_at) {
            int ignored = 0;
            if (int line = radix_stage_storage_case(mode, precision, strided, shuffle, fail_at, throws, false, &ignored)) return line;
        }
        if (mode != 3 || !shuffle) {
            // The final boosted shuffle only remaps names and emits no shader text.
            if (int line = radix_stage_storage_case(mode, precision, strided, shuffle, 0, false, true, &attempts)) return line;
        }
    }
    return 0;
}

extern "C" int vkfft_test_codegen_radix_stage_layout() {
    return codegen_fake::radix_stage_invalid_layout_case();
}
