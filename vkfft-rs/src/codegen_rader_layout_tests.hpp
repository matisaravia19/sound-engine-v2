// CPU regressions keep malformed metadata away from both native indexing and the GPU.
namespace codegen_fake {

int invalid_rader_layout_case(bool direct, bool strided, int mode) {
    state = {};
    VkFFTApplication app{};
    VkFFTAxis axis{};
    if (int line = initialize_rader_fixture(&app, &axis, 1))
        return line;
    auto& sc = axis.specializationConstants;
    sc.stridedSharedLayout = strided;
    auto& rader = *sc.currentRaderContainer;
    rader.type = direct;
    const auto saved = sc;
    const auto saved_rader = rader;
    PfContainer stage{}, sum{}, angle{}, radix{};
    stage.type = sum.type = radix.type = 31;
    stage.data.i = 1;
    radix.data.i = 17;
    angle.type = 22;
    angle.data.d = -sc.double_PI;
    switch (mode) {
    case 0:
        sc.currentRaderContainer = nullptr;
        break;
    case 1:
        sc.regIDs = nullptr;
        break;
    case 2:
        sc.registers_per_thread = 0;
        break;
    case 3:
        sc.registers_per_thread = -1;
        break;
    case 4:
        sc.registerBoost = 0;
        break;
    case 5:
        sc.registerBoost = -1;
        break;
    case 6:
        sc.registerBoost = INT_MAX;
        break;
    case 7:
        sc.fftDim.data.i = 0;
        break;
    case 8:
        sc.fftDim.data.i = -17;
        break;
    case 9:
        sc.fftDim.data.i = 18;
        break;
    case 10:
        stage.data.i = 0;
        break;
    case 11:
        stage.data.i = -1;
        break;
    case 12:
        stage.data.i = 2;
        break;
    case 13:
        radix.data.i = 2;
        break;
    case 14:
        radix.data.i = INT64_MAX;
        break;
    case 15:
        sc.localSize[0].data.i = 0;
        break;
    case 16:
        sc.localSize[1].data.i = -1;
        break;
    case 17:
        rader.type = !direct;
        break;
    case 18:
        rader.generator = 0;
        break;
    case 19:
        rader.generator = 17;
        break;
    case 20:
        sc.useRader = 0;
        break;
    case 21:
        // A previous error must survive even when every layout field is invalid.
        sc.res = VKFFT_ERROR_INSUFFICIENT_CODE_BUFFER;
        sc.currentRaderContainer = nullptr;
        break;
    default:
        if (direct) {
            switch (mode) {
            case 22:
                sc.localSize[strided ? 1 : 0].data.i = 8;
                break;
            case 23:
                sc.useRaderMult = 0;
                break;
            case 24:
                sc.raderRegisters = 0;
                break;
            case 25:
                sc.raderRegisters = -2;
                break;
            case 26:
                sc.raderRegisters = 68;
                break;
            case 27:
                sc.localSize[strided ? 1 : 0].data.i = 9;
                sc.fftDim.data.i = 17 * 3; // Three groups exceed two initialized x0 registers.
                break;
            case 28:
                sc.registers_per_thread = 1; // A direct group needs two complex registers.
                break;
            case 29:
                sc.localSize[strided ? 1 : 0].data.i = 9;
                sc.raderRegisters = 66;
                sc.fftDim.data.i = 17 * 9; // x0 fits; the physical register array does not.
                break;
            case 30:
                sc.localSize[strided ? 1 : 0].data.i = INT64_MAX;
                break; // The rounded workgroup extent would overflow its signed product.
            }
        } else {
            switch (mode) {
            case 22:
                sc.useRaderFFT = 0;
                break;
            case 23:
                sc.numSharedBanks = 1;
                break;
            case 24:
                rader.numStages = 0;
                break;
            case 25:
                rader.numStages = 21;
                break;
            case 26:
                rader.stageRadix[0] = 0;
                break;
            case 27:
                rader.stageRadix[0] = 33;
                break;
            case 28:
                rader.stageRadix[0] = 17;
                break;
            case 29:
                rader.stageRadix[1] = 8;
                break;
            case 30:
                rader.stageRadix[1] = 2;
                break;
            case 31:
                rader.registers_per_thread_per_radix[4] = 0;
                break;
            case 32:
                rader.registers_per_thread_per_radix[4] = -4;
                break;
            case 33:
                rader.registers_per_thread_per_radix[4] = 6;
                break;
            case 34:
                rader.registers_per_thread_per_radix[4] = 32;
                break;
            case 35:
                rader.containerFFTDim = 0;
                break;
            case 36:
                rader.containerFFTDim = 15;
                break;
            case 37:
                rader.containerFFTNum = 0;
                break;
            case 38:
                rader.containerFFTNum = 2;
                break;
            }
        }
    }
    const int baseline = live();
    state.attempts = 0;
    if (direct)
        appendMultRaderStage(&sc, &stage, &sum, &angle, &radix, 0);
    else
        appendFFTRaderStage(&sc, &stage, &sum, &angle, &radix, 0);
    const auto expected = mode == 21 ? VKFFT_ERROR_INSUFFICIENT_CODE_BUFFER : VKFFT_ERROR_MATH_FAILED;
    const bool failed = sc.res != expected || state.attempts || live() != baseline ||
                        sc.currentLen != saved.currentLen || state.invalid;
    // Only metadata was changed; restore the original allocation dimensions for teardown.
    sc = saved;
    rader = saved_rader;
    freeRegisterInitialization(&sc, 500);
    deleteAxis(&app, &axis, 0);
    return failed || live() || state.invalid ? __LINE__ : 0;
}

int valid_rader_layout_case(bool direct, bool strided, bool double_precision, bool lut, int tiles,
                            int boost) {
    state = {};
    VkFFTApplication app{};
    VkFFTAxis axis{};
    if (int line = initialize_rader_fixture(&app, &axis, double_precision ? 2 : 1))
        return line;
    auto& sc = axis.specializationConstants;
    freeRegisterInitialization(&sc, 500);
    sc.stridedSharedLayout = strided;
    sc.currentRaderContainer->type = direct;
    sc.currentRaderContainer->containerFFTNum = tiles;
    sc.LUT = lut;
    sc.registerBoost = boost;
    sc.raderRegisters = tiles == 33 ? 66 : 16;
    sc.registers_per_thread = tiles == 33 ? 66 : 16;
    sc.warpSize = 32;
    sc.fftDim.data.i = sc.fft_dim_full.data.i = sc.sourceFFTSize.data.i = 17 * tiles;
    sc.localSize[strided ? 1 : 0].data.i = tiles == 33 ? 9 : direct ? 18 : 64;
    appendRegisterInitialization(&sc, 500);
    if (sc.res != VKFFT_SUCCESS)
        return __LINE__;
    PfContainer stage{}, sum{}, angle{}, radix{};
    stage.type = sum.type = radix.type = 31;
    stage.data.i = tiles == 2 ? 2 : 1;
    radix.data.i = 17;
    angle.type = 22;
    angle.data.d = -sc.double_PI;
    const int baseline = live();
    const auto length = sc.currentLen;
    state.attempts = 0;
    if (direct)
        appendMultRaderStage(&sc, &stage, &sum, &angle, &radix, 0);
    else
        appendFFTRaderStage(&sc, &stage, &sum, &angle, &radix, 0);
    if (sc.res != VKFFT_SUCCESS || live() != baseline || state.invalid || sc.currentLen <= length ||
        (direct && state.attempts) || std::strstr(sc.code0, "inf") || std::strstr(sc.code0, "nan"))
        return __LINE__;
    freeRegisterInitialization(&sc, 500);
    deleteAxis(&app, &axis, 0);
    return live() || state.invalid ? __LINE__ : 0;
}

} // namespace codegen_fake

extern "C" int vkfft_test_codegen_rader_layout() {
    using namespace codegen_fake;
    for (bool direct : {false, true})
        for (bool strided : {false, true})
            for (int mode = 0; mode < (direct ? 31 : 39); ++mode)
                if (int line = invalid_rader_layout_case(direct, strided, mode))
                    return line;
    return 0;
}

extern "C" int vkfft_test_codegen_rader_layout_valid() {
    using namespace codegen_fake;
    for (bool direct : {false, true})
        for (bool strided : {false, true})
            for (bool double_precision : {false, true})
                for (bool lut : {false, true})
                    for (int tiles : {1, 2, 8})
                        for (int boost : {1, 2})
                            if (int line = valid_rader_layout_case(direct, strided, double_precision, lut,
                                                                   tiles, boost))
                                return line;
    for (bool strided : {false, true})
        if (int line = valid_rader_layout_case(true, strided, false, true, 33, 1))
            return line;
    return 0;
}
