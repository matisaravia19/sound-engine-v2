// CPU regressions for dispatcher metadata, normalization arithmetic and direct Rader stages.
namespace codegen_fake {

int normalization_policy_case() {
    state = {};
    VkFFTSpecializationConstantsLayout sc{};
    sc.actualInverse = sc.normalize = 1;
    constexpr pfINT maximum = INT64_MAX;
    struct Case {
        pfINT source, full;
        int dct, dst, blue;
        pfINT expected;
    };
    const Case cases[] = {
        {32, 64, 0, 0, 0, 32},
        {32, 64, 1, 0, 0, 62},
        {32, 64, 0, 1, 0, 66},
        {32, 64, 2, 0, 0, 64},
        {32, 64, 0, 2, 0, 64},
        {32, 64, 0, 0, 1, 2048},
        {32, 64, 1, 0, 1, 3968},
        {32, 64, 0, 1, 1, 4224},
        {maximum, 1, 0, 0, 0, maximum},
        {maximum / 2, 1, 2, 0, 0, maximum - 1},
        {maximum / 2 - 1, 1, 0, 1, 0, maximum - 1},
        {maximum / 2 + 1, 1, 1, 0, 0, maximum - 1},
        {0, 64, 0, 0, 0, 0},
        {-1, 64, 0, 0, 0, 0},
        {1, 64, 1, 0, 0, 0},
        {maximum, 64, 0, 1, 0, 0},
        {maximum / 2 + 1, 1, 2, 0, 0, 0},
        {maximum / 2, 1, 0, 1, 0, 0},
        {maximum / 2 + 2, 1, 1, 0, 0, 0},
        {maximum, 2, 0, 0, 1, 0},
        {32, 0, 0, 0, 1, 0},
        {32, -1, 0, 0, 1, 0},
    };
    for (const auto& c : cases) {
        sc.res = VKFFT_SUCCESS;
        sc.sourceFFTSize.data.i = c.source;
        sc.fft_dim_full.data.i = c.full;
        sc.performDCT = c.dct;
        sc.performDST = c.dst;
        sc.useBluesteinFFT = c.blue;
        pfINT divisor = 77;
        bool valid = vkfft_codegen::stage_normalization(&sc, 1, 1, divisor);
        if (valid != (c.expected != 0) || divisor != (valid ? c.expected : 77) ||
            sc.res != (valid ? VKFFT_SUCCESS : VKFFT_ERROR_MATH_FAILED))
            return __LINE__;
    }
    // Unused dimensions must not affect later stages, forward transforms or uploads.
    for (int mode = 0; mode < 6; ++mode) {
        sc = {};
        sc.actualInverse = sc.normalize = 1;
        pfINT stage = 1;
        pfLD angle = 1;
        switch (mode) {
        case 0:
            stage = 2;
            break;
        case 1:
            sc.axis_upload_id = 1;
            break;
        case 2:
            sc.actualInverse = 0;
            break;
        case 3:
            sc.useBluesteinFFT = 1;
            angle = -1;
            break;
        case 4:
            sc.normalize = 0;
            break;
        case 5:
            sc.actualInverse = 0;
            sc.convolutionStep = 1;
            angle = -1;
            break;
        }
        pfINT divisor = 77;
        if (!vkfft_codegen::stage_normalization(&sc, stage, angle, divisor) || divisor != 1)
            return __LINE__;
    }
    sc = {};
    sc.convolutionStep = 1;
    sc.performDCT = 1;
    sc.sourceFFTSize.data.i = 32;
    pfINT divisor = 77;
    if (!vkfft_codegen::stage_normalization(&sc, 1, 1, divisor) || divisor != 32)
        return __LINE__;
    sc.res = VKFFT_ERROR_MALLOC_FAILED;
    if (vkfft_codegen::stage_normalization(&sc, 1, 1, divisor) || divisor != 32 ||
        sc.res != VKFFT_ERROR_MALLOC_FAILED || state.attempts)
        return __LINE__;
    return 0;
}

int stage_dispatch_policy_case() {
    state = {};
    VkFFTSpecializationConstantsLayout sc{};
    PfContainer size{}, sum{}, angle{}, radix{};
    size.type = sum.type = radix.type = 31;
    size.data.i = 1;
    radix.data.i = 4;
    angle.type = 22;
    angle.data.d = -1;
    for (bool shuffle : {false, true})
        for (int mode = 0; mode < 9; ++mode) {
            sc = {};
            sc.numStages = 1;
            sc.stageRadix[0] = 4;
            int stage = 0, layout = 0;
            switch (mode) {
            case 0:
                stage = -1;
                break;
            case 1:
                stage = 1;
                break;
            case 2:
                sc.numStages = 0;
                break;
            case 3:
                sc.numStages = 34;
                break;
            case 4:
                sc.stageRadix[0] = 2;
                break;
            case 5:
                sc.rader_generator[0] = -1;
                break;
            case 6:
                layout = 3;
                break;
            case 7:
                layout = 509;
                break;
            case 8:
                layout = -4;
                break;
            }
            if (shuffle)
                appendRadixShuffle(&sc, &size, &sum, &angle, &radix, &radix, stage, layout);
            else
                appendRadixStage(&sc, &size, &sum, &angle, &radix, stage, layout);
            if (sc.res != VKFFT_ERROR_MATH_FAILED || state.attempts || sc.currentLen)
                return __LINE__;
        }
    for (int mode = 0; mode < 7; ++mode) {
        VkFFTRaderContainer candidates[2]{};
        candidates[0].prime = candidates[1].prime = 17;
        candidates[0].generator = candidates[1].generator = 3;
        sc = {};
        sc.numStages = 1;
        sc.stageRadix[0] = 17;
        sc.rader_generator[0] = 3;
        sc.numRaderPrimes = 1;
        sc.raderContainer = candidates;
        sc.currentRaderContainer = &candidates[1];
        radix.data.i = 17;
        switch (mode) {
        case 0:
            sc.raderContainer = nullptr;
            break;
        case 1:
            sc.numRaderPrimes = 0;
            break;
        case 2:
            sc.numRaderPrimes = 34;
            break;
        case 3:
            candidates[0].prime = 19;
            break;
        case 4:
            candidates[0].type = 2;
            break;
        case 5:
            candidates[0].generator = 2;
            break;
        case 6:
            sc.numRaderPrimes = 2;
            break;
        }
        appendRadixStage(&sc, &size, &sum, &angle, &radix, 0, 0);
        if (sc.res != VKFFT_ERROR_MATH_FAILED || sc.currentRaderContainer || state.attempts || sc.currentLen)
            return __LINE__;
    }
    VkFFTRaderContainer candidates[2]{};
    candidates[0].prime = 19;
    candidates[1].prime = 17;
    candidates[1].generator = 3;
    sc = {};
    sc.numStages = 1;
    sc.stageRadix[0] = 17;
    sc.rader_generator[0] = 3;
    sc.numRaderPrimes = 2;
    sc.raderContainer = candidates;
    for (int type : {0, 1}) {
        candidates[1].type = type;
        if (!vkfft_codegen::stage_dispatch_valid(&sc, 0, 17, 502) ||
            !vkfft_codegen::select_rader_stage(&sc, 0, 17) || sc.currentRaderContainer != &candidates[1])
            return __LINE__;
    }
    return state.attempts || state.invalid ? __LINE__ : 0;
}

int normalization_entry_case(bool rader, bool direct, bool strided, bool overflow) {
    state = {};
    VkFFTApplication app{};
    VkFFTAxis axis{};
    int line = rader ? initialize_rader_fixture(&app, &axis, 0)
                     : initialize_radix_stage_fixture(&app, &axis, 0, 0, strided);
    if (line)
        return line;
    auto& sc = axis.specializationConstants;
    sc.actualInverse = sc.normalize = 1;
    sc.sourceFFTSize.data.i = overflow ? INT64_MAX : 0;
    sc.performDST = overflow;
    sc.stridedSharedLayout = strided;
    const int baseline = live();
    const auto length = sc.currentLen;
    state.attempts = 0;
    if (rader) {
        PfContainer stage{}, sum{}, angle{}, radix{};
        stage.type = sum.type = radix.type = 31;
        stage.data.i = 1;
        radix.data.i = 17;
        angle.type = 22;
        angle.data.d = 1;
        if (direct)
            appendMultRaderStage(&sc, &stage, &sum, &angle, &radix, 0);
        else
            appendFFTRaderStage(&sc, &stage, &sum, &angle, &radix, 0);
    } else
        generate_radix_stage(&sc, 0, strided, true);
    if (sc.res != VKFFT_ERROR_MATH_FAILED || state.attempts || live() != baseline || sc.currentLen != length)
        return __LINE__;
    freeRegisterInitialization(&sc, 500);
    deleteAxis(&app, &axis, 0);
    return live() || state.invalid ? __LINE__ : 0;
}

int direct_rader_policy_case(bool strided, bool double_precision, bool lut, int stage_size) {
    state = {};
    VkFFTApplication app{};
    VkFFTAxis axis{};
    if (int line = initialize_rader_fixture(&app, &axis, double_precision ? 2 : 0))
        return line;
    auto& sc = axis.specializationConstants;
    sc.stridedSharedLayout = strided;
    sc.currentRaderContainer->type = 1;
    if (!lut) {
        freeRegisterInitialization(&sc, 500);
        sc.LUT = 0;
        appendRegisterInitialization(&sc, 500);
    }
    sc.fftDim.data.i = sc.fft_dim_full.data.i = sc.sourceFFTSize.data.i = 17 * stage_size;
    sc.rader_generator[0] = 3;
    PfContainer stage{}, sum{}, angle{}, radix{};
    stage.type = sum.type = radix.type = 31;
    stage.data.i = stage_size;
    radix.data.i = 17;
    angle.type = 22;
    angle.data.d = -sc.double_PI;
    const int baseline = live();
    state.attempts = 0;
    state.fail_at = 1;
    appendRadixStage(&sc, &stage, &sum, &angle, &radix, 0, strided ? 1 : 0);
    if (sc.res != VKFFT_SUCCESS || state.attempts || live() != baseline || std::strstr(sc.code0, "inf") ||
        std::strstr(sc.code0, "nan"))
        return __LINE__;
    if (!lut && stage_size > 1) {
        // Both halves use the integer radix; compare the emitted coefficient to an independent calculation.
        char coefficient[100];
        std::snprintf(coefficient, sizeof(coefficient), "%.17Le", -2 * sc.double_PI / 17);
        if (!std::strstr(sc.code0, coefficient))
            return __LINE__;
    }
    freeRegisterInitialization(&sc, 500);
    deleteAxis(&app, &axis, 0);
    return live() || state.invalid ? __LINE__ : 0;
}
} // namespace codegen_fake

extern "C" int vkfft_test_codegen_stage_policy() {
    using namespace codegen_fake;
    if (int line = normalization_policy_case())
        return line;
    if (int line = stage_dispatch_policy_case())
        return line;
    for (bool rader : {false, true})
        for (bool direct : {false, true})
            for (bool strided : {false, true})
                for (bool overflow : {false, true})
                    if (int line = normalization_entry_case(rader, direct, strided, overflow))
                        return line;
    return 0;
}
extern "C" int vkfft_test_codegen_direct_rader() {
    using namespace codegen_fake;
    for (bool strided : {false, true})
        for (bool double_precision : {false, true})
            for (bool lut : {false, true})
                for (int stage_size : {1, 2})
                    if (int line = direct_rader_policy_case(strided, double_precision, lut, stage_size))
                        return line;
    return 0;
}
