// Included by the CPU-only LUT test translation unit after its allocator macros.
#ifndef VKFFT_RS_LUT_LAYOUT_TESTS_HPP
#define VKFFT_RS_LUT_LAYOUT_TESTS_HPP

namespace lut_fake {

struct LayoutFixture {
    VkFFTApplication app{};
    std::unique_ptr<VkFFTPlan> plan = std::make_unique<VkFFTPlan>();
    VkFFTRaderContainer rader[2]{};

    LayoutFixture() {
        state = {};
        app.configuration.useLUT = 1;
        app.configuration.makeForwardPlanOnly = 1;
        app.configuration.size[0] = 16;
        auto& constants = sc();
        constants.numStages = 2;
        constants.stageRadix[0] = constants.stageRadix[1] = 4;
        constants.numAxisUploads = 1;
        constants.registerBoost = 1;
        rader[0].prime = 17;
        rader[1].prime = 31;
        for (auto& item : rader) { item.generator = 3; item.type = 1; }
    }
    VkFFTAxis& axis() { return plan->axes[0][0]; }
    VkFFTSpecializationConstantsLayout& sc() { return axis().specializationConstants; }
    void use_rader() {
        sc().useRader = sc().raderUintLUT = 1;
        sc().numRaderPrimes = 2;
        sc().raderContainer = rader;
    }
    VkFFTResult build() { return VkFFT_AllocateLUT(&app, plan.get(), &axis(), 0); }
    VkFFTResult build_integer() { return VkFFT_AllocateRaderUintLUT(&app, &axis()); }
};

static double uploaded_scalar(int precision, size_t index) {
    if (!precision) {
        float value;
        std::memcpy(&value, state.upload.data() + index * sizeof(float), sizeof(value));
        return value;
    }
    double value;
    std::memcpy(&value, state.upload.data() + index * sizeof(double), sizeof(value));
    return value;
}

} // namespace lut_fake

#define LAYOUT_CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

extern "C" int vkfft_test_lut_layout_bounds() {
    using namespace lut_fake;
    for (int invalid = 0; invalid < 22; ++invalid) {
        LayoutFixture fixture;
        auto& sc = fixture.sc();
        auto& config = fixture.app.configuration;
        if (invalid >= 10) fixture.use_rader();
        switch (invalid) {
        case 0: sc.numStages = -1; break;
        case 1: sc.numStages = 34; break;
        case 2: sc.stageRadix[1] = 0; break;
        case 3: sc.stageRadix[1] = INT_MIN; break;
        case 4:
            sc.numStages = 3;
            sc.stageRadix[0] = INT_MAX;
            sc.stageRadix[1] = sc.stageRadix[2] = 2;
            break;
        case 5: case 6:
            config.useLUT_4step = 1;
            sc.axis_upload_id = 1;
            sc.numAxisUploads = 2;
            sc.stageStartSize.data.i = invalid == 5 ? int64_t(1) << 32 : -1;
            sc.fftDim.data.i = int64_t(1) << 32;
            break;
        case 7: sc.axis_id = VKFFT_MAX_FFT_DIMENSIONS; break;
        case 8: sc.axis_id = -1; break;
        case 9: sc.axis_upload_id = 4; break;
        case 10: sc.numRaderPrimes = -1; break;
        case 11: sc.numRaderPrimes = 34; break;
        case 12: sc.raderContainer = nullptr; break;
        case 13: case 14:
            sc.inline_rader_kernel = 1;
            fixture.rader[0].type = 0;
            fixture.rader[0].numStages = invalid == 13 ? 21 : -1;
            break;
        case 15: fixture.rader[0].prime = 0; break;
        case 16: fixture.rader[0].generator = fixture.rader[0].prime; break;
        case 17: fixture.rader[0].type = 2; break;
        case 18: sc.stageRadix[1] = 11; sc.rader_generator[1] = 2; break;
        case 19: sc.numAxisUploads = 0; break;
        case 20: sc.numAxisUploads = 5; break;
        case 21:
            fixture.rader[0].type = 0;
            fixture.rader[0].numStages = 1;
            fixture.rader[0].stageRadix[0] = 16;
            fixture.rader[0].raderFFTkernel = nullptr;
            break;
        }
        // Rejection must precede both host allocation and transfer preparation.
        state.failure = Host;
        LAYOUT_CHECK(fixture.build() == VKFFT_ERROR_MATH_FAILED);
        LAYOUT_CHECK(!state.attempts && !state.device_calls && !state.transfer_calls && !live() && !state.invalid);
        LAYOUT_CHECK(state.deleted == 1);
    }
    pfUINT total = 7;
    LAYOUT_CHECK(!vkfft_lut::add_product(total, UINT64_MAX, 2, UINT64_MAX) && total == 7);
    LAYOUT_CHECK(!vkfft_lut::add_product(total, UINT64_MAX, 1, UINT64_MAX) && total == 7);
    LAYOUT_CHECK(vkfft_lut::add_product(total, UINT64_MAX - 7, 1, UINT64_MAX) && total == UINT64_MAX);
    LAYOUT_CHECK(vkfft_lut::add_product(total, UINT64_MAX, 0, UINT64_MAX) && total == UINT64_MAX);
    const int boundary[] = {INT_MAX, 2, 2};
    total = 0;
    LAYOUT_CHECK(vkfft_lut::add_stages(boundary, 2, false, false, total) && total == INT_MAX);
    total = 0;
    LAYOUT_CHECK(!vkfft_lut::add_stages(boundary, 3, false, false, total));
    LayoutFixture fixture;
    fixture.sc().numStages = 0; // Length-one transforms need only a fallback table.
    LAYOUT_CHECK(fixture.build() == VKFFT_SUCCESS && state.uploaded == sizeof(float));
    LAYOUT_CHECK(!live() && !state.invalid);
    return 0;
}

extern "C" int vkfft_test_lut_rader_layout_and_reuse() {
    using namespace lut_fake;
    for (int precision = 0; precision < 5; ++precision) {
        LayoutFixture fixture;
        fixture.use_rader();
        auto& config = fixture.app.configuration;
        config.doublePrecision = precision == 1;
        config.doublePrecisionFloatMemory = precision == 2;
        config.quadDoubleDoublePrecision = precision == 3;
        config.quadDoubleDoublePrecisionDoubleMemory = precision == 4;
        fixture.sc().inline_rader_kernel = 1;
        for (auto& rader : fixture.rader) rader.type = 0;
        auto& first = fixture.rader[0];
        first.numStages = 2;
        first.stageRadix[0] = first.stageRadix[1] = 4;
        auto& second = fixture.rader[1];
        second.numStages = 3;
        second.stageRadix[0] = 2;
        second.stageRadix[1] = 3;
        second.stageRadix[2] = 5;
        LAYOUT_CHECK(fixture.build() == VKFFT_SUCCESS);
        LAYOUT_CHECK(fixture.sc().maxStageSumLUT == 77 && !live() && !state.invalid);
        LAYOUT_CHECK(first.RaderRadixOffsetLUT == 8 && second.RaderRadixOffsetLUT == 16);
        LAYOUT_CHECK(first.RaderRadixOffsetLUTiFFT == 44 && second.RaderRadixOffsetLUTiFFT == 52);
        const size_t lanes = precision >= 3 ? 4 : 2;
        LAYOUT_CHECK(state.uploaded == 77 * lanes * (precision ? sizeof(double) : sizeof(float)));
        // Independent angles at the end of each nested stage detect overlap and
        // differing forward/inverse offsets, beyond merely checking table size.
        const double pi = 3.1415926535897932384626433832795;
        const size_t indices[] = {15, 19, 43, 51, 61, 76};
        const double angles[] = {3*pi/8, pi/3, pi/3, 3*pi/8, 8*pi/15, 14*pi/15};
        for (size_t i = 0; i < 6; ++i) {
            for (size_t component = 0; component < 2; ++component) {
                const size_t scalar = indices[i] * lanes + component * lanes / 2;
                const double value = uploaded_scalar(precision, scalar) +
                    (lanes == 4 ? uploaded_scalar(precision, scalar + 1) : 0);
                const double expected = component ? std::sin(angles[i]) : std::cos(angles[i]);
                LAYOUT_CHECK(std::abs(value - expected) < (precision ? 1e-14 : 2e-7));
            }
        }
    }
    // Valid four-step and real-transform extensions must retain their layout.
    for (int precision = 0; precision < 5; ++precision) {
        for (int kind = 0; kind < 7; ++kind) {
            for (int inverse = 0; inverse < 2; ++inverse) {
                LayoutFixture fixture;
                auto& config = fixture.app.configuration;
                config.doublePrecision = precision == 1;
                config.doublePrecisionFloatMemory = precision == 2;
                config.quadDoubleDoublePrecision = precision == 3;
                config.quadDoubleDoublePrecisionDoubleMemory = precision == 4;
                auto& sc = fixture.sc();
                sc.actualInverse = inverse;
                const bool fourth = kind == 3 || kind == 6;
                const size_t count = kind == 0 ? 24 : fourth ? 22 : 18;
                if (kind == 0) {
                    config.useLUT_4step = 1;
                    sc.axis_upload_id = 1;
                    sc.numAxisUploads = 2;
                    sc.stageStartSize.data.i = sc.fftDim.data.i = 4;
                } else if (kind < 4) sc.performDCT = kind + 1;
                else sc.performDST = kind - 2;
                LAYOUT_CHECK(fixture.build() == VKFFT_SUCCESS && !live() && !state.invalid);
                const size_t lanes = precision >= 3 ? 4 : 2;
                LAYOUT_CHECK(state.uploaded == count * lanes * (precision ? sizeof(double) : sizeof(float)));
                if (kind) LAYOUT_CHECK(sc.startDCT3LUT.data.i == 8);
                if (fourth) LAYOUT_CHECK(sc.startDCT4LUT.data.i == 14);
                const double pi = 3.1415926535897932384626433832795;
                const double angle = kind == 0 ? 9*pi/8 : fourth ? -15*pi/64 : 9*pi/32;
                for (size_t component = 0; component < 2; ++component) {
                    const size_t scalar = (count - 1) * lanes + component * lanes / 2;
                    const double value = uploaded_scalar(precision, scalar) +
                        (lanes == 4 ? uploaded_scalar(precision, scalar + 1) : 0);
                    const double expected = component ? std::sin(angle) : std::cos(angle);
                    LAYOUT_CHECK(std::abs(value - expected) < (precision ? 1e-14 : 2e-7));
                }
            }
        }
    }
    for (int mode = 0; mode < 6; ++mode) {
        LayoutFixture fixture;
        fixture.use_rader();
        const pfUINT bytes = (17 + 31) * sizeof(uint32_t);
        if (mode < 2) {
            fixture.app.bufferRaderUintLUT[0][0] = handle<VkBuffer>(3);
            fixture.app.bufferRaderUintLUTDeviceMemory[0][0] = handle<VkDeviceMemory>(4);
            fixture.app.bufferRaderUintLUTSize[0][0] = mode == 0 ? bytes : bytes - 4;
        } else if (mode == 2) {
            fixture.rader[0].prime = fixture.rader[1].prime = INT_MAX;
        } else if (mode == 3) fixture.rader[0].prime = -1;
        else if (mode == 4) fixture.sc().numRaderPrimes = 0;
        else fixture.sc().axis_upload_id = 4;
        state.failure = Host;
        LAYOUT_CHECK(fixture.build_integer() == (mode == 0 ? VKFFT_SUCCESS : VKFFT_ERROR_MATH_FAILED));
        LAYOUT_CHECK(!state.attempts && !state.device_calls && !state.transfer_calls && !live() && !state.invalid);
        if (mode == 0) {
            LAYOUT_CHECK(fixture.rader[0].raderUintLUToffset == 0 && fixture.rader[1].raderUintLUToffset == 17);
            LAYOUT_CHECK(fixture.axis().bufferRaderUintLUT == handle<VkBuffer>(3));
            LAYOUT_CHECK(fixture.axis().bufferRaderUintLUTDeviceMemory == handle<VkDeviceMemory>(4));
            LAYOUT_CHECK(fixture.axis().bufferRaderUintLUTSize == bytes);
        }
    }
    return 0;
}

#undef LAYOUT_CHECK
#endif
