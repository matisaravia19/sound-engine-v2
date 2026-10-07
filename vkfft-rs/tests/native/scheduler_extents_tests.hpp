// CPU-only input-boundary tests; large logical extents never reach a GPU allocator.
#include "scheduler_extents.hpp"

namespace codegen_fake {

int transform_extent_boundaries_case() {
    VkFFTConfiguration c{};
    const auto maximum = (std::numeric_limits<pfUINT>::max)();
    for (bool sine : {false, true}) {
        c.performDCT = sine ? 0 : 1;
        c.performDST = sine ? 1 : 0;
        for (pfUINT input = sine ? 1 : 2; input <= 128; ++input) {
            pfUINT result = input;
            const auto expected = sine ? 2 * input + 2 : 2 * input - 2;
            if (vkfft_scheduler::real_transform_length(c, input, result) != VKFFT_SUCCESS ||
                result != expected)
                return __LINE__;
        }
        const auto largest = sine ? maximum / 2 - 1 : maximum / 2 + 1;
        pfUINT result = 123;
        if (vkfft_scheduler::real_transform_length(c, largest, result) != VKFFT_SUCCESS ||
            result != maximum - 1)
            return __LINE__;
        for (auto input : {pfUINT(0), largest + 1, maximum}) {
            result = 123;
            if (vkfft_scheduler::real_transform_length(c, input, result) != VKFFT_ERROR_MATH_FAILED ||
                result != 123)
                return __LINE__;
        }
    }
    c = {};
    pfUINT split = 32;
    if (vkfft_scheduler::real_transform_length(c, 64, split) != VKFFT_SUCCESS || split != 32)
        return __LINE__;
    for (bool sine : {false, true}) {
        c.performDCT = sine ? 0 : 4;
        c.performDST = sine ? 4 : 0;
        for (pfUINT input : {16, 17}) {
            auto result = input;
            if (vkfft_scheduler::real_transform_length(c, input, result) != VKFFT_SUCCESS ||
                result != (input % 2 ? input : input / 2))
                return __LINE__;
        }
    }
    pfUINT minimum = 123;
    for (pfUINT length : {pfUINT(1), pfUINT(17), maximum / 2 + 1})
        if (vkfft_scheduler::bluestein_minimum(length, minimum) != VKFFT_SUCCESS ||
            minimum != length + (length - 1))
            return __LINE__;
    for (auto length : {pfUINT(0), maximum / 2 + 2, maximum}) {
        minimum = 123;
        if (vkfft_scheduler::bluestein_minimum(length, minimum) != VKFFT_ERROR_MATH_FAILED || minimum != 123)
            return __LINE__;
    }
    return 0;
}

int scheduler_transform_extent_case() {
    const struct {
        bool sine;
        pfUINT length;
        VkFFTResult expected;
    } cases[] = {{false, 1, VKFFT_ERROR_MATH_FAILED},
                 {false, UINT64_MAX, VKFFT_ERROR_MATH_FAILED},
                 {true, 1, VKFFT_SUCCESS},
                 {true, UINT64_MAX / 2, VKFFT_ERROR_MATH_FAILED},
                 {true, UINT64_MAX, VKFFT_ERROR_MATH_FAILED}};
    for (const auto& test : cases) {
        state = {};
        {
            SchedulerFixture fixture(1, false, false, 1);
            fixture.app.configuration.size[0] = test.length;
            fixture.app.configuration.performDCT = test.sine ? 0 : 1;
            fixture.app.configuration.performDST = test.sine ? 1 : 0;
            if (VkFFTScheduler(&fixture.app, fixture.plan.get(), 0) != test.expected)
                return __LINE__;
            if (test.expected != VKFFT_SUCCESS && (state.attempts || fixture.temporary_bytes))
                return __LINE__;
        }
        if (live() || state.invalid)
            return __LINE__;
    }
    return transform_extent_boundaries_case();
}

int padding_table_selection_case() {
    VkFFTConfiguration c{};
    pfUINT primes[] = {17, 43};
    pfUINT pads[] = {84, 128};
    c.primeSizes = primes;
    c.paddedSizes = pads;
    for (bool automatic : {false, true}) {
        c.autoCustomBluesteinPaddingPattern = automatic ? 2 : 0;
        c.useCustomBluesteinPaddingPattern = automatic ? 0 : 2;
        for (pfUINT length : {1, 16, 17, 42, 43, 64, 65}) {
            pfUINT minimum = 123, padded = 456;
            bool selected = false;
            const pfUINT expected = length < 17 || length > 64 ? 2 * length - 1 : length < 43 ? 84 : 128;
            if (vkfft_scheduler::initial_bluestein_padding(c, length, minimum, padded, selected) !=
                    VKFFT_SUCCESS ||
                minimum != 2 * length - 1 || padded != expected || selected != (length >= 17 && length <= 64))
                return __LINE__;
        }
    }
    for (pfUINT padding : {0, 16, 34}) {
        pads[0] = padding;
        pfUINT minimum = 123, padded = 456;
        bool selected = true;
        if (vkfft_scheduler::initial_bluestein_padding(c, 17, minimum, padded, selected) == VKFFT_SUCCESS ||
            minimum != 123 || padded != 456 || !selected)
            return __LINE__;
    }
    // An explicit radix limit bypasses the optional table, as in upstream VkFFT.
    c.fixMaxRadixBluestein = 2;
    c.primeSizes = nullptr;
    pfUINT minimum = 123, padded = 456;
    bool selected = true;
    if (vkfft_scheduler::initial_bluestein_padding(c, 17, minimum, padded, selected) != VKFFT_SUCCESS ||
        minimum != 33 || padded != 33 || selected)
        return __LINE__;
    return 0;
}

enum class PaddingScenario {
    ForcedShort,
    ForcedUnsupported,
    ForcedValid,
    TableZero,
    TableShort,
    TableUnsupported,
    TableValid,
    MissingPrimes,
    MissingPads,
    ExcessCount,
    Unsorted,
    Duplicate,
    ZeroPrime,
    FinalEntryFallback,
    RealRetryForced,
    RealRetryTable,
};

int scheduler_padding_case(PaddingScenario scenario, bool precision) {
    state = {};
    {
        SchedulerFixture fixture(17, precision, false, 1);
        auto& c = fixture.app.configuration;
        c.fixMaxRaderPrimeMult = c.fixMaxRaderPrimeFFT = 17;
        pfUINT primes[] = {17, 3000, 5000};
        pfUINT pads[] = {64, 8192, 16384};
        c.useCustomBluesteinPaddingPattern = 3;
        c.primeSizes = primes;
        c.paddedSizes = pads;
        auto expected = VKFFT_ERROR_MATH_FAILED;
        pfUINT expected_length = 0;
        switch (scenario) {
        case PaddingScenario::ForcedShort:
            c.forceBluesteinSequenceSize = 32;
            break;
        case PaddingScenario::ForcedUnsupported:
            c.forceBluesteinSequenceSize = 37;
            expected = VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;
            break;
        case PaddingScenario::ForcedValid:
            c.forceBluesteinSequenceSize = 33;
            expected_length = 33;
            expected = VKFFT_SUCCESS;
            break;
        case PaddingScenario::TableZero:
            pads[0] = 0;
            break;
        case PaddingScenario::TableShort:
            pads[0] = 32;
            break;
        case PaddingScenario::TableUnsupported:
            pads[0] = 37;
            expected = VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;
            break;
        case PaddingScenario::TableValid:
            pads[0] = expected_length = 33;
            expected = VKFFT_SUCCESS;
            break;
        case PaddingScenario::MissingPrimes:
            c.primeSizes = nullptr;
            expected = VKFFT_ERROR_EMPTY_useCustomBluesteinPaddingPattern_arrays;
            break;
        case PaddingScenario::MissingPads:
            c.paddedSizes = nullptr;
            expected = VKFFT_ERROR_EMPTY_useCustomBluesteinPaddingPattern_arrays;
            break;
        case PaddingScenario::ExcessCount:
            c.useCustomBluesteinPaddingPattern = UINT64_MAX;
            break;
        case PaddingScenario::Unsorted:
            primes[1] = 16;
            break;
        case PaddingScenario::Duplicate:
            primes[1] = 17;
            break;
        case PaddingScenario::ZeroPrime:
            primes[0] = 0;
            break;
        case PaddingScenario::FinalEntryFallback:
            c.size[0] = 257;
            c.useCustomBluesteinPaddingPattern = 1;
            expected = VKFFT_SUCCESS;
            break;
        case PaddingScenario::RealRetryForced:
        case PaddingScenario::RealRetryTable:
            c.size[0] = 4006;
            c.performR2C = 1;
            pads[0] = 4096;
            if (scenario == PaddingScenario::RealRetryForced) {
                c.useCustomBluesteinPaddingPattern = 0;
                c.forceBluesteinSequenceSize = 4096;
            }
            expected_length = 4096;
            expected = VKFFT_SUCCESS;
            break;
        }
        const auto result = VkFFTScheduler(&fixture.app, fixture.plan.get(), 0);
        if (result != expected)
            return __LINE__;
        if (expected == VKFFT_SUCCESS) {
            const auto length = fixture.plan->actualFFTSizePerAxis[0][0];
            if (expected_length && length != expected_length)
                return __LINE__;
            if (!expected_length && length < 513)
                return __LINE__;
            if (c.performR2C && !fixture.plan->bigSequenceEvenR2C)
                return __LINE__;
            const auto bytes = (length + (c.performR2C ? 1 : 0)) * (precision ? 16 : 8);
            if (fixture.temporary_bytes != bytes)
                return __LINE__;
        } else if (state.attempts || fixture.temporary_bytes) {
            return __LINE__;
        }
    }
    return live() || state.invalid ? __LINE__ : 0;
}

} // namespace codegen_fake

extern "C" int vkfft_test_scheduler_transform_extents() {
    return codegen_fake::scheduler_transform_extent_case();
}

extern "C" int vkfft_test_scheduler_padding_extents() {
    using codegen_fake::PaddingScenario;
    for (bool precision : {false, true})
        for (auto scenario :
             {PaddingScenario::ForcedShort, PaddingScenario::ForcedUnsupported, PaddingScenario::ForcedValid,
              PaddingScenario::TableZero, PaddingScenario::TableShort, PaddingScenario::TableUnsupported,
              PaddingScenario::TableValid, PaddingScenario::MissingPrimes, PaddingScenario::MissingPads,
              PaddingScenario::ExcessCount, PaddingScenario::Unsorted, PaddingScenario::Duplicate,
              PaddingScenario::ZeroPrime, PaddingScenario::FinalEntryFallback,
              PaddingScenario::RealRetryForced, PaddingScenario::RealRetryTable})
            if (int line = codegen_fake::scheduler_padding_case(scenario, precision))
                return line;
    return codegen_fake::padding_table_selection_case();
}
