// Independent candidate references and full scheduler checks; no GPU buffers are allocated.
#include "scheduler/padding.hpp"

namespace codegen_fake {

int padding_search_reference_case() {
    constexpr size_t limit = 16384;
    for (int radix = 2; radix <= 13; ++radix) {
        // This dynamic-programming reference uses divisibility, not prime-power enumeration.
        std::array<bool, limit + 1> smooth{};
        smooth[1] = true;
        for (size_t value = 2; value <= limit; ++value)
            for (int divisor = 2; divisor <= radix; ++divisor)
                if (value % divisor == 0 && smooth[value / divisor]) {
                    smooth[value] = true;
                    break;
                }
        size_t expected = 1;
        for (pfUINT minimum = 1; minimum <= limit / 2; ++minimum) {
            if (expected < minimum)
                expected = static_cast<size_t>(minimum);
            while (!smooth[expected])
                ++expected;
            pfUINT result = 123;
            if (vkfft_scheduler::next_smooth_padding(minimum, radix, result) != VKFFT_SUCCESS ||
                result != expected)
                return __LINE__;
        }
    }
    return 0;
}

int padding_search_boundaries_case() {
    const auto maximum = (std::numeric_limits<pfUINT>::max)();
    const pfUINT high_bit = pfUINT(1) << 63;
    pfUINT output = 123;
    for (pfUINT value : {pfUINT(0), high_bit + 1, maximum})
        if (vkfft_scheduler::power_of_two_ceiling(value, output) || output != 123)
            return __LINE__;
    for (int exponent = 0; exponent < 64; ++exponent) {
        const auto value = pfUINT(1) << exponent;
        if (!vkfft_scheduler::power_of_two_ceiling(value, output) || output != value)
            return __LINE__;
        if (value > 2 &&
            (!vkfft_scheduler::power_of_two_ceiling(value - 1, output) || output != value))
            return __LINE__;
    }
    for (int radix : {2, 3, 7, 13}) {
        if (vkfft_scheduler::next_smooth_padding(high_bit, radix, output) != VKFFT_SUCCESS ||
            output != high_bit)
            return __LINE__;
        output = 123;
        if (vkfft_scheduler::next_smooth_padding(maximum, radix, output) !=
                VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH ||
            output != 123)
            return __LINE__;
    }
    for (int radix : {0, 1, 14})
        if (vkfft_scheduler::next_smooth_padding(17, radix, output) != VKFFT_ERROR_MATH_FAILED ||
            output != 123)
            return __LINE__;
    if (vkfft_scheduler::next_smooth_padding(0, 7, output) != VKFFT_ERROR_MATH_FAILED ||
        output != 123 ||
        vkfft_scheduler::next_smooth_padding(high_bit + 1, 2, output) !=
            VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH ||
        output != 123)
        return __LINE__;
    for (pfUINT left : {pfUINT(0), pfUINT(1), pfUINT(2), pfUINT(32), maximum})
        for (pfUINT right : {pfUINT(0), pfUINT(1), pfUINT(2), pfUINT(32), maximum})
            for (pfUINT threshold : {pfUINT(0), pfUINT(1), pfUINT(128), maximum}) {
                const bool overflow = right && left > maximum / right;
                const bool expected = overflow || left * right >= threshold;
                if (vkfft_scheduler::product_at_least(left, right, threshold) != expected)
                    return __LINE__;
            }
    return 0;
}

// Keep the old sequential search only as a small-input oracle. Its floating-point
// rounding and integer trial division are safe within this test's bounded domain.
template <class Accept>
pfUINT reference_default_padding(pfUINT length, pfUINT minimum, int shared_limit, Accept accept) {
    for (auto candidate = minimum;; ++candidate) {
        const auto rounded = static_cast<pfUINT>(std::pow(2, std::ceil(std::log2(candidate))));
        if (length < 128 ||
            (candidate >= rounded * 0.75 && (rounded <= static_cast<pfUINT>(shared_limit) ||
                                             minimum > static_cast<pfUINT>(shared_limit))))
            candidate = rounded;
        int factors[33]{};
        auto remainder = candidate;
        for (int divisor = 2; divisor <= 7; ++divisor)
            while (remainder % divisor == 0) {
                remainder /= divisor;
                ++factors[divisor];
            }
        bool accepted = false;
        if (remainder == 1 && accept(candidate, factors, accepted) == VKFFT_SUCCESS && accepted)
            return candidate;
    }
}

int padding_register_policy_case() {
    for (int precision : {0, 1, 2}) {
        SchedulerFixture fixture(17, precision == 1, false, 1);
        fixture.app.configuration.quadDoubleDoublePrecision = precision == 2;
        auto accept = [&](pfUINT candidate, int(&factors)[33], bool& accepted) {
            int registers[33], maximum = 0, minimum = 10000000, good = 0;
            auto result = VkFFTGetRegistersPerThread(&fixture.app, candidate, 0, 0, 0, factors,
                                                     registers, &maximum, &minimum, &good);
            accepted = good != 0;
            return result;
        };
        for (int shared : {8, 32, 128, 4096})
            for (pfUINT length = 17; length <= 2048; length += 7) {
                const auto minimum = 2 * length - 1;
                pfUINT padded = 123;
                if (vkfft_scheduler::automatic_bluestein_padding(length, minimum, 0, shared, accept,
                                                                 padded) != VKFFT_SUCCESS ||
                    padded != reference_default_padding(length, minimum, shared, accept))
                    return __LINE__;
            }
        for (int exponent : {31, 32, 63}) {
            int factors[33]{};
            factors[2] = exponent;
            bool accepted = false;
            if (accept(pfUINT(1) << exponent, factors, accepted) != VKFFT_SUCCESS || !accepted)
                return __LINE__;
        }
    }
    pfUINT padded = 123;
    int calls = 0;
    auto fail = [&](pfUINT, int(&)[33], bool&) {
        ++calls;
        return VKFFT_ERROR_MALLOC_FAILED;
    };
    if (vkfft_scheduler::automatic_bluestein_padding(257, 513, 0, 4096, fail, padded) !=
            VKFFT_ERROR_MALLOC_FAILED ||
        padded != 123 || calls != 1)
        return __LINE__;
    if (vkfft_scheduler::automatic_bluestein_padding(257, 513, 2, 4096, fail, padded) !=
            VKFFT_SUCCESS ||
        padded != 1024 || calls != 1)
        return __LINE__;
    calls = 0;
    auto reject_twice = [&](pfUINT candidate, int(&)[33], bool& accepted) {
        const pfUINT expected[] = {64, 128, 256};
        if (calls >= 3 || candidate != expected[calls])
            return VKFFT_ERROR_MATH_FAILED;
        accepted = ++calls == 3;
        return VKFFT_SUCCESS;
    };
    if (vkfft_scheduler::automatic_bluestein_padding(17, 33, 0, 4096, reject_twice, padded) !=
            VKFFT_SUCCESS ||
        padded != 256 || calls != 3)
        return __LINE__;
    return 0;
}

int scheduler_wide_padding_case() {
    for (pfUINT length : {1073741827ull, 1073741828ull, 2147483651ull})
        for (bool precision : {false, true})
            for (bool strided : {false, true})
                for (bool real : {false, true})
                    for (int radix : {0, 2, 3, 5, 7, 13}) {
                        state = {};
                        {
                            SchedulerFixture fixture(17, precision, strided, 1);
                            auto& c = fixture.app.configuration;
                            c.size[0] = length;
                            c.performR2C = real;
                            c.fixMaxRaderPrimeMult = c.fixMaxRaderPrimeFFT = 17;
                            c.fixMaxRadixBluestein = radix;
                            const auto original_config = c;
                            const auto result = VkFFTScheduler(&fixture.app, fixture.plan.get(), 0);
                            if (result != VKFFT_SUCCESS &&
                                result != VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH)
                                return __LINE__;
                            const auto& plan = *fixture.plan;
                            auto padded = plan.actualFFTSizePerAxis[0][0];
                            const auto effective = plan.bigSequenceEvenR2C ? length / 2 : length;
                            if (!fixture.app.useBluesteinFFT[0] || padded < 2 * effective - 1)
                                return __LINE__;
                            int factors[33];
                            if (!vkfft_scheduler::padding_factors(padded, radix ? radix : 7,
                                                                  factors))
                                return __LINE__;
                            if (radix == 2) {
                                pfUINT expected = 0;
                                if (!vkfft_scheduler::power_of_two_ceiling(2 * effective - 1,
                                                                           expected) ||
                                    padded != expected)
                                    return __LINE__;
                            }
                            // Bypass search with the selected size: some shapes exceed the
                            // native scheduler's three-pass limit even with valid padding.
                            SchedulerFixture control(17, precision, strided, 1);
                            control.app.configuration = original_config;
                            auto& reference = control.app.configuration;
                            pfUINT first = 1;
                            reference.tempBufferSize = &control.temporary_bytes;
                            reference.fixMaxRadixBluestein = 0;
                            reference.useCustomBluesteinPaddingPattern = 1;
                            reference.primeSizes = &first;
                            reference.paddedSizes = &padded;
                            if (VkFFTScheduler(&control.app, control.plan.get(), 0) != result)
                                return __LINE__;
                            if (result == VKFFT_SUCCESS) {
                                if (!plan.numAxisUploads[0] || plan.numAxisUploads[0] > 3)
                                    return __LINE__;
                                pfUINT product = 1;
                                for (pfUINT upload = 0; upload < plan.numAxisUploads[0]; ++upload) {
                                    product *= plan.axisSplit[0][upload];
                                    const auto& sc = plan.axes[0][upload].specializationConstants;
                                    pfUINT stages = 1;
                                    for (int stage = 0; stage < sc.numStages; ++stage)
                                        stages *= sc.stageRadix[stage];
                                    if (stages != plan.axisSplit[0][upload] ||
                                        sc.registers_per_thread <= 0)
                                        return __LINE__;
                                }
                                const auto bytes = (padded + (plan.bigSequenceEvenR2C ? 1 : 0)) *
                                                   (precision ? 16 : 8);
                                if (product != padded || fixture.temporary_bytes != bytes)
                                    return __LINE__;
                            } else if (fixture.temporary_bytes || plan.numAxisUploads[0] ||
                                       live()) {
                                return __LINE__;
                            }
                        }
                        if (live() || state.invalid)
                            return __LINE__;
                    }
    return 0;
}

} // namespace codegen_fake

extern "C" int vkfft_test_padding_search() {
    if (int line = codegen_fake::padding_search_reference_case())
        return line;
    return codegen_fake::padding_search_boundaries_case();
}

extern "C" int vkfft_test_padding_register_policy() {
    return codegen_fake::padding_register_policy_case();
}

extern "C" int vkfft_test_scheduler_wide_padding() {
    return codegen_fake::scheduler_wide_padding_case();
}
