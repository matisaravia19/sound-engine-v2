// CPU-only arithmetic regressions; even very large factors allocate no FFT buffers.
#include "scheduler/arithmetic.hpp"

namespace codegen_fake {

// Independent small-prime oracle: enumerate powers until the first return to one.
int enumerated_root(int prime) {
    for (int candidate = 1; candidate < prime; ++candidate) {
        uint64_t value = 1;
        int order = 0;
        do {
            value = value * candidate % prime;
            ++order;
        } while (value != 1 && order < prime - 1);
        if (value == 1 && order == prime - 1)
            return candidate;
    }
    return 0;
}

int scheduler_primitive_roots_case() {
    // Sieve is independent of the implementation's trial primality check.
    std::array<bool, 16385> composite{};
    for (size_t i = 2; i * i < composite.size(); ++i)
        if (!composite[i])
            for (size_t multiple = i * i; multiple < composite.size(); multiple += i)
                composite[multiple] = true;
    for (int prime = 2; prime < static_cast<int>(composite.size()); ++prime) {
        int generator = -1;
        auto result = vkfft_scheduler::primitive_root(prime, generator);
        if (composite[prime]) {
            if (result != VKFFT_ERROR_MATH_FAILED || generator != -1)
                return __LINE__;
        } else if (result != VKFFT_SUCCESS || generator != enumerated_root(prime)) {
            return __LINE__;
        }
    }
    for (auto pair :
         {std::pair{65537, 3}, {786433, 10}, {754974721, 11}, {2013265921, 31}, {INT_MAX, 7}}) {
        int generator = -1;
        if (vkfft_scheduler::primitive_root(pair.first, generator) != VKFFT_SUCCESS ||
            generator != pair.second)
            return __LINE__;
    }
    for (int invalid : {INT_MIN, -1, 0, 1, INT_MAX - 1}) {
        int generator = -1;
        if (vkfft_scheduler::primitive_root(invalid, generator) != VKFFT_ERROR_MATH_FAILED ||
            generator != -1)
            return __LINE__;
    }
    return 0;
}

int scheduler_rader_counts_case() {
    struct Case {
        int part;
        pfUINT sequence;
        int prime;
        int expected; // Negative means rejection, without publishing an output.
    };
    for (auto c : {Case{INT_MAX, 17, 17, INT_MAX},
                   {1, 6975757441ull, 17, 410338673},
                   {1, static_cast<pfUINT>(INT_MAX) * 17, 17, INT_MAX},
                   {2, static_cast<pfUINT>(INT_MAX / 2) * 17, 17, INT_MAX - 1},
                   {2, static_cast<pfUINT>(INT_MAX / 2 + 1) * 17, 17, -1},
                   {1, 118587876497ull, 17, -1},
                   {1, UINT64_MAX, 17, -1},
                   {0, 17, 17, -1},
                   {-1, 17, 17, -1},
                   {1, 0, 17, -1},
                   {1, 18, 17, -1},
                   {1, 17, 0, -1},
                   {1, 17, -1, -1}}) {
        int count = -123;
        auto result = vkfft_scheduler::rader_fft_count(c.part, c.sequence, c.prime, count);
        if (result != (c.expected < 0 ? VKFFT_ERROR_MATH_FAILED : VKFFT_SUCCESS) ||
            count != (c.expected < 0 ? -123 : c.expected))
            return __LINE__;
    }
    for (auto sequence : {17ull, 6975757441ull, 118587876497ull, 41ull, 17ull * 41, 786433ull}) {
        state = {};
        {
            SchedulerFixture fixture(17, false, false, 4);
            fixture.app.configuration.fixMaxRaderPrimeFFT = 786434;
            auto& sc = fixture.plan->axes[0][0].specializationConstants;
            pfUINT remaining = sequence;
            int part = sequence == 17 ? INT_MAX : 1;
            auto result = VkFFTConstructRaderTree(&fixture.app, &sc.raderContainer, &remaining,
                                                  &sc.numRaderPrimes, part);
            if (sequence == 118587876497ull) {
                if (result != VKFFT_ERROR_MATH_FAILED || !sc.raderContainer ||
                    sc.raderContainer[0].containerFFTNum)
                    return __LINE__;
            } else {
                if (result != VKFFT_SUCCESS || remaining != 1 || !sc.numRaderPrimes)
                    return __LINE__;
                for (int i = 0; i < sc.numRaderPrimes; ++i) {
                    const auto& rader = sc.raderContainer[i];
                    if (rader.type || rader.generator != enumerated_root(rader.prime) ||
                        rader.containerFFTNum != sequence / rader.prime * static_cast<pfUINT>(part))
                        return __LINE__;
                }
                // Re-entry must preserve the axis's existing ownership.
                auto* saved = sc.raderContainer;
                remaining = sequence;
                if (VkFFTConstructRaderTree(&fixture.app, &sc.raderContainer, &remaining,
                                            &sc.numRaderPrimes, part) != VKFFT_ERROR_MATH_FAILED ||
                    saved != sc.raderContainer || remaining != sequence)
                    return __LINE__;
            }
        }
        if (live() || state.invalid)
            return __LINE__;
    }
    return 0;
}

int scheduler_factor_limits_case() {
    using Field = pfUINT VkFFTConfiguration::*;
    struct InvalidLimit {
        Field field;
        pfUINT value;
    };
    for (auto c : {InvalidLimit{&VkFFTConfiguration::fixMinRaderPrimeMult, 0},
                   {&VkFFTConfiguration::fixMinRaderPrimeMult, 1},
                   {&VkFFTConfiguration::fixMinRaderPrimeMult, 18},
                   {&VkFFTConfiguration::fixMinRaderPrimeMult, 34},
                   {&VkFFTConfiguration::fixMinRaderPrimeMult, UINT64_MAX},
                   {&VkFFTConfiguration::fixMinRaderPrimeFFT, 16},
                   {&VkFFTConfiguration::fixMinRaderPrimeFFT, static_cast<pfUINT>(INT_MAX) + 1},
                   {&VkFFTConfiguration::fixMaxRaderPrimeMult, static_cast<pfUINT>(INT_MAX) + 1},
                   {&VkFFTConfiguration::fixMaxRaderPrimeFFT, static_cast<pfUINT>(INT_MAX) + 1},
                   {&VkFFTConfiguration::fixMaxRadixBluestein, 1},
                   {&VkFFTConfiguration::fixMaxRadixBluestein, 14},
                   {&VkFFTConfiguration::fixMaxRadixBluestein, 33},
                   {&VkFFTConfiguration::fixMaxRadixBluestein, UINT64_MAX}}) {
        state = {};
        SchedulerFixture fixture(17, false, false, 4);
        fixture.app.configuration.*c.field = c.value;
        auto& sc = fixture.plan->axes[0][0].specializationConstants;
        pfUINT remaining = 17;
        if (VkFFTScheduler(&fixture.app, fixture.plan.get(), 0) != VKFFT_ERROR_MATH_FAILED ||
            VkFFTConstructRaderTree(&fixture.app, &sc.raderContainer, &remaining,
                                    &sc.numRaderPrimes, 1) != VKFFT_ERROR_MATH_FAILED ||
            sc.raderContainer || sc.numRaderPrimes || remaining != 17 || state.attempts)
            return __LINE__;
    }
    for (pfUINT minimum : {2, 3, 5, 11, 17}) {
        SchedulerFixture fixture(17, false, false, 4);
        auto& config = fixture.app.configuration;
        config.fixMinRaderPrimeMult = minimum;
        if (!vkfft_scheduler::factor_limits_valid(config) ||
            VkFFTScheduler(&fixture.app, fixture.plan.get(), 0) != VKFFT_SUCCESS)
            return __LINE__;
        config.fixMaxRaderPrimeFFT = config.fixMinRaderPrimeFFT;
        config.fixMaxRaderPrimeMult = minimum;
        if (!vkfft_scheduler::factor_limits_valid(config))
            return __LINE__;
    }
    return live() || state.invalid ? __LINE__ : 0;
}

int scheduler_input_bounds_case() {
    state = {};
    SchedulerFixture fixture(17, false, false, 4);
    auto& app = fixture.app;
    auto* plan = fixture.plan.get();
    auto& sc = plan->axes[0][0].specializationConstants;
    pfUINT sequence = 17;
    if (VkFFTConstructRaderTree(nullptr, &sc.raderContainer, &sequence, &sc.numRaderPrimes, 1) !=
            VKFFT_ERROR_MATH_FAILED ||
        VkFFTConstructRaderTree(&app, nullptr, &sequence, &sc.numRaderPrimes, 1) !=
            VKFFT_ERROR_MATH_FAILED ||
        VkFFTConstructRaderTree(&app, &sc.raderContainer, nullptr, &sc.numRaderPrimes, 1) !=
            VKFFT_ERROR_MATH_FAILED ||
        VkFFTConstructRaderTree(&app, &sc.raderContainer, &sequence, nullptr, 1) !=
            VKFFT_ERROR_MATH_FAILED)
        return __LINE__;
    for (int part : {0, -1})
        if (VkFFTConstructRaderTree(&app, &sc.raderContainer, &sequence, &sc.numRaderPrimes,
                                    part) != VKFFT_ERROR_MATH_FAILED)
            return __LINE__;
    sequence = 0;
    if (VkFFTConstructRaderTree(&app, &sc.raderContainer, &sequence, &sc.numRaderPrimes, 1) !=
        VKFFT_ERROR_MATH_FAILED)
        return __LINE__;
    sequence = 1;
    if (VkFFTConstructRaderTree(&app, &sc.raderContainer, &sequence, &sc.numRaderPrimes, 1) !=
        VKFFT_SUCCESS)
        return __LINE__;
    sequence = 17;
    app.configuration.fixMaxRaderPrimeMult = app.configuration.fixMaxRaderPrimeFFT = 17;
    if (VkFFTConstructRaderTree(&app, &sc.raderContainer, &sequence, &sc.numRaderPrimes, 1) !=
            VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH ||
        sequence != 17 || sc.raderContainer || sc.numRaderPrimes || state.attempts)
        return __LINE__;
    if (VkFFTScheduler(nullptr, plan, 0) != VKFFT_ERROR_MATH_FAILED ||
        VkFFTScheduler(&app, nullptr, 0) != VKFFT_ERROR_MATH_FAILED)
        return __LINE__;
    for (int axis : {-1, 1, VKFFT_MAX_FFT_DIMENSIONS})
        if (VkFFTScheduler(&app, plan, axis) != VKFFT_ERROR_MATH_FAILED)
            return __LINE__;
    for (pfUINT rank : {0, VKFFT_MAX_FFT_DIMENSIONS + 1}) {
        app.configuration.FFTdim = rank;
        if (VkFFTScheduler(&app, plan, 0) != VKFFT_ERROR_MATH_FAILED)
            return __LINE__;
    }
    app.configuration.FFTdim = 2;
    app.configuration.size[1] = 0;
    if (VkFFTScheduler(&app, plan, 0) != VKFFT_ERROR_MATH_FAILED || state.attempts)
        return __LINE__;
    return live() || state.invalid ? __LINE__ : 0;
}

} // namespace codegen_fake

extern "C" int vkfft_test_scheduler_primitive_roots() {
    return codegen_fake::scheduler_primitive_roots_case();
}
extern "C" int vkfft_test_scheduler_rader_counts() {
    return codegen_fake::scheduler_rader_counts_case();
}
extern "C" int vkfft_test_scheduler_factor_limits() {
    if (int line = codegen_fake::scheduler_factor_limits_case())
        return line;
    return codegen_fake::scheduler_input_bounds_case();
}
