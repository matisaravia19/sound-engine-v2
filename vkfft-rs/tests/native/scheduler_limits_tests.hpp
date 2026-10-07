// CPU-only resource-limit regressions against the actual scheduler.
#include "scheduler_limits.hpp"

namespace codegen_fake {

bool same_scheduler_layout(const SchedulerFixture& left, const SchedulerFixture& right, int axis) {
    const auto& a = *left.plan;
    const auto& b = *right.plan;
    if (left.temporary_bytes != right.temporary_bytes || a.numAxisUploads[axis] != b.numAxisUploads[axis] ||
        a.bigSequenceEvenR2C != b.bigSequenceEvenR2C)
        return false;
    for (int i = 0; i < VKFFT_MAX_FFT_DIMENSIONS; ++i)
        if (a.actualFFTSizePerAxis[axis][i] != b.actualFFTSizePerAxis[axis][i])
            return false;
    for (pfUINT upload = 0; upload < a.numAxisUploads[axis]; ++upload) {
        const auto& x = a.axes[axis][upload].specializationConstants;
        const auto& y = b.axes[axis][upload].specializationConstants;
        if (a.axisSplit[axis][upload] != b.axisSplit[axis][upload] || x.numStages != y.numStages ||
            x.registers_per_thread != y.registers_per_thread ||
            x.min_registers_per_thread != y.min_registers_per_thread || x.registerBoost != y.registerBoost)
            return false;
        for (int stage = 0; stage < x.numStages; ++stage)
            if (x.stageRadix[stage] != y.stageRadix[stage])
                return false;
    }
    return true;
}

enum class LimitFailure {
    ZeroCoalescing,
    WideCoalescing,
    CoalescingExceedsShared,
    ZeroThreads,
    WideThreads,
    ZeroWarp,
    WideWarp,
    ZeroShared,
    WideShared,
    ZeroPowerShared,
    NonPowerShared,
    PowerSharedExceedsTotal,
    InsufficientComplexStorage,
    ZeroBoost,
    ZeroFourStepBoost,
    WideBandwidthBoost,
};

int scheduler_invalid_limit_case(LimitFailure scenario) {
    state = {};
    {
        SchedulerFixture fixture(17, false, false, 1);
        auto& c = fixture.app.configuration;
        c.coalescedMemory = 1;
        const pfUINT too_wide = pfUINT(1) << 31;
        switch (scenario) {
        case LimitFailure::ZeroCoalescing:
            c.coalescedMemory = 0;
            break;
        case LimitFailure::WideCoalescing:
            c.coalescedMemory = too_wide;
            break;
        case LimitFailure::CoalescingExceedsShared:
            c.coalescedMemory = 65536;
            break;
        case LimitFailure::ZeroThreads:
            c.maxThreadsNum = 0;
            break;
        case LimitFailure::WideThreads:
            c.maxThreadsNum = too_wide;
            break;
        case LimitFailure::ZeroWarp:
            c.warpSize = 0;
            break;
        case LimitFailure::WideWarp:
            c.warpSize = too_wide;
            break;
        case LimitFailure::ZeroShared:
            c.sharedMemorySize = 0;
            break;
        case LimitFailure::WideShared:
            c.sharedMemorySize = too_wide;
            break;
        case LimitFailure::ZeroPowerShared:
            c.sharedMemorySizePow2 = 0;
            break;
        case LimitFailure::NonPowerShared:
            c.sharedMemorySizePow2 = 32767;
            break;
        case LimitFailure::PowerSharedExceedsTotal:
            c.sharedMemorySizePow2 = 65536;
            break;
        case LimitFailure::InsufficientComplexStorage:
            c.sharedMemorySizePow2 = 4;
            break;
        case LimitFailure::ZeroBoost:
            c.registerBoost = 0;
            break;
        case LimitFailure::ZeroFourStepBoost:
            c.registerBoost4Step = 0;
            break;
        case LimitFailure::WideBandwidthBoost:
            c.performBandwidthBoost = too_wide;
            break;
        }
        const auto coalesced = c.coalescedMemory;
        fixture.temporary_bytes = 123;
        if (VkFFTScheduler(&fixture.app, fixture.plan.get(), 0) != VKFFT_ERROR_MATH_FAILED ||
            c.coalescedMemory != coalesced || fixture.temporary_bytes != 123 || state.attempts ||
            fixture.plan->numAxisUploads[0] || fixture.plan->actualFFTSizePerAxis[0][0])
            return __LINE__;
    }
    return live() || state.invalid ? __LINE__ : 0;
}

int scheduler_coalescing_case() {
    for (int precision : {0, 1, 2}) {
        const pfUINT complex_size = pfUINT(8) << precision;
        for (pfUINT hint = 1; hint <= complex_size; ++hint)
            for (int length : {16, 17, 257, 65536})
                for (int mode : {0, 1, 2}) {
                    state = {};
                    {
                        SchedulerFixture actual(length, precision == 1, mode == 1, 4);
                        SchedulerFixture control(length, precision == 1, mode == 1, 4);
                        const int axis = mode == 2 ? 1 : 0;
                        for (auto* fixture : {&actual, &control}) {
                            auto& c = fixture->app.configuration;
                            c.quadDoubleDoublePrecision = precision == 2;
                            if (axis) {
                                c.FFTdim = 2;
                                c.size[0] = 16;
                                c.size[1] = length;
                            }
                        }
                        actual.app.configuration.coalescedMemory = hint;
                        control.app.configuration.coalescedMemory = complex_size;
                        if (VkFFTScheduler(&actual.app, actual.plan.get(), axis) != VKFFT_SUCCESS ||
                            VkFFTScheduler(&control.app, control.plan.get(), axis) != VKFFT_SUCCESS ||
                            actual.app.configuration.coalescedMemory != complex_size ||
                            !same_scheduler_layout(actual, control, axis))
                            return __LINE__;
                    }
                    if (live() || state.invalid)
                        return __LINE__;
                }
    }
    return 0;
}

int scheduler_resource_fallback_case() {
    for (bool memory_limited : {false, true}) {
        state = {};
        {
            SchedulerFixture fixture(17, false, false, 1);
            auto& c = fixture.app.configuration;
            c.fixMaxRaderPrimeFFT = 17;
            if (memory_limited) {
                c.sharedMemorySize = c.sharedMemorySizePow2 = 128;
                c.coalescedMemory = 8;
            } else {
                c.maxThreadsNum = 1;
                c.coalescedMemory = 1024;
            }
            if (VkFFTScheduler(&fixture.app, fixture.plan.get(), 0) != VKFFT_SUCCESS ||
                !fixture.app.useBluesteinFFT[0] || c.fixMaxRaderPrimeMult > 17 ||
                fixture.temporary_bytes != 512)
                return __LINE__;
        }
        if (live() || state.invalid)
            return __LINE__;
    }
    // Capacity one may serve a trivial axis, but cannot be the logarithm base for a larger strided FFT.
    state = {};
    {
        SchedulerFixture fixture(64, false, true, 1);
        fixture.app.configuration.sharedMemorySize = fixture.app.configuration.sharedMemorySizePow2 = 32;
        if (VkFFTScheduler(&fixture.app, fixture.plan.get(), 0) != VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH ||
            fixture.temporary_bytes || state.attempts)
            return __LINE__;
    }
    return live() || state.invalid ? __LINE__ : 0;
}

int scheduler_boost_hint_case() {
    for (bool precision : {false, true})
        for (bool strided : {false, true})
            for (bool four_step : {false, true})
                for (int length : {16, 64, 81, 4099, 65536, 65539})
                    for (pfUINT hint : {65536ull, UINT64_MAX}) {
                        state = {};
                        {
                            SchedulerFixture actual(length, precision, strided, 1);
                            SchedulerFixture control(length, precision, strided, 1);
                            // No admissible boost can exceed sqrt(the padded length). For
                            // this small-input matrix 1024 is a safe, independently bounded ceiling.
                            if (four_step) {
                                actual.app.configuration.registerBoost4Step = hint;
                                control.app.configuration.registerBoost4Step = 1024;
                            } else {
                                actual.app.configuration.registerBoost = hint;
                                control.app.configuration.registerBoost = 1024;
                            }
                            actual.app.configuration.fixMaxRaderPrimeFFT =
                                control.app.configuration.fixMaxRaderPrimeFFT = 17;
                            if (VkFFTScheduler(&actual.app, actual.plan.get(), 0) != VKFFT_SUCCESS ||
                                VkFFTScheduler(&control.app, control.plan.get(), 0) != VKFFT_SUCCESS ||
                                !same_scheduler_layout(actual, control, 0))
                                return __LINE__;
                        }
                        if (live() || state.invalid)
                            return __LINE__;
                    }
    // A modest four-step boost also used to feed a zero quotient into log2.
    state = {};
    {
        SchedulerFixture fixture(4099, false, false, 1);
        fixture.app.configuration.fixMaxRaderPrimeFFT = 17;
        fixture.app.configuration.registerBoost4Step = 4;
        if (VkFFTScheduler(&fixture.app, fixture.plan.get(), 0) != VKFFT_SUCCESS || !fixture.temporary_bytes)
            return __LINE__;
    }
    return live() || state.invalid ? __LINE__ : 0;
}

} // namespace codegen_fake

extern "C" int vkfft_test_scheduler_invalid_resources() {
    using codegen_fake::LimitFailure;
    for (auto scenario :
         {LimitFailure::ZeroCoalescing, LimitFailure::WideCoalescing, LimitFailure::CoalescingExceedsShared,
          LimitFailure::ZeroThreads, LimitFailure::WideThreads, LimitFailure::ZeroWarp,
          LimitFailure::WideWarp, LimitFailure::ZeroShared, LimitFailure::WideShared,
          LimitFailure::ZeroPowerShared, LimitFailure::NonPowerShared, LimitFailure::PowerSharedExceedsTotal,
          LimitFailure::InsufficientComplexStorage, LimitFailure::ZeroBoost, LimitFailure::ZeroFourStepBoost,
          LimitFailure::WideBandwidthBoost})
        if (int line = codegen_fake::scheduler_invalid_limit_case(scenario))
            return line;
    return 0;
}

extern "C" int vkfft_test_scheduler_coalescing_limits() {
    if (int line = codegen_fake::scheduler_coalescing_case())
        return line;
    return codegen_fake::scheduler_resource_fallback_case();
}

extern "C" int vkfft_test_scheduler_boost_hints() { return codegen_fake::scheduler_boost_hint_case(); }
