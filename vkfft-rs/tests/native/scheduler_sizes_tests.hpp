// CPU-only scheduler regressions: large logical workloads never allocate transform buffers.
#include "scheduler/sizes.hpp"

namespace codegen_fake {

int scheduler_wide_batches_case() {
    for (bool precision : {false, true})
        for (bool real : {false, true})
            for (int rank : {1, 2, 4})
                for (int axis = 0; axis < rank; ++axis)
                    for (auto batches : {1ull, 2147483648ull, 4294967297ull})
                        for (bool internal_batches : {false, true}) {
                            state = {};
                            {
                                SchedulerFixture fixture(16, precision, false, 4);
                                auto& app = fixture.app;
                                auto& c = app.configuration;
                                c.FFTdim = rank;
                                c.size[1] = 3;
                                c.size[2] = 5;
                                c.size[3] = 7;
                                c.performR2C = real;
                                c.coordinateFeatures = 2;
                                c.numberKernels = 3;
                                if (internal_batches)
                                    app.actualNumBatches = batches;
                                else
                                    c.numberBatches = batches;
                                // These small shapes do not use the even-R2C decomposition.
                                const pfUINT first = real && axis > 0 ? 9 : 16;
                                const pfUINT rest = rank == 1 ? 1 : rank == 2 ? 3 : 105;
                                const pfUINT expected = first * rest * 6 * batches * (precision ? 16 : 8);
                                if (VkFFTScheduler(&app, fixture.plan.get(), axis) != VKFFT_SUCCESS ||
                                    fixture.temporary_bytes != expected)
                                    return __LINE__;
                            }
                            if (live() || state.invalid)
                                return __LINE__;
                        }
    return 0;
}

int scheduler_real_padding_case() {
    for (bool precision : {false, true}) {
        state = {};
        {
            SchedulerFixture fixture(65536, precision, false, 4);
            fixture.app.configuration.performR2C = 1;
            fixture.app.configuration.FFTdim = 2;
            fixture.app.configuration.size[1] = 3;
            fixture.app.configuration.numberBatches = 4294967297ull;
            const pfUINT expected = 32769ull * 3 * 4294967297ull * (precision ? 16 : 8);
            if (VkFFTScheduler(&fixture.app, fixture.plan.get(), 0) != VKFFT_SUCCESS ||
                !fixture.plan->bigSequenceEvenR2C || fixture.temporary_bytes != expected)
                return __LINE__;
        }
        if (live() || state.invalid)
            return __LINE__;
    }
    return 0;
}

int scheduler_user_workspace_case() {
    const pfUINT required = 128ull * 4294967297ull;
    for (auto capacity : {512ull, required - 1, required, required + 128}) {
        state = {};
        {
            SchedulerFixture fixture(16, false, false, 4);
            fixture.app.configuration.numberBatches = 4294967297ull;
            fixture.app.configuration.userTempBuffer = 1;
            fixture.temporary_bytes = capacity;
            const auto expected =
                capacity < required ? VKFFT_ERROR_INVALID_user_tempBuffer_too_small : VKFFT_SUCCESS;
            if (VkFFTScheduler(&fixture.app, fixture.plan.get(), 0) != expected ||
                fixture.temporary_bytes != capacity)
                return __LINE__;
        }
        if (live() || state.invalid)
            return __LINE__;
    }
    return 0;
}

// Sizing must fail before publishing an allocation requirement. Some cases
// overflow the element count; others fit in elements but overflow in bytes.
enum class SizeFailure {
    BatchProduct,
    DimensionProduct,
    FeatureProduct,
    KernelProduct,
    BatchBytes,
    DimensionBytes,
    ZeroBatches,
    ZeroFeatures,
    ZeroKernels,
    MissingSizeStorage,
};

int scheduler_size_failure_case(SizeFailure scenario) {
    state = {};
    {
        SchedulerFixture fixture(16, false, false, 4);
        auto& c = fixture.app.configuration;
        switch (scenario) {
        case SizeFailure::BatchProduct:
            c.numberBatches = 1ull << 60;
            break;
        case SizeFailure::DimensionProduct:
            c.FFTdim = 2;
            c.size[1] = 1ull << 60;
            break;
        case SizeFailure::FeatureProduct:
            c.coordinateFeatures = 1ull << 60;
            break;
        case SizeFailure::KernelProduct:
            c.numberKernels = 1ull << 60;
            break;
        case SizeFailure::BatchBytes:
            c.numberBatches = 1ull << 57;
            break;
        case SizeFailure::DimensionBytes:
            c.FFTdim = 2;
            c.size[1] = 1ull << 57;
            break;
        case SizeFailure::ZeroBatches:
            c.numberBatches = 0;
            break;
        case SizeFailure::ZeroFeatures:
            c.coordinateFeatures = 0;
            break;
        case SizeFailure::ZeroKernels:
            c.numberKernels = 0;
            break;
        case SizeFailure::MissingSizeStorage:
            c.tempBufferSize = nullptr;
            break;
        }
        fixture.temporary_bytes = 1234;
        if (VkFFTScheduler(&fixture.app, fixture.plan.get(), 0) != VKFFT_ERROR_MATH_FAILED ||
            fixture.temporary_bytes != 1234 || state.attempts)
            return __LINE__;
    }
    return live() || state.invalid ? __LINE__ : 0;
}

int scheduler_size_boundaries_case() {
    SchedulerFixture fixture(16, false, false, 4);
    auto& app = fixture.app;
    auto& plan = *fixture.plan;
    auto& shape = plan.actualFFTSizePerAxis[0];
    pfUINT output = 1234;
    for (int complex_size : {8, 16, 32}) {
        shape[0] = UINT64_MAX / complex_size;
        if (vkfft_scheduler::temporary_bytes(app, plan, 0, complex_size, output) != VKFFT_SUCCESS ||
            output != UINT64_MAX - (complex_size - 1))
            return __LINE__;
        ++shape[0];
        output = 1234;
        if (vkfft_scheduler::temporary_bytes(app, plan, 0, complex_size, output) != VKFFT_ERROR_MATH_FAILED ||
            output != 1234)
            return __LINE__;
    }
    app.configuration.performR2C = plan.bigSequenceEvenR2C = 1;
    for (auto extent : {UINT64_MAX, UINT64_MAX / 8}) {
        shape[0] = extent;
        if (vkfft_scheduler::temporary_bytes(app, plan, 0, 8, output) != VKFFT_ERROR_MATH_FAILED ||
            output != 1234)
            return __LINE__;
    }
    app.configuration.performR2C = plan.bigSequenceEvenR2C = 0;
    shape[0] = 16;
    for (int axis : {-1, 1, VKFFT_MAX_FFT_DIMENSIONS})
        if (vkfft_scheduler::temporary_bytes(app, plan, axis, 8, output) != VKFFT_ERROR_MATH_FAILED ||
            output != 1234)
            return __LINE__;
    for (int complex_size : {-1, 0})
        if (vkfft_scheduler::temporary_bytes(app, plan, 0, complex_size, output) != VKFFT_ERROR_MATH_FAILED ||
            output != 1234)
            return __LINE__;
    for (pfUINT rank : {0, VKFFT_MAX_FFT_DIMENSIONS + 1}) {
        app.configuration.FFTdim = rank;
        if (vkfft_scheduler::element_count(app, shape, false, output) != VKFFT_ERROR_MATH_FAILED ||
            output != 1234)
            return __LINE__;
    }
    return 0;
}

} // namespace codegen_fake

extern "C" int vkfft_test_scheduler_wide_sizes() {
    if (int line = codegen_fake::scheduler_wide_batches_case())
        return line;
    if (int line = codegen_fake::scheduler_real_padding_case())
        return line;
    return codegen_fake::scheduler_user_workspace_case();
}

extern "C" int vkfft_test_scheduler_size_failures() {
    using codegen_fake::SizeFailure;
    for (auto scenario : {SizeFailure::BatchProduct, SizeFailure::DimensionProduct,
                          SizeFailure::FeatureProduct, SizeFailure::KernelProduct, SizeFailure::BatchBytes,
                          SizeFailure::DimensionBytes, SizeFailure::ZeroBatches, SizeFailure::ZeroFeatures,
                          SizeFailure::ZeroKernels, SizeFailure::MissingSizeStorage})
        if (int line = codegen_fake::scheduler_size_failure_case(scenario))
            return line;
    return codegen_fake::scheduler_size_boundaries_case();
}
