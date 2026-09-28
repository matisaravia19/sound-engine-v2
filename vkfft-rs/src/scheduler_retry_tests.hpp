// Host-only regressions for retry ownership; the large natural fallback allocates no GPU buffers.
#include "scheduler_retry.hpp"

namespace codegen_fake {

// A byte snapshot borrows resource pointers for comparison only; it never tears down axes.
std::unique_ptr<VkFFTPlan> snapshot_schedule(const VkFFTPlan& plan) {
    auto snapshot = std::make_unique<VkFFTPlan>();
    std::memcpy(snapshot.get(), &plan, sizeof(plan));
    return snapshot;
}

bool unchanged_other_axes(const VkFFTPlan& plan, const VkFFTPlan& saved, int changed_axis) {
    for (int axis = 0; axis < VKFFT_MAX_FFT_DIMENSIONS; ++axis) {
        if (axis == changed_axis)
            continue;
        if (plan.numAxisUploads[axis] != saved.numAxisUploads[axis] ||
            plan.actualPerformR2CPerAxis[axis] != saved.actualPerformR2CPerAxis[axis] ||
            std::memcmp(plan.axes[axis], saved.axes[axis], sizeof(plan.axes[axis])) ||
            std::memcmp(plan.inverseBluesteinAxes[axis], saved.inverseBluesteinAxes[axis],
                        sizeof(plan.inverseBluesteinAxes[axis])) ||
            std::memcmp(plan.actualFFTSizePerAxis[axis], saved.actualFFTSizePerAxis[axis],
                        sizeof(plan.actualFFTSizePerAxis[axis])) ||
            std::memcmp(plan.axisSplit[axis], saved.axisSplit[axis], sizeof(plan.axisSplit[axis])))
            return false;
    }
    return true;
}

int natural_scheduler_retry_case(bool inverse, bool precision) {
    state = {};
    {
        SchedulerFixture fixture(17, precision, false, 4);
        auto& app = fixture.app;
        auto* plan = fixture.plan.get();
        app.configuration.FFTdim = 2;
        app.configuration.size[1] = 89ull * 89 * 89 * 89;
        if (inverse)
            app.localFFTPlan_inverse = plan;
        else
            app.localFFTPlan = plan;
        auto result = VkFFTScheduler(&app, plan, 0);
        if (result != VKFFT_SUCCESS || !plan->numAxisUploads[0] || !live())
            return __LINE__;
        const auto saved = snapshot_schedule(*plan);
        const auto old_fft = app.configuration.fixMaxRaderPrimeFFT;
        const auto old_mult = app.configuration.fixMaxRaderPrimeMult;
        int calls = 0;
        VkFFTResult first_result = VKFFT_SUCCESS;
        result = vkfft_scheduler::schedule_axis(&app, plan, 1, [&](auto* a, auto* p, int axis) {
            ++calls;
            auto status = VkFFTScheduler(a, p, axis);
            if (calls == 1)
                first_result = status;
            return status;
        });
        if (result != VKFFT_SUCCESS || first_result != VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH || calls != 2 ||
            !plan->numAxisUploads[1] || !app.useBluesteinFFT[1] || !unchanged_other_axes(*plan, *saved, 1) ||
            app.configuration.fixMaxRaderPrimeFFT != old_fft ||
            app.configuration.fixMaxRaderPrimeMult != old_mult)
            return __LINE__;
    }
    return live() || state.invalid ? __LINE__ : 0;
}

// Exit paths are named so allocation and exception expectations stay explicit.
enum class RetryScenario {
    FirstSuccess,
    FirstError,
    RetrySuccess,
    RetryError,
    RetryException,
    RetryUnsupported,
    FirstAllocationFailure,
    RetryAllocationFailure,
};

// Partial allocations can sit in upload slots not yet published in the plan count.
int scheduler_retry_exit_case(int target, RetryScenario scenario) {
    state = {};
    {
        SchedulerFixture fixture(17, false, false, 4);
        auto& app = fixture.app;
        auto* plan = fixture.plan.get();
        app.configuration.FFTdim = 2;
        app.configuration.size[1] = 17;
        const int other = 1 - target;
        if (VkFFTScheduler(&app, plan, other) != VKFFT_SUCCESS)
            return __LINE__;
        plan->inverseBluesteinAxes[other][0] = plan->axes[other][0];
        plan->bigSequenceEvenR2C = 1;
        const auto saved = snapshot_schedule(*plan);
        const int previous_allocations = live();
        const auto old_fft = app.configuration.fixMaxRaderPrimeFFT;
        const auto old_mult = app.configuration.fixMaxRaderPrimeMult;
        int calls = 0;
        bool invalid_state = false;
        bool caught = false;
        VkFFTResult result = VKFFT_SUCCESS;
        try {
            result = vkfft_scheduler::schedule_axis(&app, plan, target, [&](auto* a, auto* p, int axis) {
                ++calls;
                auto& c = a->configuration;
                if (calls == 1) {
                    invalid_state |= c.fixMaxRaderPrimeFFT != old_fft || c.fixMaxRaderPrimeMult != old_mult;
                    p->numAxisUploads[axis] = 1;
                    p->actualPerformR2CPerAxis[axis] = 1;
                    p->actualFFTSizePerAxis[axis][axis] = 17;
                    p->axisSplit[axis][0] = 17;
                } else {
                    invalid_state |= calls != 2 || live() != previous_allocations ||
                                     c.fixMaxRaderPrimeFFT != c.fixMinRaderPrimeFFT ||
                                     c.fixMaxRaderPrimeMult != c.fixMinRaderPrimeMult ||
                                     p->numAxisUploads[axis] || p->actualPerformR2CPerAxis[axis] ||
                                     p->actualFFTSizePerAxis[axis][axis] || p->axisSplit[axis][0] ||
                                     p->axes[axis][3].specializationConstants.raderContainer ||
                                     p->bigSequenceEvenR2C != (axis == 0 ? 0 : 1);
                }
                if ((scenario == RetryScenario::FirstAllocationFailure && calls == 1) ||
                    (scenario == RetryScenario::RetryAllocationFailure && calls == 2))
                    state.fail_at = state.attempts + 1;
                auto& sc = p->axes[axis][calls == 1 ? 3 : 2].specializationConstants;
                sc.numRaderPrimes = 1;
                sc.raderContainer =
                    static_cast<VkFFTRaderContainer*>(allocate(1, sizeof(VkFFTRaderContainer)));
                if (!sc.raderContainer)
                    return VKFFT_ERROR_MALLOC_FAILED;
                sc.raderContainer[0].prime = 17;
                if (calls == 1)
                    return scenario == RetryScenario::FirstSuccess ? VKFFT_SUCCESS
                           : scenario == RetryScenario::FirstError ? VKFFT_ERROR_MATH_FAILED
                                                                   : VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;
                if (scenario == RetryScenario::RetryException)
                    throw std::bad_alloc();
                return scenario == RetryScenario::RetryError         ? VKFFT_ERROR_MATH_FAILED
                       : scenario == RetryScenario::RetryUnsupported ? VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH
                                                                     : VKFFT_SUCCESS;
            });
        } catch (const std::bad_alloc&) {
            caught = true;
        }
        VkFFTResult expected = VKFFT_SUCCESS;
        if (scenario == RetryScenario::FirstError || scenario == RetryScenario::RetryError)
            expected = VKFFT_ERROR_MATH_FAILED;
        else if (scenario == RetryScenario::RetryUnsupported)
            expected = VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;
        else if (scenario == RetryScenario::FirstAllocationFailure ||
                 scenario == RetryScenario::RetryAllocationFailure)
            expected = VKFFT_ERROR_MALLOC_FAILED;
        const bool first_attempt_only = scenario == RetryScenario::FirstSuccess ||
                                        scenario == RetryScenario::FirstError ||
                                        scenario == RetryScenario::FirstAllocationFailure;
        const int expected_calls = first_attempt_only ? 1 : 2;
        if (invalid_state || caught != (scenario == RetryScenario::RetryException) ||
            (!caught && result != expected) || calls != expected_calls ||
            app.configuration.fixMaxRaderPrimeFFT != old_fft ||
            app.configuration.fixMaxRaderPrimeMult != old_mult ||
            !unchanged_other_axes(*plan, *saved, target) || state.invalid)
            return __LINE__;
    }
    return live() || state.invalid ? __LINE__ : 0;
}

int scheduler_retry_invalid_axis_case() {
    SchedulerFixture fixture(17, false, false, 4);
    int calls = 0;
    auto schedule = [&](VkFFTApplication*, VkFFTPlan*, int) {
        ++calls;
        return VKFFT_SUCCESS;
    };
    for (int axis : {-1, 1, VKFFT_MAX_FFT_DIMENSIONS})
        if (vkfft_scheduler::schedule_axis(&fixture.app, fixture.plan.get(), axis, schedule) !=
            VKFFT_ERROR_MATH_FAILED)
            return __LINE__;
    if (vkfft_scheduler::schedule_axis(nullptr, fixture.plan.get(), 0, schedule) != VKFFT_ERROR_MATH_FAILED ||
        vkfft_scheduler::schedule_axis(&fixture.app, nullptr, 0, schedule) != VKFFT_ERROR_MATH_FAILED ||
        calls)
        return __LINE__;
    fixture.app.configuration.FFTdim = VKFFT_MAX_FFT_DIMENSIONS + 1;
    if (vkfft_scheduler::schedule_axis(&fixture.app, fixture.plan.get(), 0, schedule) !=
            VKFFT_ERROR_MATH_FAILED ||
        calls)
        return __LINE__;
    return 0;
}

} // namespace codegen_fake

extern "C" int vkfft_test_scheduler_retry_real() {
    for (bool inverse : {false, true})
        for (bool precision : {false, true})
            if (int line = codegen_fake::natural_scheduler_retry_case(inverse, precision))
                return line;
    return 0;
}

extern "C" int vkfft_test_scheduler_retry_exits() {
    using codegen_fake::RetryScenario;
    for (int axis : {0, 1})
        for (auto scenario :
             {RetryScenario::FirstSuccess, RetryScenario::FirstError, RetryScenario::RetrySuccess,
              RetryScenario::RetryError, RetryScenario::RetryException, RetryScenario::RetryUnsupported,
              RetryScenario::FirstAllocationFailure, RetryScenario::RetryAllocationFailure})
            if (int line = codegen_fake::scheduler_retry_exit_case(axis, scenario))
                return line;
    return codegen_fake::scheduler_retry_invalid_axis_case();
}
