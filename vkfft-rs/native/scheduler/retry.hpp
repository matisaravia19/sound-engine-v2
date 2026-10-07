// Ownership-preserving fallback while native axes are still being scheduled.
#ifndef VKFFT_RS_SCHEDULER_RETRY_HPP
#define VKFFT_RS_SCHEDULER_RETRY_HPP

#include "vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_Scheduler.h"
#include "vkFFT/vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_DeletePlan.h"
#include <cstring>

namespace vkfft_scheduler {

// Borrows configuration during serial scheduling. The configuration must outlive
// this scope; restoration also runs when allocation failure unwinds the retry.
class ScopedRaderLimits {
  public:
    // Equal exclusive/minimum bounds disable both Rader variants for the retry.
    explicit ScopedRaderLimits(VkFFTConfiguration& configuration) noexcept
        : configuration_(configuration), fft_(configuration.fixMaxRaderPrimeFFT),
          mult_(configuration.fixMaxRaderPrimeMult) {
        configuration_.fixMaxRaderPrimeFFT = configuration_.fixMinRaderPrimeFFT;
        configuration_.fixMaxRaderPrimeMult = configuration_.fixMinRaderPrimeMult;
    }
    // Later axes and plans must see the caller's original tuning limits.
    ~ScopedRaderLimits() {
        configuration_.fixMaxRaderPrimeFFT = fft_;
        configuration_.fixMaxRaderPrimeMult = mult_;
    }
    ScopedRaderLimits(const ScopedRaderLimits&) = delete;
    ScopedRaderLimits& operator=(const ScopedRaderLimits&) = delete;

  private:
    VkFFTConfiguration& configuration_;
    pfUINT fft_;
    pfUINT mult_;
};

// Called only during initial scheduling, before kernel compilation. Release all
// upload slots, including partial ones, before clearing this axis's host state.
// Other axes and their borrowed inverse-Bluestein copies retain their ownership.
static inline void reset_scheduled_axis(VkFFTApplication* app, VkFFTPlan* plan, int axis) {
    for (auto& upload : plan->axes[axis])
        deleteAxis(app, &upload, 0);
    std::memset(plan->axes[axis], 0, sizeof(plan->axes[axis]));
    std::memset(plan->inverseBluesteinAxes[axis], 0, sizeof(plan->inverseBluesteinAxes[axis]));
    std::memset(plan->actualFFTSizePerAxis[axis], 0, sizeof(plan->actualFFTSizePerAxis[axis]));
    std::memset(plan->axisSplit[axis], 0, sizeof(plan->axisSplit[axis]));
    plan->numAxisUploads[axis] = 0;
    plan->actualPerformR2CPerAxis[axis] = 0;
    if (axis == 0)
        plan->bigSequenceEvenR2C = 0;
}

// App and plan remain owned by the initializer. Retry only an unsupported length,
// preserving completed axes and restoring temporary limits on every exit path.
template <class Schedule>
static inline VkFFTResult schedule_axis(VkFFTApplication* app, VkFFTPlan* plan, int axis, Schedule schedule) {
    if (!app || !plan || app->configuration.FFTdim > VKFFT_MAX_FFT_DIMENSIONS || axis < 0 ||
        axis >= VKFFT_MAX_FFT_DIMENSIONS || static_cast<pfUINT>(axis) >= app->configuration.FFTdim)
        return VKFFT_ERROR_MATH_FAILED;
    auto result = schedule(app, plan, axis);
    if (result != VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH)
        return result;
    reset_scheduled_axis(app, plan, axis);
    ScopedRaderLimits limits(app->configuration);
    return schedule(app, plan, axis);
}

} // namespace vkfft_scheduler
#endif
