// Checked workload and temporary-storage sizes used during native planning.
#ifndef VKFFT_RS_SCHEDULER_SIZES_HPP
#define VKFFT_RS_SCHEDULER_SIZES_HPP

#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include <limits>

namespace vkfft_scheduler {

// A storage extent must stay positive and representable. Leave it unchanged on error.
static inline bool multiply_extent(pfUINT& extent, pfUINT factor) {
    if (!extent || !factor || extent > (std::numeric_limits<pfUINT>::max)() / factor)
        return false;
    extent *= factor;
    return true;
}

// Count all elements in a borrowed shape, including independent batches/features/kernels.
// Optional R2C padding adds one complex element to the first dimension. The output
// is published only after every factor has been checked.
static inline VkFFTResult element_count(const VkFFTApplication& app,
                                        const pfUINT (&shape)[VKFFT_MAX_FFT_DIMENSIONS],
                                        bool pad_first, pfUINT& elements) {
    const auto& c = app.configuration;
    if (!c.FFTdim || c.FFTdim > VKFFT_MAX_FFT_DIMENSIONS)
        return VKFFT_ERROR_MATH_FAILED;
    pfUINT count = c.numberBatches > app.actualNumBatches ? c.numberBatches : app.actualNumBatches;
    if (!multiply_extent(count, c.coordinateFeatures) || !multiply_extent(count, c.numberKernels))
        return VKFFT_ERROR_MATH_FAILED;
    for (pfUINT axis = 0; axis < c.FFTdim; ++axis) {
        auto extent = shape[axis];
        if (!extent)
            return VKFFT_ERROR_MATH_FAILED;
        if (axis == 0 && pad_first) {
            if (extent == (std::numeric_limits<pfUINT>::max)())
                return VKFFT_ERROR_MATH_FAILED;
            ++extent;
        }
        if (!multiply_extent(count, extent))
            return VKFFT_ERROR_MATH_FAILED;
    }
    elements = count;
    return VKFFT_SUCCESS;
}

// The plan's scheduled shape already includes any Bluestein expansion. Count it
// once, apply even-R2C padding when required, then check conversion to bytes.
static inline VkFFTResult temporary_bytes(const VkFFTApplication& app, const VkFFTPlan& plan,
                                          int axis, int complex_size, pfUINT& bytes) {
    if (axis < 0 || axis >= VKFFT_MAX_FFT_DIMENSIONS ||
        static_cast<pfUINT>(axis) >= app.configuration.FFTdim || complex_size <= 0)
        return VKFFT_ERROR_MATH_FAILED;
    const bool pad_first = app.configuration.performR2C && axis == 0 && plan.bigSequenceEvenR2C;
    pfUINT elements = 0;
    auto result = element_count(app, plan.actualFFTSizePerAxis[axis], pad_first, elements);
    if (result != VKFFT_SUCCESS)
        return result;
    if (!multiply_extent(elements, static_cast<pfUINT>(complex_size)))
        return VKFFT_ERROR_MATH_FAILED;
    bytes = elements;
    return VKFFT_SUCCESS;
}

} // namespace vkfft_scheduler
#endif
