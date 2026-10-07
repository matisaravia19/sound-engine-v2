// Validate transform and convolution extents before factorization can consume them.
#ifndef VKFFT_RS_SCHEDULER_EXTENTS_HPP
#define VKFFT_RS_SCHEDULER_EXTENTS_HPP

#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include <initializer_list>
#include <limits>

namespace vkfft_scheduler {

// Preserve an existing R2C split unless a DCT/DST mode replaces it. Zero or wrapped
// lengths would make the scheduler's repeated-factor division fail to terminate.
static inline VkFFTResult real_transform_length(const VkFFTConfiguration& c, pfUINT input,
                                                pfUINT& length) {
    if (!input || !length)
        return VKFFT_ERROR_MATH_FAILED;
    const auto maximum = (std::numeric_limits<pfUINT>::max)();
    auto result = length;
    if (c.performDCT == 1) {
        if (input <= 1 || input - 1 > maximum / 2)
            return VKFFT_ERROR_MATH_FAILED;
        result = 2 * (input - 1);
    }
    if (c.performDST == 1) {
        if (input > maximum / 2 - 1)
            return VKFFT_ERROR_MATH_FAILED;
        result = 2 * (input + 1);
    }
    if ((c.performDCT == 4 || c.performDST == 4) && input % 2 == 0)
        result = input / 2;
    length = result;
    return VKFFT_SUCCESS;
}

// Compute the linear-convolution minimum without overflowing the intermediate 2*N.
// Outputs in this header remain unchanged on error.
static inline VkFFTResult bluestein_minimum(pfUINT length, pfUINT& minimum) {
    if (!length || length - 1 > (std::numeric_limits<pfUINT>::max)() / 2)
        return VKFFT_ERROR_MATH_FAILED;
    minimum = 2 * (length - 1) + 1;
    return VKFFT_SUCCESS;
}

// User-selected padding must contain the convolution and use supported radix factors.
static inline VkFFTResult validate_bluestein_padding(pfUINT padded, pfUINT minimum) {
    if (!minimum || padded < minimum)
        return VKFFT_ERROR_MATH_FAILED;
    for (pfUINT prime : {2, 3, 5, 7, 11, 13})
        while (padded % prime == 0)
            padded /= prime;
    return padded == 1 ? VKFFT_SUCCESS : VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;
}

// Select a borrowed table entry, or leave the minimum for the existing automatic
// search. The native caller owns both arrays and supplies their common length.
// Beyond the final entry's capacity, VkFFT's documented default search takes over.
static inline VkFFTResult initial_bluestein_padding(const VkFFTConfiguration& c, pfUINT length,
                                                    pfUINT& minimum, pfUINT& padded,
                                                    bool& selected) {
    pfUINT required = 0;
    auto result = bluestein_minimum(length, required);
    if (result != VKFFT_SUCCESS)
        return result;
    auto candidate = required;
    bool found = false;
    const auto count = c.useCustomBluesteinPaddingPattern ? c.useCustomBluesteinPaddingPattern
                                                          : c.autoCustomBluesteinPaddingPattern;
    if (count && !c.fixMaxRadixBluestein) {
        if (!c.primeSizes || !c.paddedSizes)
            return VKFFT_ERROR_EMPTY_useCustomBluesteinPaddingPattern_arrays;
        if (count > static_cast<pfUINT>((std::numeric_limits<int>::max)()))
            return VKFFT_ERROR_MATH_FAILED;
        for (pfUINT i = 0; i < count; ++i)
            if (!c.primeSizes[i] || (i && c.primeSizes[i] <= c.primeSizes[i - 1]))
                return VKFFT_ERROR_MATH_FAILED;
        for (pfUINT i = 0; i < count && length >= c.primeSizes[i]; ++i) {
            const bool last = i == count - 1;
            if ((!last && length < c.primeSizes[i + 1]) || (last && required <= c.paddedSizes[i])) {
                candidate = c.paddedSizes[i];
                result = validate_bluestein_padding(candidate, required);
                if (result != VKFFT_SUCCESS)
                    return result;
                found = true;
                break;
            }
        }
    }
    minimum = required;
    padded = candidate;
    selected = found;
    return VKFFT_SUCCESS;
}

} // namespace vkfft_scheduler
#endif
