// Checked layout calculations shared by the patched native LUT builders.
#ifndef VKFFT_RS_LUT_LAYOUT_HPP
#define VKFFT_RS_LUT_LAYOUT_HPP

#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include <climits>
#include <cstdint>

namespace vkfft_lut {
namespace {

// Check a product and its accumulated offset before either operation can wrap.
// On failure the destination is unchanged; limits come from the consuming field.
static inline bool add_product(pfUINT& total, pfUINT count, pfUINT width, pfUINT limit) {
    if (total > limit || (width && count > (limit - total) / width))
        return false;
    total += count * width;
    return true;
}

// Native plan arrays have four upload slots per axis, independently of FFT rank.
static inline bool valid_axis(const VkFFTSpecializationConstantsLayout& sc) {
    return sc.axis_id >= 0 && sc.axis_id < VKFFT_MAX_FFT_DIMENSIONS && sc.numAxisUploads >= 1 &&
           sc.numAxisUploads <= 4 && sc.axis_upload_id >= 0 &&
           sc.axis_upload_id < sc.numAxisUploads;
}

// The scheduler owns this array for the lifetime of the specialization state.
// Its distinct primes cannot outnumber the fixed array of FFT stages.
static inline bool valid_rader_list(const VkFFTSpecializationConstantsLayout& sc) {
    if (sc.numRaderPrimes < 0 || sc.numRaderPrimes > int(sizeof(sc.stageRadix) / sizeof(int)) ||
        (sc.numRaderPrimes && !sc.raderContainer))
        return false;
    for (int i = 0; i < sc.numRaderPrimes; ++i) {
        const auto& rader = sc.raderContainer[i];
        if (rader.prime < 2 || rader.generator < 1 || rader.generator >= rader.prime ||
            (rader.type != 0 && rader.type != 1))
            return false;
    }
    return true;
}

// These reservations match the shader generator's stage-offset convention.
static inline pfUINT stage_columns(int radix, bool double_double) {
    switch (radix) {
        case 2:
            return 1;
        case 4:
            return 2;
        case 8:
            return 3;
        case 16:
            return 4;
        case 32:
            return 5;
        case 11:
        case 13:
            return double_double ? radix : radix - 1;
        default:
            return radix >= 3 && radix <= 15 ? radix - 1 : radix;
    }
}

// Stage counts are checked against their actual fixed-array extent before reads.
// The accumulated result must fit maxStageSumLUT, which is a signed native int.
template <size_t N>
static inline bool add_stages(const int (&radices)[N], int count, bool double_double, bool reverse,
                              pfUINT& total, const int* generators = nullptr) {
    if (count < 0 || static_cast<size_t>(count) > N)
        return false;
    pfUINT dimension = 1;
    for (int step = 0; step < count; ++step) {
        const int index = reverse ? count - 1 - step : step;
        const int radix = radices[index];
        if (radix < 2)
            return false;
        const pfUINT columns = stage_columns(radix, double_double);
        // A Rader stage writes a full radix, unlike ordinary small odd radices.
        if (step && generators && generators[index] > 0 && (radix & (radix - 1)) &&
            static_cast<pfUINT>(radix) > columns)
            return false;
        if (step && !add_product(total, dimension, columns, INT_MAX))
            return false;
        pfUINT product = 0;
        if (!add_product(product, dimension, radix, UINT64_MAX))
            return false;
        dimension = product;
    }
    return true;
}

// Lay out outer twiddles, forward Rader twiddles and inverse Rader twiddles in
// their existing order. No table allocation or trigonometry occurs here.
static inline bool stage_layout(const VkFFTConfiguration& config,
                                VkFFTSpecializationConstantsLayout& sc, pfUINT& count) {
    count = 0;
    const bool double_double =
        config.quadDoubleDoublePrecision || config.quadDoubleDoublePrecisionDoubleMemory;
    if (!valid_axis(sc) || !valid_rader_list(sc) ||
        !add_stages(sc.stageRadix, sc.numStages, double_double, false, count, sc.rader_generator))
        return false;
    for (int direction = 0; direction < 2; ++direction) {
        const bool reverse = direction != 0;
        for (int i = 0; i < sc.numRaderPrimes; ++i) {
            auto& rader = sc.raderContainer[i];
            if (rader.type != 0)
                continue;
            if (rader.numStages < 1)
                return false;
            if (sc.useRader && !sc.inline_rader_kernel && !rader.raderFFTkernel)
                return false;
            if (reverse)
                rader.RaderRadixOffsetLUTiFFT = static_cast<pfINT>(count);
            else
                rader.RaderRadixOffsetLUT = static_cast<pfINT>(count);
            if (!add_stages(rader.stageRadix, rader.numStages, double_double, reverse, count))
                return false;
        }
    }
    sc.maxStageSumLUT = static_cast<int>(count);
    return true;
}

// Table precision is fixed for every axis in an application.
static inline size_t complex_element_bytes(const VkFFTConfiguration& config) {
    return config.quadDoubleDoublePrecision || config.quadDoubleDoublePrecisionDoubleMemory
               ? 4 * sizeof(double)
           : config.doublePrecision || config.doublePrecisionFloatMemory ? 2 * sizeof(double)
                                                                         : 2 * sizeof(float);
}

// Complex indices become signed offsets and byte-sized allocations later.
static inline pfUINT complex_limit(const VkFFTConfiguration& config) {
    return SIZE_MAX / complex_element_bytes(config);
}

// Check all integer-Rader offsets before changing the array or allocating. The
// shader's signed int offsets also bound the total accessible element count.
static inline bool rader_uint_layout(VkFFTSpecializationConstantsLayout& sc, pfUINT& bytes) {
    bytes = 0;
    if (!valid_axis(sc) || !valid_rader_list(sc) || sc.numRaderPrimes == 0)
        return false;
    pfUINT count = 0;
    const pfUINT limit =
        SIZE_MAX / sizeof(uint32_t) < INT_MAX ? SIZE_MAX / sizeof(uint32_t) : INT_MAX;
    for (int i = 0; i < sc.numRaderPrimes; ++i) {
        if (!add_product(count, sc.raderContainer[i].prime, 1, limit))
            return false;
    }
    pfUINT offset = 0;
    for (int i = 0; i < sc.numRaderPrimes; ++i) {
        sc.raderContainer[i].raderUintLUToffset = static_cast<int>(offset);
        offset += sc.raderContainer[i].prime;
    }
    bytes = count * sizeof(uint32_t);
    return true;
}

} // namespace
} // namespace vkfft_lut
#endif
