// Shared native stage selection and normalization invariants.
#ifndef VKFFT_RS_CODEGEN_STAGE_POLICY_HPP
#define VKFFT_RS_CODEGEN_STAGE_POLICY_HPP

#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include <limits>

namespace vkfft_codegen {

// Validate the fixed-array lookup before either dispatcher reads its stage metadata.
static inline bool stage_dispatch_valid(VkFFTSpecializationConstantsLayout* sc, pfUINT stage,
                                        pfINT radix, pfUINT layout) {
    if (sc->res != VKFFT_SUCCESS)
        return false;
    constexpr size_t stages = sizeof(sc->stageRadix) / sizeof(sc->stageRadix[0]);
    if (sc->numStages <= 0 || static_cast<size_t>(sc->numStages) > stages ||
        stage >= static_cast<pfUINT>(sc->numStages) || radix < 2 ||
        radix != sc->stageRadix[stage] || sc->rader_generator[stage] < 0 ||
        layout > static_cast<pfUINT>(std::numeric_limits<int>::max()) || layout % 10 > 2) {
        sc->res = VKFFT_ERROR_MATH_FAILED;
        return false;
    }
    return true;
}

// Clear a previous selection before searching: a missing stage must never borrow
// an unrelated Rader kernel. The selected container remains owned by the axis.
static inline bool select_rader_stage(VkFFTSpecializationConstantsLayout* sc, pfUINT stage,
                                      pfINT radix) {
    if (!stage_dispatch_valid(sc, stage, radix, 0))
        return false;
    sc->currentRaderContainer = nullptr;
    constexpr int capacity = sizeof(sc->stageRadix) / sizeof(sc->stageRadix[0]);
    if (sc->numRaderPrimes <= 0 || sc->numRaderPrimes > capacity || !sc->raderContainer) {
        sc->res = VKFFT_ERROR_MATH_FAILED;
        return false;
    }
    for (int i = 0; i < sc->numRaderPrimes; ++i) {
        auto& candidate = sc->raderContainer[i];
        if (candidate.prime != radix)
            continue;
        if (sc->currentRaderContainer || candidate.generator != sc->rader_generator[stage] ||
            candidate.generator <= 0 || candidate.generator >= radix ||
            (candidate.type != 0 && candidate.type != 1)) {
            sc->currentRaderContainer = nullptr;
            sc->res = VKFFT_ERROR_MATH_FAILED;
            return false;
        }
        sc->currentRaderContainer = &candidate;
    }
    if (!sc->currentRaderContainer)
        sc->res = VKFFT_ERROR_MATH_FAILED;
    return sc->res == VKFFT_SUCCESS;
}

// Only active normalization factors need valid dimensions. Check each integer
// adjustment/product before evaluation; leave the caller's divisor unchanged on error.
static inline bool stage_normalization(VkFFTSpecializationConstantsLayout* sc, pfINT stage_size,
                                       pfLD angle, pfINT& divisor) {
    if (sc->res != VKFFT_SUCCESS)
        return false;
    constexpr pfINT maximum = std::numeric_limits<pfINT>::max();
    pfINT value = 1;
    const bool first_upload_stage = stage_size == 1 && sc->axis_upload_id == 0;
    if (first_upload_stage &&
        ((sc->actualInverse && sc->normalize) || (sc->convolutionStep && angle > 0)) &&
        !(sc->useBluesteinFFT && angle < 0)) {
        value = sc->sourceFFTSize.data.i;
        if (value <= 0) {
            sc->res = VKFFT_ERROR_MATH_FAILED;
            return false;
        }
        if ((sc->performDCT || sc->performDST) && sc->actualInverse) {
            if (sc->performDST == 1) {
                if (value == maximum) {
                    sc->res = VKFFT_ERROR_MATH_FAILED;
                    return false;
                }
                ++value;
            } else if (sc->performDCT == 1) {
                --value;
            }
            if (value <= 0 || value > maximum / 2) {
                sc->res = VKFFT_ERROR_MATH_FAILED;
                return false;
            }
            value *= 2;
        }
    }
    if (first_upload_stage && sc->useBluesteinFFT && angle > 0) {
        const pfINT length = sc->fft_dim_full.data.i;
        if (length <= 0 || value > maximum / length) {
            sc->res = VKFFT_ERROR_MATH_FAILED;
            return false;
        }
        value *= length;
    }
    divisor = value;
    return true;
}

} // namespace vkfft_codegen
#endif
