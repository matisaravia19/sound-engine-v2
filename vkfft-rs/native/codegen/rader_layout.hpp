// Bounds for the Rader generators' native arithmetic and register indexing.
#ifndef VKFFT_RS_CODEGEN_RADER_LAYOUT_HPP
#define VKFFT_RS_CODEGEN_RADER_LAYOUT_HPP

#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include <climits>
#include <limits>

namespace vkfft_codegen {

// The axis owns the selected container and registers. Their allocation dimensions
// must remain unchanged until generation and register teardown have finished.
static inline bool rader_stage_layout(VkFFTSpecializationConstantsLayout* sc, pfINT radix, pfINT stage_size,
                                      bool direct) {
    if (sc->res != VKFFT_SUCCESS)
        return false;
    const auto* rader = sc->currentRaderContainer;
    if (!rader || !sc->useRader || rader->type != static_cast<int>(direct) || radix < 3 ||
        radix != rader->prime || rader->generator <= 0 || rader->generator >= radix || !sc->regIDs ||
        sc->registers_per_thread <= 0 || sc->registerBoost <= 0 ||
        sc->registers_per_thread > INT_MAX / sc->registerBoost || sc->fftDim.data.i <= 0 ||
        sc->fftDim.data.i % radix || stage_size <= 0 || stage_size > sc->fftDim.data.i / radix ||
        sc->localSize[0].data.i <= 0 || sc->localSize[1].data.i <= 0) {
        sc->res = VKFFT_ERROR_MATH_FAILED;
        return false;
    }
    return true;
}

// Every inner FFT uses an inline radix kernel and maps logical registers into
// the physical allocation. Check all stages before emitting any partial shader.
static inline bool rader_fft_layout(VkFFTSpecializationConstantsLayout* sc, pfINT radix, pfINT stage_size) {
    if (!rader_stage_layout(sc, radix, stage_size, false))
        return false;
    const auto& rader = *sc->currentRaderContainer;
    constexpr int capacity = sizeof(rader.stageRadix) / sizeof(rader.stageRadix[0]);
    if (!sc->useRaderFFT || sc->numSharedBanks < 2 || rader.numStages <= 0 || rader.numStages > capacity ||
        rader.containerFFTDim != radix - 1 || rader.containerFFTNum != sc->fftDim.data.i / radix) {
        sc->res = VKFFT_ERROR_MATH_FAILED;
        return false;
    }
    pfINT length = 1;
    for (int stage = 0; stage < rader.numStages; ++stage) {
        const int inner_radix = rader.stageRadix[stage];
        // The inline generator implements 2..16 and 32; other primes need a Rader stage.
        if (inner_radix < 2 || (inner_radix > 16 && inner_radix != 32) ||
            length > rader.containerFFTDim / inner_radix) {
            sc->res = VKFFT_ERROR_MATH_FAILED;
            return false;
        }
        const int logical = rader.registers_per_thread_per_radix[inner_radix];
        if (logical < inner_radix || logical % inner_radix || logical > sc->registers_per_thread) {
            sc->res = VKFFT_ERROR_MATH_FAILED;
            return false;
        }
        length *= inner_radix;
    }
    if (length != rader.containerFFTDim) {
        sc->res = VKFFT_ERROR_MATH_FAILED;
        return false;
    }
    return true;
}

// Each logical group uses x0[group] and two complex registers. Check the actual
// initialized x0 count as well as its fixed capacity before the generator divides.
static inline bool rader_direct_layout(VkFFTSpecializationConstantsLayout* sc, pfINT radix,
                                       pfINT stage_size) {
    if (!rader_stage_layout(sc, radix, stage_size, true))
        return false;
    constexpr pfINT capacity = sizeof(sc->x0) / sizeof(sc->x0[0]);
    const pfINT x0_count = sc->raderRegisters / 2;
    const pfINT subgroup_count = sc->localSize[sc->stridedSharedLayout ? 1 : 0].data.i / ((radix + 1) / 2);
    if (!sc->useRaderMult || x0_count <= 0 || x0_count > capacity || subgroup_count <= 0) {
        sc->res = VKFFT_ERROR_MATH_FAILED;
        return false;
    }
    const pfINT transforms = sc->fftDim.data.i / radix;
    const pfINT groups = transforms / subgroup_count + (transforms % subgroup_count != 0);
    const pfINT registers = sc->registers_per_thread * sc->registerBoost;
    if (groups > x0_count || groups > registers / 2 ||
        subgroup_count > std::numeric_limits<pfINT>::max() / groups / radix) {
        sc->res = VKFFT_ERROR_MATH_FAILED;
        return false;
    }
    return true;
}

} // namespace vkfft_codegen
#endif
