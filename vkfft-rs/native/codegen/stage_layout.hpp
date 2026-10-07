// Bounds shared by ordinary radix stages and their register shuffles.
#ifndef VKFFT_RS_CODEGEN_STAGE_LAYOUT_HPP
#define VKFFT_RS_CODEGEN_STAGE_LAYOUT_HPP

#include "codegen/storage.hpp"

namespace vkfft_codegen {

// The physical register allocation must outlive generation and keep these dimensions.
// Bounding its signed-int product also bounds the logical register index formulas.
static inline bool stage_register_layout(VkFFTSpecializationConstantsLayout* sc, pfINT radix) {
    if (sc->res != VKFFT_SUCCESS) return false;
    constexpr pfINT radix_count = sizeof(sc->registers_per_thread_per_radix) /
        sizeof(sc->registers_per_thread_per_radix[0]);
    if (radix < 2 || radix >= radix_count || !sc->regIDs ||
        sc->registers_per_thread <= 0 || sc->registerBoost <= 0 ||
        sc->registers_per_thread > INT_MAX / sc->registerBoost) {
        sc->res = VKFFT_ERROR_MATH_FAILED;
        return false;
    }
    const int logical = sc->registers_per_thread_per_radix[radix];
    if (logical < radix || logical > sc->registers_per_thread || logical % radix) {
        sc->res = VKFFT_ERROR_MATH_FAILED;
        return false;
    }
    return true;
}

// Validate divisors and stage products before the generator uses native arithmetic.
static inline bool stage_geometry(VkFFTSpecializationConstantsLayout* sc, pfINT radix, pfINT size) {
    if (!stage_register_layout(sc, radix)) return false;
    if (size <= 0 || sc->fftDim.data.i <= 0 || size > sc->fftDim.data.i / radix ||
        sc->localSize[0].data.i <= 0 || sc->localSize[1].data.i <= 0 || sc->numSharedBanks < 2) {
        sc->res = VKFFT_ERROR_MATH_FAILED;
        return false;
    }
    return true;
}

// Every register uses the same container type, but retains independent name storage.
static inline bool allocate_register_containers(VkFFTSpecializationConstantsLayout* sc,
                                                PfContainer* values, pfINT count) {
    if (sc->res != VKFFT_SUCCESS) return false;
    for (pfINT i = 0; i < count; ++i) {
        values[i].type = sc->regIDs[0].type;
        PfAllocateContainerFlexible(sc, &values[i], 50);
        if (sc->res != VKFFT_SUCCESS) return false;
    }
    return true;
}

} // namespace vkfft_codegen
#endif
