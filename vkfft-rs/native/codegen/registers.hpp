// Allocation bounds and shared teardown for shader-generation register storage.
#ifndef VKFFT_RS_CODEGEN_REGISTERS_HPP
#define VKFFT_RS_CODEGEN_REGISTERS_HPP

#include "vkFFT/vkFFT_CodeGen/vkFFT_MathUtils/vkFFT_MathUtils.h"
#include <climits>

namespace vkfft_codegen {

// These dimensions remain unchanged between register allocation and teardown.
static inline int register_count(const VkFFTSpecializationConstantsLayout* sc, int type, bool r2c) {
    if (sc->registers_per_thread <= 0)
        return 0;
    uint64_t count = static_cast<uint64_t>(sc->registers_per_thread);
    if (r2c) {
        if (count < 4)
            return 0; // The decomposition kernel accesses four complex registers.
    } else {
        if (sc->registerBoost <= 0 || (sc->convolutionStep && sc->matrixConvolution <= 0))
            return 0;
        if (sc->mergeSequencesR2C == 1 && type == 500)
            count += 2;
        const uint64_t coordinates = sc->convolutionStep ? sc->matrixConvolution : 1;
        if (count > INT_MAX / static_cast<uint64_t>(sc->registerBoost))
            return 0;
        count *= sc->registerBoost;
        if (count > INT_MAX / coordinates)
            return 0;
        count *= coordinates;
    }
    if (count > INT_MAX || count > SIZE_MAX / sizeof(PfContainer))
        return 0;
    return static_cast<int>(count);
}

static inline bool register_layout_valid(const VkFFTSpecializationConstantsLayout* sc) {
    const int locals = sizeof(sc->locID) / sizeof(sc->locID[0]);
    const int rader = sizeof(sc->x0) / sizeof(sc->x0[0]);
    const int stages = sizeof(sc->stageRadix) / sizeof(sc->stageRadix[0]);
    if (sc->usedLocRegs < 0 || sc->usedLocRegs > locals || sc->numStages < 0 ||
        sc->numStages > stages)
        return false;
    if (sc->useRaderMult &&
        (sc->usedLocRegs < 1 || sc->raderRegisters < 0 || sc->raderRegisters / 2 > rader ||
         sc->raderRegisters / 2 < (sc->useRaderFFT ? 2 : 0)))
        return false;
    bool needs_second_local = sc->useRaderFFT != 0;
    for (int i = 0; i < sc->numStages; ++i)
        needs_second_local |=
            sc->stageRadix[i] == 8 || sc->stageRadix[i] == 16 || sc->stageRadix[i] == 32;
    return !needs_second_local || sc->maxNonPow2Radix <= 1 || sc->usedLocRegs >= 2;
}

static inline void release_array(VkFFTSpecializationConstantsLayout* sc, PfContainer*& array,
                                 int count) {
    if (!array)
        return;
    for (int i = 0; i < count; ++i)
        PfDeallocateContainer(sc, &array[i]);
    free(array);
    array = nullptr;
}

// Tags identify initialized fields, so errors and optional features need no cleanup branches.
static inline void release_registers(VkFFTSpecializationConstantsLayout* sc, int count) {
    release_array(sc, sc->regIDs, count);
    release_array(sc, sc->regIDs_copy, count);
    release_array(sc, sc->temp_conv, sc->matrixConvolution);
    for (auto& value : sc->locID)
        PfDeallocateContainer(sc, &value);
    for (auto& value : sc->x0)
        PfDeallocateContainer(sc, &value);
    PfContainer* const fields[] = {
        &sc->w,
        &sc->temp,
        &sc->tempFloat,
        &sc->tempQuad,
        &sc->tempQuad2,
        &sc->tempQuad3,
        &sc->tempIntQuad,
        &sc->tempInt,
        &sc->tempInt2,
        &sc->shiftX,
        &sc->shiftY,
        &sc->shiftZ,
        &sc->iw,
        &sc->stageInvocationID,
        &sc->blockInvocationID,
        &sc->sdataID,
        &sc->combinedID,
        &sc->inoutID,
        &sc->inoutID2,
        &sc->inoutID_x,
        &sc->inoutID_y,
        &sc->disableThreads,
        &sc->raderIDx,
        &sc->raderIDx2,
        &sc->LUTId,
        &sc->angle,
        &sc->mult,
    };
    for (auto* value : fields)
        PfDeallocateContainer(sc, value);
}

} // namespace vkfft_codegen
#endif
