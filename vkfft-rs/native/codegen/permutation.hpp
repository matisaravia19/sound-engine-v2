// Checked permutations of existing register names, with no temporary allocations.
#ifndef VKFFT_RS_CODEGEN_PERMUTATION_HPP
#define VKFFT_RS_CODEGEN_PERMUTATION_HPP

#include "codegen/names.hpp"

// values owns capacity containers for the call. stride selects a sub-sequence;
// only names/values move, preserving every nested container array's address.
static inline void PfPermuteContainers(VkFFTSpecializationConstantsLayout* sc,
                                      const pfUINT* permutation, pfUINT count,
                                      PfContainer* values, pfUINT capacity, pfUINT stride = 1) {
    if (sc->res != VKFFT_SUCCESS || count == 0) return;
    constexpr size_t maximum = sizeof(sc->locID) / sizeof(sc->locID[0]);
    if (!permutation || !values || count > maximum || !capacity || !stride ||
        count - 1 > (capacity - 1) / stride) {
        sc->res = VKFFT_ERROR_MATH_FAILED;
        return;
    }
    // Validate the entire map before touching any register. Duplicates cannot be
    // executed by swapping: this operation requires a bijective permutation.
    bool seen[maximum]{};
    for (pfUINT i = 0; i < count; ++i) {
        if (permutation[i] >= count || seen[permutation[i]] ||
            values[i * stride].type != values[0].type) {
            sc->res = VKFFT_ERROR_MATH_FAILED;
            return;
        }
        seen[permutation[i]] = true;
    }
    // The upstream operation only permutes named registers, not numeric constants.
    if (values[0].type <= 100) return;
    bool visited[maximum]{};
    for (pfUINT start = 0; start < count; ++start) {
        if (visited[start]) continue;
        pfUINT current = start;
        for (;;) {
            visited[current] = true;
            const pfUINT next = permutation[current];
            if (next == start) break;
            PfSwapContainers(sc, &values[current * stride], &values[next * stride]);
            if (sc->res != VKFFT_SUCCESS) return;
            current = next;
        }
    }
}

#endif
