// Fixed-capacity stage construction and recursive scheduler metadata checks.
#ifndef VKFFT_RS_SCHEDULER_STAGES_HPP
#define VKFFT_RS_SCHEDULER_STAGES_HPP

#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include <climits>

namespace vkfft_scheduler {

// Arrays are borrowed from the owning plan. Preserve both arrays and the count
// on failure; successful appends always keep radix/generator metadata together.
template <size_t N>
static inline VkFFTResult append_stage(int& count, int (&radices)[N], int (&generators)[N],
                                       int radix, int generator) {
    if (count < 0 || static_cast<size_t>(count) >= N || radix < 2 || generator < 0 ||
        generator >= radix)
        return VKFFT_ERROR_MATH_FAILED;
    radices[count] = radix;
    generators[count] = generator;
    ++count;
    return VKFFT_SUCCESS;
}

// The plan owns an acyclic factor tree. Bound traversal as a defensive check:
// signed-int prime factors cannot require 33 levels of decreasing factorization.
static inline bool tree_list_valid(const VkFFTRaderContainer* containers, int count,
                                   unsigned depth) {
    return depth < 33 && count >= 0 && count <= 33 && (!count || containers);
}

// Resolve only a matching prime/generator pair; missing or duplicate metadata
// must not leave register sizing dependent on a previous or unrelated stage.
static inline const VkFFTRaderContainer* find_rader(const VkFFTRaderContainer* containers,
                                                    int count, int prime, int generator) {
    const VkFFTRaderContainer* found = nullptr;
    for (int i = 0; i < count; ++i) {
        const auto& candidate = containers[i];
        if (candidate.prime != prime)
            continue;
        if (found || generator <= 0 || generator >= prime || candidate.generator != generator ||
            (candidate.type != 0 && candidate.type != 1))
            return nullptr;
        found = &candidate;
    }
    return found;
}

// Inspect each selected subtree once, using its own stage indexes. Callers keep
// local extrema until validation succeeds, so failure cannot publish partial sizing.
template <size_t N>
static inline VkFFTResult
register_limits(int count, const int (&radices)[N], const int (&generators)[N],
                const int (&registers)[33], const VkFFTRaderContainer* containers,
                int container_count, int& minimum, int& maximum, unsigned depth = 0) {
    if (count < 0 || static_cast<size_t>(count) > N ||
        !tree_list_valid(containers, container_count, depth))
        return VKFFT_ERROR_MATH_FAILED;
    for (int stage = 0; stage < count; ++stage) {
        const int radix = radices[stage];
        if (generators[stage] == 0) {
            if (radix < 2 || radix >= 33 || registers[radix] < 0)
                return VKFFT_ERROR_MATH_FAILED;
            const int value = registers[radix];
            if (value > 0) {
                if (value < minimum)
                    minimum = value;
                if (value > maximum)
                    maximum = value;
            }
        } else {
            const auto* rader = find_rader(containers, container_count, radix, generators[stage]);
            if (!rader)
                return VKFFT_ERROR_MATH_FAILED;
            if (rader->type == 0) {
                if (rader->numStages <= 0)
                    return VKFFT_ERROR_MATH_FAILED;
                auto result = register_limits(
                    rader->numStages, rader->stageRadix, rader->stage_rader_generator,
                    rader->registers_per_thread_per_radix, rader->container, rader->numSubPrimes,
                    minimum, maximum, depth + 1);
                if (result != VKFFT_SUCCESS)
                    return result;
            }
        }
    }
    return VKFFT_SUCCESS;
}

// Ceil division uses positive integers, and the product is checked before it is
// stored in the scheduler's signed-int thread count. No floating conversion is needed.
static inline VkFFTResult rader_threads(const VkFFTRaderContainer* containers, int count,
                                        int& maximum, unsigned depth = 0) {
    if (!tree_list_valid(containers, count, depth))
        return VKFFT_ERROR_MATH_FAILED;
    for (int i = 0; i < count; ++i) {
        const auto& rader = containers[i];
        if (rader.type == 1)
            continue;
        constexpr int stages = sizeof(rader.stageRadix) / sizeof(rader.stageRadix[0]);
        if (rader.type != 0 || rader.numStages <= 0 || rader.numStages > stages ||
            rader.containerFFTDim <= 0 || rader.containerFFTNum <= 0)
            return VKFFT_ERROR_MATH_FAILED;
        auto result = rader_threads(rader.container, rader.numSubPrimes, maximum, depth + 1);
        if (result != VKFFT_SUCCESS)
            return result;
        for (int stage = 0; stage < rader.numStages; ++stage) {
            const int radix = rader.stageRadix[stage];
            if (rader.stage_rader_generator[stage] != 0) {
                if (!find_rader(rader.container, rader.numSubPrimes, radix,
                                rader.stage_rader_generator[stage]))
                    return VKFFT_ERROR_MATH_FAILED;
                continue;
            }
            if (radix < 2 || radix >= 33 || rader.registers_per_thread_per_radix[radix] <= 0)
                return VKFFT_ERROR_MATH_FAILED;
            const int registers = rader.registers_per_thread_per_radix[radix];
            const int groups =
                rader.containerFFTDim / registers + (rader.containerFFTDim % registers != 0);
            if (rader.containerFFTNum > INT_MAX / groups)
                return VKFFT_ERROR_MATH_FAILED;
            const int threads = rader.containerFFTNum * groups;
            if (threads > maximum)
                maximum = threads;
        }
    }
    return VKFFT_SUCCESS;
}

// Rader primes index a separate container list, never the 33-slot ordinary-radix
// register table. Move the first matching ordinary stage with its generator tag.
static inline VkFFTResult prioritize_register_stage(VkFFTSpecializationConstantsLayout& sc,
                                                    int minimum) {
    constexpr int capacity = sizeof(sc.stageRadix) / sizeof(sc.stageRadix[0]);
    if (sc.numStages < 0 || sc.numStages > capacity)
        return VKFFT_ERROR_MATH_FAILED;
    for (int stage = 0; stage < sc.numStages; ++stage) {
        if (sc.rader_generator[stage] != 0)
            continue;
        const int radix = sc.stageRadix[stage];
        if (radix < 2 || radix >= 33)
            return VKFFT_ERROR_MATH_FAILED;
        if (sc.registers_per_thread_per_radix[radix] == minimum) {
            const int first_radix = sc.stageRadix[0];
            const int first_generator = sc.rader_generator[0];
            sc.stageRadix[0] = radix;
            sc.rader_generator[0] = 0;
            sc.stageRadix[stage] = first_radix;
            sc.rader_generator[stage] = first_generator;
            break;
        }
    }
    return VKFFT_SUCCESS;
}

} // namespace vkfft_scheduler
#endif
