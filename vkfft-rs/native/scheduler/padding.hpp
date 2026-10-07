// Bounded, allocation-free candidate search for Bluestein convolution lengths.
#ifndef VKFFT_RS_SCHEDULER_PADDING_HPP
#define VKFFT_RS_SCHEDULER_PADDING_HPP

#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include <limits>

namespace vkfft_scheduler {

// Supported ordinary prime factors, in the order used by the scheduler's tables.
static constexpr pfUINT padding_primes[] = {2, 3, 5, 7, 11, 13};

// Round in integer arithmetic; zero and an unrepresentable result leave output untouched.
static inline bool power_of_two_ceiling(pfUINT value, pfUINT& rounded) {
    if (!value)
        return false;
    auto bits = value - 1;
    for (unsigned shift = 1; shift < std::numeric_limits<pfUINT>::digits; shift *= 2)
        bits |= bits >> shift;
    if (bits == (std::numeric_limits<pfUINT>::max)())
        return false;
    rounded = bits + 1;
    return true;
}

// Register heuristics need only this comparison, not a potentially overflowing product.
static inline bool product_at_least(pfUINT left, pfUINT right, pfUINT threshold) {
    if (!threshold)
        return true;
    return right && left >= threshold / right + (threshold % right != 0);
}

// Factor a positive candidate into the fixed radix table. Composite radix limits
// have the same meaning as the old trial-division loop (e.g. four admits two/three).
static inline bool padding_factors(pfUINT value, int max_radix, int (&factors)[33]) {
    for (auto& factor : factors)
        factor = 0;
    if (!value || max_radix < 2 || max_radix > 13)
        return false;
    for (auto prime : padding_primes) {
        if (prime > static_cast<pfUINT>(max_radix))
            break;
        while (value % prime == 0) {
            value /= prime;
            ++factors[prime];
        }
    }
    return value == 1;
}

// Enumerate supported prime powers, pruning against the best complete candidate.
// Recursion is at most six levels; the power-of-two leaf is rounded directly.
static inline void search_padding_products(pfUINT minimum, int prime_index, pfUINT product, pfUINT& best,
                                           bool& found) {
    if (product > best)
        return;
    if (product >= minimum) {
        best = product;
        found = true;
        return;
    }
    if (prime_index == 0) {
        const auto multiplier = minimum / product + (minimum % product != 0);
        pfUINT power = 0;
        if (power_of_two_ceiling(multiplier, power) && power <= best / product) {
            best = power * product;
            found = true;
        }
        return;
    }
    const auto prime = padding_primes[prime_index];
    for (;;) {
        search_padding_products(minimum, prime_index - 1, product, best, found);
        if (product > best / prime)
            break;
        product *= prime;
    }
}

// Return the smallest representable supported length >= minimum. Large gaps
// between candidates require no linear scan or temporary container allocations.
static inline VkFFTResult next_smooth_padding(pfUINT minimum, int max_radix, pfUINT& padded) {
    if (!minimum || max_radix < 2 || max_radix > 13)
        return VKFFT_ERROR_MATH_FAILED;
    int factors[33];
    if (padding_factors(minimum, max_radix, factors)) {
        padded = minimum;
        return VKFFT_SUCCESS;
    }
    pfUINT best = (std::numeric_limits<pfUINT>::max)();
    bool found = power_of_two_ceiling(minimum, best);
    int prime_index = 5;
    while (padding_primes[prime_index] > static_cast<pfUINT>(max_radix))
        --prime_index;
    search_padding_products(minimum, prime_index, 1, best, found);
    if (!found)
        return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;
    padded = best;
    return VKFFT_SUCCESS;
}

// Preserve VkFFT's three-quarter occupancy preference without floating-point
// rounding or signed-int narrowing at large sequence lengths.
static inline pfUINT preferred_padding(pfUINT length, pfUINT minimum, pfUINT candidate, pfUINT shared_limit) {
    pfUINT rounded = 0;
    if (power_of_two_ceiling(candidate, rounded) &&
        (length < 128 ||
         (candidate >= rounded - rounded / 4 && (rounded <= shared_limit || minimum > shared_limit))))
        return rounded;
    return candidate;
}

// The callback evaluates the existing register heuristic for default radix-seven
// candidates. Explicit radix limits select the first smooth length directly.
// On callback failure or exhausted representable lengths, leave padded unchanged.
template <class Accept>
static inline VkFFTResult automatic_bluestein_padding(pfUINT length, pfUINT minimum, int max_radix,
                                                      int shared_limit, Accept accept, pfUINT& padded) {
    if (!length || !minimum || shared_limit <= 0)
        return VKFFT_ERROR_MATH_FAILED;
    if (max_radix)
        return next_smooth_padding(minimum, max_radix, padded);
    auto candidate = minimum;
    for (;;) {
        candidate = preferred_padding(length, minimum, candidate, static_cast<pfUINT>(shared_limit));
        auto result = next_smooth_padding(candidate, 7, candidate);
        if (result != VKFFT_SUCCESS)
            return result;
        // A skipped gap may cross the point where the old scan preferred a power of two.
        candidate = preferred_padding(length, minimum, candidate, static_cast<pfUINT>(shared_limit));
        int factors[33];
        if (!padding_factors(candidate, 7, factors))
            return VKFFT_ERROR_MATH_FAILED;
        bool accepted = false;
        result = accept(candidate, factors, accepted);
        if (result != VKFFT_SUCCESS)
            return result;
        if (accepted) {
            padded = candidate;
            return VKFFT_SUCCESS;
        }
        if (candidate == (std::numeric_limits<pfUINT>::max)())
            return VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH;
        ++candidate;
    }
}

} // namespace vkfft_scheduler
#endif
