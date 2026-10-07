// Checked host arithmetic for native FFT factorization and Rader planning.
#ifndef VKFFT_RS_SCHEDULER_ARITHMETIC_HPP
#define VKFFT_RS_SCHEDULER_ARITHMETIC_HPP

#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include <climits>
#include <cstdint>

namespace vkfft_scheduler {

// The scheduler indexes fixed radix tables and implements ordinary primes only
// through 13. Bounds are exclusive; equal Rader limits intentionally disable it.
static inline bool factor_limits_valid(const VkFFTConfiguration& c) {
    return c.fixMinRaderPrimeMult >= 2 && c.fixMinRaderPrimeMult <= 17 &&
           c.fixMinRaderPrimeFFT >= c.fixMinRaderPrimeMult && c.fixMinRaderPrimeFFT <= INT_MAX &&
           c.fixMaxRaderPrimeMult <= INT_MAX && c.fixMaxRaderPrimeFFT <= INT_MAX &&
           (c.fixMaxRadixBluestein == 0 || (c.fixMaxRadixBluestein >= 2 && c.fixMaxRadixBluestein <= 13));
}

// Inputs come from a positive signed-int prime. Wide products cannot overflow,
// and repeated squaring avoids enumerating every power for each candidate root.
static inline uint32_t modular_power(uint32_t base, uint32_t exponent, uint32_t modulus) {
    uint64_t result = 1;
    uint64_t power = base;
    while (exponent) {
        if (exponent & 1)
            result = result * power % modulus;
        power = power * power % modulus;
        exponent >>= 1;
    }
    return static_cast<uint32_t>(result);
}

// Return the smallest primitive root, preserving the output on invalid input.
// For a prime p, a candidate has order p-1 iff none of its powers (p-1)/q is one,
// for distinct prime factors q of p-1. Reject composites before using this test.
static inline VkFFTResult primitive_root(int prime, int& generator) {
    static_assert(INT_MAX <= UINT32_MAX, "Rader primes must fit the modular arithmetic width");
    if (prime < 2)
        return VKFFT_ERROR_MATH_FAILED;
    for (int divisor = 2; divisor <= prime / divisor; ++divisor)
        if (prime % divisor == 0)
            return VKFFT_ERROR_MATH_FAILED;
    if (prime == 2) {
        generator = 1;
        return VKFFT_SUCCESS;
    }
    // A 32-bit positive integer has fewer than 32 distinct prime factors.
    int factors[32]{};
    int count = 0;
    int remaining = prime - 1;
    for (int divisor = 2; divisor <= remaining / divisor; ++divisor) {
        if (remaining % divisor)
            continue;
        factors[count++] = divisor;
        do {
            remaining /= divisor;
        } while (remaining % divisor == 0);
    }
    if (remaining > 1)
        factors[count++] = remaining;
    for (int candidate = 2; candidate < prime; ++candidate) {
        bool full_order = true;
        for (int i = 0; i < count; ++i) {
            if (modular_power(candidate, (prime - 1) / factors[i], prime) == 1) {
                full_order = false;
                break;
            }
        }
        if (full_order) {
            generator = candidate;
            return VKFFT_SUCCESS;
        }
    }
    return VKFFT_ERROR_MATH_FAILED;
}

// Divide the exact factor first; both the original product and the un-divided
// sequence can exceed int even when the resulting count fits. Publish only on success.
static inline VkFFTResult rader_fft_count(int radix_part, pfUINT sequence, int prime, int& count) {
    if (radix_part <= 0 || prime < 2 || !sequence || sequence % static_cast<pfUINT>(prime))
        return VKFFT_ERROR_MATH_FAILED;
    const auto quotient = sequence / static_cast<pfUINT>(prime);
    if (quotient > static_cast<pfUINT>(INT_MAX / radix_part))
        return VKFFT_ERROR_MATH_FAILED;
    count = radix_part * static_cast<int>(quotient);
    return VKFFT_SUCCESS;
}

} // namespace vkfft_scheduler
#endif
