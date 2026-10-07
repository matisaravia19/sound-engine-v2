// Resource preconditions for the scheduler's signed-int capacities and divisions.
#ifndef VKFFT_RS_SCHEDULER_LIMITS_HPP
#define VKFFT_RS_SCHEDULER_LIMITS_HPP

#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include <limits>

namespace vkfft_scheduler {

// Normalize a positive coalescing hint to at least one complete complex element.
// Validate before mutation, including capacities used by later axis splitting.
static inline VkFFTResult prepare_limits(VkFFTConfiguration& c, int complex_size) {
    const auto signed_max = static_cast<pfUINT>((std::numeric_limits<int>::max)());
    if (complex_size <= 0 || !c.coalescedMemory || c.coalescedMemory > signed_max ||
        !c.maxThreadsNum || c.maxThreadsNum > signed_max || !c.warpSize ||
        c.warpSize > signed_max || !c.sharedMemorySize || c.sharedMemorySize > signed_max ||
        !c.sharedMemorySizePow2 || c.sharedMemorySizePow2 > c.sharedMemorySize ||
        (c.sharedMemorySizePow2 & (c.sharedMemorySizePow2 - 1)) || !c.registerBoost ||
        !c.registerBoost4Step || c.performBandwidthBoost > signed_max)
        return VKFFT_ERROR_MATH_FAILED;
    const auto bytes = static_cast<pfUINT>(complex_size);
    const auto coalesced = c.coalescedMemory < bytes ? bytes : c.coalescedMemory;
    if (coalesced > c.sharedMemorySizePow2)
        return VKFFT_ERROR_MATH_FAILED;
    c.coalescedMemory = coalesced;
    return VKFFT_SUCCESS;
}

// Direct Rader kernels reserve (prime - 1) complex elements. Leave one coalesced
// group available, and disable this route if a group has no available threads.
// prepare_limits has already made all divisors positive and bounded the products.
static inline pfUINT rader_multiplication_limit(const VkFFTConfiguration& c, int complex_size,
                                                int shared_bytes) {
    const auto bytes = static_cast<pfUINT>(complex_size);
    const auto threads = c.maxThreadsNum / (c.coalescedMemory / bytes);
    const auto thread_limit = threads ? 2 * threads - 1 : 0;
    const auto memory_limit = (static_cast<pfUINT>(shared_bytes) - c.coalescedMemory) / bytes + 2;
    auto limit = c.fixMaxRaderPrimeMult;
    if (limit > thread_limit)
        limit = thread_limit;
    if (limit > memory_limit)
        limit = memory_limit;
    return limit;
}

// Boost settings are upper bounds, not mandatory factors. Only consider factors
// whose virtual shared-memory capacity remains representable by the scheduler.
static inline pfUINT register_boost_limit(pfUINT requested, int shared_elements) {
    const auto capacity = static_cast<pfUINT>((std::numeric_limits<int>::max)() / shared_elements);
    return requested < capacity ? requested : capacity;
}

} // namespace vkfft_scheduler
#endif
