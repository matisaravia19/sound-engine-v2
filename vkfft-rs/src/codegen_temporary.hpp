// Ownership for individual temporaries in the native arithmetic generator.
#ifndef VKFFT_RS_CODEGEN_TEMPORARY_HPP
#define VKFFT_RS_CODEGEN_TEMPORARY_HPP

#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"

static inline void PfDeallocateContainer(VkFFTSpecializationConstantsLayout*, PfContainer*);

namespace vkfft_codegen {
namespace {

// The specialization state outlives this value. Partially initialized children
// unwind on allocation errors and exceptions, preserving the first error.
class Temporary {
public:
    explicit Temporary(VkFFTSpecializationConstantsLayout* sc) : sc_(sc) {}
    Temporary(const Temporary&) = delete;
    Temporary& operator=(const Temporary&) = delete;
    ~Temporary() { PfDeallocateContainer(sc_, &value_); }

    // Borrowed until this owner leaves scope; must not be shallow-copied.
    PfContainer& get() { return value_; }

private:
    VkFFTSpecializationConstantsLayout* sc_;
    PfContainer value_{};
};

} // namespace
} // namespace vkfft_codegen
#endif
