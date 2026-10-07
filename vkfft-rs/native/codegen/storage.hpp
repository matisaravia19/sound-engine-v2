// Scoped ownership of the shader generator's temporary container arrays.
#ifndef VKFFT_RS_CODEGEN_STORAGE_HPP
#define VKFFT_RS_CODEGEN_STORAGE_HPP

#include "vkFFT/vkFFT_CodeGen/vkFFT_MathUtils/vkFFT_MathUtils.h"
#include <climits>

namespace vkfft_codegen {
namespace {

// Internal linkage keeps fault-injection allocators local to their translation unit.
// The specialization state outlives this owner; all initialized children are freed
// on normal return, early failure or exception, without clearing the recorded error.
class Containers {
  public:
    // Dimensions are checked even for an optional array whose contents are unused.
    explicit Containers(VkFFTSpecializationConstantsLayout* sc, pfINT count, pfINT copies = 1,
                        bool required = true)
        : sc_(sc) {
        if (sc_->res != VKFFT_SUCCESS)
            return;
        if (count <= 0 || copies <= 0 || count > INT_MAX / copies ||
            static_cast<uint64_t>(count) > SIZE_MAX / sizeof(PfContainer) / copies) {
            sc_->res = VKFFT_ERROR_MATH_FAILED;
            return;
        }
        if (!required)
            return;
        count_ = static_cast<size_t>(count) * static_cast<size_t>(copies);
        data_ = static_cast<PfContainer*>(calloc(count_, sizeof(PfContainer)));
        if (!data_)
            sc_->res = VKFFT_ERROR_MALLOC_FAILED;
    }
    Containers(const Containers&) = delete;
    Containers& operator=(const Containers&) = delete;

    // Zero-initialized tags make partial construction safe to unwind.
    ~Containers() {
        if (!data_)
            return;
        for (size_t i = 0; i < count_; ++i)
            PfDeallocateContainer(sc_, &data_[i]);
        free(data_);
    }

    // Borrowed storage remains valid only until this owner leaves scope.
    PfContainer* get() const {
        return data_;
    }

  private:
    VkFFTSpecializationConstantsLayout* sc_;
    PfContainer* data_ = nullptr;
    size_t count_ = 0;
};

} // namespace
} // namespace vkfft_codegen
#endif
