// Host storage for VkFFT lookup tables, independent of application teardown.
#ifndef VKFFT_RS_LUT_STORAGE_HPP
#define VKFFT_RS_LUT_STORAGE_HPP

#include <cstdint>
#include <cstdlib>

namespace vkfft_lut {
namespace {

// Internal linkage keeps CPU fault-injection allocators out of production code.
// Uploads are synchronous; storage survives through transfer completion and is
// reclaimed on every return or exception, even after deleteVkFFT clears the axis.
class HostBuffer {
  public:
    explicit HostBuffer(pfUINT bytes) {
        // Empty fallback tables and any unused padding must not expose host data.
        if (bytes && bytes <= SIZE_MAX)
            data_ = calloc(1, static_cast<size_t>(bytes));
    }
    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;
    ~HostBuffer() {
        free(data_);
    }

    // Borrowed data remains valid until this local owner leaves scope.
    void* get() const {
        return data_;
    }

  private:
    void* data_ = nullptr;
};

// Check the element-to-byte conversion before assigning native allocation sizes.
static inline bool byte_size(pfUINT count, size_t element_bytes, size_t minimum, pfUINT* result) {
    *result = 0;
    if (!element_bytes || count > SIZE_MAX / element_bytes)
        return false;
    *result = count ? count * element_bytes : minimum;
    return *result != 0;
}

} // namespace
} // namespace vkfft_lut
#endif
