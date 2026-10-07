// C++ exceptions must be contained before returning through Rust's C ABI.
#ifndef VKFFT_RS_FFI_GUARD_HPP
#define VKFFT_RS_FFI_GUARD_HPP

#include <new>
#include <type_traits>

namespace vkfft_ffi {

// Rollback handles partially initialized native state after local RAII unwinds.
// Ordinary error returns retain the native function's existing cleanup contract.
template <class Operation, class Rollback>
VkFFTResult invoke(Operation operation, Rollback rollback, VkFFTResult unexpected) noexcept {
    static_assert(std::is_nothrow_invocable_v<Rollback>, "FFI rollback must not throw");
    try {
        return operation();
    } catch (const std::bad_alloc&) {
        rollback();
        return VKFFT_ERROR_MALLOC_FAILED;
    } catch (...) {
        rollback();
        return unexpected;
    }
}

} // namespace vkfft_ffi
#endif
