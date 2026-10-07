// Bounded formatting for the shader generator's caller-owned buffers.
#ifndef VKFFT_RS_CODEGEN_FORMAT_HPP
#define VKFFT_RS_CODEGEN_FORMAT_HPP

#include "vkFFT/vkFFT_Structs/vkFFT_Structs.h"
#include <cstdarg>
#include <cstdio>

// A failed format records the error before any later append can consume truncation.
static inline int PfFormatTemp(VkFFTSpecializationConstantsLayout* sc, const char* format, ...) {
    if (sc->res != VKFFT_SUCCESS)
        return -1;
    if (!sc->tempStr || sc->maxTempLength <= 0 ||
        static_cast<uint64_t>(sc->maxTempLength) > SIZE_MAX) {
        sc->res = VKFFT_ERROR_INSUFFICIENT_TEMP_BUFFER;
        return -1;
    }
    va_list arguments;
    va_start(arguments, format);
    const int length =
        vsnprintf(sc->tempStr, static_cast<size_t>(sc->maxTempLength), format, arguments);
    va_end(arguments);
    if (length < 0 || length >= sc->maxTempLength)
        sc->res = VKFFT_ERROR_INSUFFICIENT_TEMP_BUFFER;
    return length;
}

// Arguments must not alias the destination, as with snprintf. Container copies
// and in-place prefixing use the overlap-safe helpers in codegen_names.hpp.
static inline int PfFormatName(VkFFTSpecializationConstantsLayout* sc, PfContainer* out,
                               const char* format, ...) {
    if (sc->res != VKFFT_SUCCESS)
        return -1;
    if (!out || !out->name || out->size <= 0) {
        sc->res = VKFFT_ERROR_MATH_FAILED;
        return -1;
    }
    va_list arguments;
    va_start(arguments, format);
    const int length = vsnprintf(out->name, static_cast<size_t>(out->size), format, arguments);
    va_end(arguments);
    if (length < 0 || length >= out->size)
        sc->res = VKFFT_ERROR_MATH_FAILED;
    return length;
}

#endif
