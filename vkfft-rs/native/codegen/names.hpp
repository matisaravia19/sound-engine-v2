// Adapted from VkFFT's container operations.
// Copyright (C) 2021 - present Dmitrii Tolmachev <dtolm96@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// Container operations preserve each destination's nested allocation ownership.
#ifndef VKFFT_RS_CODEGEN_NAMES_HPP
#define VKFFT_RS_CODEGEN_NAMES_HPP

#include "codegen/format.hpp"
#include <cstring>
#include <utility>

// A container's capacity also bounds reads; malformed names must not escape it.
static inline size_t PfNameLength(VkFFTSpecializationConstantsLayout* sc, const PfContainer* value) {
    if (sc->res != VKFFT_SUCCESS) return 0;
    if (value && value->name && value->size > 0) {
        const auto* end = static_cast<const char*>(std::memchr(value->name, 0, value->size));
        if (end) return static_cast<size_t>(end - value->name);
    }
    sc->res = VKFFT_ERROR_MATH_FAILED;
    return 0;
}

// The caller supplies a valid source span including its terminator. memmove
// permits self-copy and substring renaming without an extra allocation.
static inline void PfCopyName(VkFFTSpecializationConstantsLayout* sc, PfContainer* out,
                              const char* name, size_t length) {
    if (sc->res != VKFFT_SUCCESS) return;
    if (!out || !out->name || out->size <= 0 || !name || length >= static_cast<size_t>(out->size)) {
        sc->res = VKFFT_ERROR_MATH_FAILED;
        return;
    }
    std::memmove(out->name, name, length + 1);
}

// The prefix is a separate, terminated string (currently the literal "consts.").
static inline void PfPrefixContainerName(VkFFTSpecializationConstantsLayout* sc,
                                         PfContainer* out, const char* prefix) {
    const size_t length = PfNameLength(sc, out);
    if (sc->res != VKFFT_SUCCESS) return;
    const size_t prefix_length = std::strlen(prefix);
    if (prefix_length >= static_cast<size_t>(out->size) - length) {
        sc->res = VKFFT_ERROR_MATH_FAILED;
        return;
    }
    std::memmove(out->name + prefix_length, out->name, length + 1);
    std::memcpy(out->name, prefix, prefix_length);
}

// Only value/register tags have nested storage; opaque type names (>= 200) do not.
static inline PfContainer* PfContainerChildren(PfContainer* value) {
    if (value->type >= 200) return nullptr;
    if ((value->type % 100) / 10 == 3 && value->type % 10 == 2) return value->data.dd;
    if (value->type % 10 == 3) return value->data.c;
    return nullptr;
}

// Separate the tag check from its pointer so partially allocated values fail safely.
static inline bool PfContainerHasChildren(const PfContainer* value) {
    return value->type < 200 &&
        (((value->type % 100) / 10 == 3 && value->type % 10 == 2) || value->type % 10 == 3);
}

// Copy values/names, never the owning pointers in the tagged union.
static inline void PfCopyContainer(VkFFTSpecializationConstantsLayout* sc, PfContainer* out,
                                   PfContainer* in) {
    if (sc->res != VKFFT_SUCCESS) return;
    if (!out || !in || out->type <= 0 || out->type != in->type) {
        sc->res = VKFFT_ERROR_MATH_FAILED;
        return;
    }
    if (out->type > 100) {
        const size_t length = PfNameLength(sc, in);
        PfCopyName(sc, out, in->name, length);
        if (sc->res != VKFFT_SUCCESS) return;
    }
    if (PfContainerHasChildren(out)) {
        auto* destination = PfContainerChildren(out);
        auto* source = PfContainerChildren(in);
        if (!destination || !source) { sc->res = VKFFT_ERROR_MATH_FAILED; return; }
        for (int i = 0; i < 2; ++i) PfCopyContainer(sc, &destination[i], &source[i]);
    } else if (out->type < 100) {
        if (out->type % 10 == 1) out->data.i = in->data.i;
        else if (out->type % 10 == 2) out->data.d = in->data.d;
        else sc->res = VKFFT_ERROR_MATH_FAILED;
    }
}

// Preserve nested array addresses while exchanging their values and name owners.
static inline void PfSwapContainers(VkFFTSpecializationConstantsLayout* sc, PfContainer* out,
                                    PfContainer* in) {
    if (sc->res != VKFFT_SUCCESS) return;
    if (!out || !in || out->type <= 0 || out->type != in->type) {
        sc->res = VKFFT_ERROR_MATH_FAILED;
        return;
    }
    if (out->type > 100) {
        std::swap(out->name, in->name);
        std::swap(out->size, in->size);
    }
    if (PfContainerHasChildren(out)) {
        auto* destination = PfContainerChildren(out);
        auto* source = PfContainerChildren(in);
        if (!destination || !source) { sc->res = VKFFT_ERROR_MATH_FAILED; return; }
        for (int i = 0; i < 2; ++i) PfSwapContainers(sc, &destination[i], &source[i]);
    } else if (out->type < 100) {
        if (out->type % 10 == 1) std::swap(out->data.i, in->data.i);
        else if (out->type % 10 == 2) std::swap(out->data.d, in->data.d);
        else sc->res = VKFFT_ERROR_MATH_FAILED;
    }
}

// Append each component suffix within its own allocation, including nested doubles.
static inline void PfSetComponentNames(VkFFTSpecializationConstantsLayout* sc, PfContainer* out) {
    if (sc->res != VKFFT_SUCCESS || !PfContainerHasChildren(out)) return;
    auto* children = PfContainerChildren(out);
    if (!children) { sc->res = VKFFT_ERROR_MATH_FAILED; return; }
    for (int i = 0; i < 2; ++i) {
        PfFormatName(sc, &children[i], "%s.%c", out->name, i ? 'y' : 'x');
        PfSetComponentNames(sc, &children[i]);
    }
}

// Input is a terminated C string; it may alias this container's existing name.
static inline void PfSetContainerName(VkFFTSpecializationConstantsLayout* sc, PfContainer* out,
                                      const char* name) {
    if (sc->res != VKFFT_SUCCESS) return;
    if (!out || !name) { sc->res = VKFFT_ERROR_MATH_FAILED; return; }
    if (out->type <= 100) return;
    PfCopyName(sc, out, name, std::strlen(name));
    PfSetComponentNames(sc, out);
}

#endif
