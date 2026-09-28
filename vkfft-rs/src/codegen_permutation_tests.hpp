// Real generator entry points, using the tracked CPU allocator from codegen_tests.cpp.
namespace codegen_fake {

int permutation_case(int count, int type, int stride, bool local) {
    state = {};
    VkFFTSpecializationConstantsLayout sc{};
    std::array<PfContainer, 66> registers{};
    auto* values = local ? sc.locID : registers.data();
    const int capacity = count * stride;
    std::array<PfContainer*, 66> owners{};
    for (int i = 0; i < capacity; ++i) {
        values[i].type = type;
        PfAllocateContainerFlexible(&sc, &values[i], 50);
        owners[i] = values[i].data.c;
    }
    if (sc.res != VKFFT_SUCCESS) return __LINE__;
    std::array<pfUINT, 33> permutation{};
    for (int i = 0; i < count; ++i) permutation[i] = i;
    const int baseline = live();
    state.attempts = 0;
    state.fail_at = 1; // Permuting already-owned storage must not allocate.
    bool repeat = true;
    while (repeat) {
        char name[50];
        for (int i = 0; i < capacity; ++i) {
            std::snprintf(name, sizeof(name), "r%d", i);
            PfSetContainerName(&sc, &values[i], name);
        }
        if (stride == 1) PfPermute(&sc, permutation.data(), count, local ? 0 : 1, values, nullptr);
        else PfPermuteContainers(&sc, permutation.data(), count, values, capacity, stride);
        if (sc.res != VKFFT_SUCCESS || state.attempts || live() != baseline) return __LINE__;
        for (int i = 0; i < capacity; ++i) {
            const auto expected = i % stride ? i : permutation[i / stride] * stride;
            std::snprintf(name, sizeof(name), "r%llu", static_cast<unsigned long long>(expected));
            if (std::strcmp(values[i].name, name) || values[i].data.c != owners[i]) return __LINE__;
            std::snprintf(name, sizeof(name), "r%llu.x", static_cast<unsigned long long>(expected));
            if (std::strcmp(values[i].data.c[0].name, name)) return __LINE__;
            if (type == 133) {
                std::snprintf(name, sizeof(name), "r%llu.y.x", static_cast<unsigned long long>(expected));
                if (std::strcmp(values[i].data.c[1].data.dd[0].name, name)) return __LINE__;
            }
        }
        if (count <= 7) {
            repeat = std::next_permutation(permutation.begin(), permutation.begin() + count);
        } else {
            // Large cases cover the capacity boundary and long cycles without n! work.
            std::rotate(permutation.begin(), permutation.begin() + 1, permutation.begin() + count);
            repeat = permutation[0] != 0;
        }
    }
    for (int i = 0; i < capacity; ++i) PfDeallocateContainer(&sc, &values[i]);
    return live() || state.invalid ? __LINE__ : 0;
}

int invalid_permutation_case() {
    state = {};
    VkFFTSpecializationConstantsLayout sc{};
    PfContainer values[33]{};
    for (int i = 0; i < 33; ++i) {
        values[i].type = 101;
        PfAllocateContainerFlexible(&sc, &values[i], 50);
        PfSetContainerName(&sc, &values[i], "original");
    }
    pfUINT map[34]{};
    for (pfUINT i = 0; i < 34; ++i) map[i] = i;
    const int baseline = live();
    state.attempts = 0;
    for (int mode = 0; mode < 10; ++mode) {
        sc.res = VKFFT_SUCCESS;
        map[1] = 1;
        switch (mode) {
            case 0: PfPermute(&sc, map, 34, 1, values, nullptr); break;
            case 1: map[1] = 2; PfPermute(&sc, map, 2, 1, values, nullptr); break;
            case 2: map[1] = 0; PfPermute(&sc, map, 2, 1, values, nullptr); break;
            case 3: PfPermute(&sc, nullptr, 2, 1, values, nullptr); break;
            case 4: PfPermute(&sc, map, 2, 1, nullptr, nullptr); break;
            case 5: PfPermute(&sc, map, 2, 2, values, nullptr); break;
            case 6: PfPermuteContainers(&sc, map, 2, values, 0); break;
            case 7: PfPermuteContainers(&sc, map, 2, values, 33, 0); break;
            case 8: PfPermuteContainers(&sc, map, 2, values, 33, UINT64_MAX); break;
            case 9: values[1].type = 111; PfPermute(&sc, map, 2, 1, values, nullptr); break;
        }
        values[1].type = 101;
        if (sc.res != VKFFT_ERROR_MATH_FAILED || live() != baseline || state.attempts) return __LINE__;
        for (const auto& value : values) if (std::strcmp(value.name, "original")) return __LINE__;
    }
    sc.res = VKFFT_SUCCESS;
    PfPermute(&sc, nullptr, 0, 1, nullptr, nullptr);
    if (sc.res != VKFFT_SUCCESS) return __LINE__;
    sc.res = VKFFT_ERROR_MALLOC_FAILED;
    PfPermute(&sc, nullptr, 34, 2, nullptr, nullptr);
    if (sc.res != VKFFT_ERROR_MALLOC_FAILED) return __LINE__;
    for (auto& value : values) PfDeallocateContainer(&sc, &value);
    return live() || state.invalid ? __LINE__ : 0;
}

int radix_allocation_case(int radix, bool double_precision, int stage, int sign) {
    state = {};
    VkFFTApplication app{};
    VkFFTAxis axis{};
    auto& sc = axis.specializationConstants;
    app.configuration.doublePrecision = double_precision;
    if (initialize_parameters(&app, &axis) != VKFFT_SUCCESS) return __LINE__;
    sc.maxCodeLength = 262144;
    sc.code0 = static_cast<char*>(allocate(sc.maxCodeLength, 1));
    if (!sc.code0) return __LINE__;
    sc.registers_per_thread = 32;
    sc.registerBoost = 1;
    sc.matrixConvolution = 1;
    sc.usedLocRegs = 33;
    sc.fftDim.type = sc.fft_dim_full.type = 31;
    sc.fftDim.data.i = sc.fft_dim_full.data.i = 64;
    sc.numStages = 1;
    sc.stageRadix[0] = 32;
    sc.maxNonPow2Radix = 13;
    sc.double_PI = 3.141592653589793238462643383279502884L;
    appendRegisterInitialization(&sc, 500);
    if (sc.res != VKFFT_SUCCESS) return __LINE__;
    const int baseline = live();
    state.attempts = 0;
    state.fail_at = 1;
    const auto previous_length = sc.currentLen;
    inlineRadixKernelVkFFT(&sc, radix, stage, 0, sign * sc.double_PI, sc.regIDs);
    if (sc.res != VKFFT_SUCCESS || state.attempts || live() != baseline || sc.currentLen <= previous_length)
        return __LINE__;
    freeRegisterInitialization(&sc, 500);
    deleteAxis(&app, &axis, 0);
    return live() || state.invalid ? __LINE__ : 0;
}

} // namespace codegen_fake

extern "C" int vkfft_test_codegen_permutations() {
    using namespace codegen_fake;
    for (int count : {1, 2, 3, 4, 5, 6, 7, 32, 33}) {
        for (int type : {113, 133}) {
            if (int line = permutation_case(count, type, 1, false)) return line;
            if (int line = permutation_case(count, type, 1, true)) return line;
            if (int line = permutation_case(count, type, 2, false)) return line;
        }
    }
    return invalid_permutation_case();
}

extern "C" int vkfft_test_codegen_radix_allocations() {
    for (bool double_precision : {false, true})
        for (int radix : {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32})
            for (int stage : {1, 4}) for (int sign : {-1, 1})
                if (int line = codegen_fake::radix_allocation_case(radix, double_precision, stage, sign)) return line;
    return 0;
}
