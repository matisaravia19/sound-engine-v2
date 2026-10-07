// Included by codegen_tests.cpp so these cases use the same tracked allocator.
namespace codegen_fake {

int name_buffer_case(int mode, const char* input, VkFFTResult expected) {
    VkFFTSpecializationConstantsLayout sc{};
    std::array<char, 80> storage;
    storage.fill('#');
    storage[0] = 0;
    PfContainer out{};
    out.type = 101;
    out.name = storage.data();
    out.size = 8;
    if (mode == 0) PfSetContainerName(&sc, &out, input);
    if (mode == 1) PfFormatName(&sc, &out, "%s", input);
    if (mode == 2) {
        PfContainer in{};
        in.type = 101;
        in.name = const_cast<char*>(input);
        in.size = static_cast<int>(std::strlen(input) + 1);
        PfCopyContainer(&sc, &out, &in);
    }
    if (sc.res != expected) return __LINE__;
    for (size_t i = 8; i < storage.size(); ++i) if (storage[i] != '#') return __LINE__;
    if (expected == VKFFT_SUCCESS) {
        PfSetContainerName(&sc, &out, out.name);
        PfCopyContainer(&sc, &out, &out);
        if (sc.res != VKFFT_SUCCESS || std::strcmp(out.name, input)) return __LINE__;
    }
    const auto before = storage;
    sc.res = VKFFT_ERROR_MALLOC_FAILED;
    PfSetContainerName(&sc, &out, "new");
    PfFormatName(&sc, &out, "%s", "new");
    PfCopyContainer(&sc, &out, &out);
    PfPrefixContainerName(&sc, &out, "prefix");
    return storage != before || sc.res != VKFFT_ERROR_MALLOC_FAILED ? __LINE__ : 0;
}

int component_names_case(const char* input, VkFFTResult expected) {
    VkFFTSpecializationConstantsLayout sc{};
    PfContainer root{}, children[2]{}, low[2]{}, high[2]{};
    std::array<std::array<char, 32>, 7> storage;
    PfContainer* nodes[] = {&root, &children[0], &children[1], &low[0], &low[1], &high[0], &high[1]};
    for (int i = 0; i < 7; ++i) {
        storage[i].fill('#');
        storage[i][0] = 0;
        nodes[i]->type = i == 0 ? 133 : i < 3 ? 132 : 122;
        nodes[i]->name = storage[i].data();
        nodes[i]->size = 8;
    }
    root.data.c = children;
    children[0].data.dd = low;
    children[1].data.dd = high;
    PfSetContainerName(&sc, &root, input);
    if (sc.res != expected) return __LINE__;
    if (expected == VKFFT_SUCCESS) {
        PfSetContainerName(&sc, &root, root.name);
        if (sc.res != VKFFT_SUCCESS || std::strcmp(low[0].name, "abc.x.x") ||
            std::strcmp(low[1].name, "abc.x.y") || std::strcmp(high[0].name, "abc.y.x") ||
            std::strcmp(high[1].name, "abc.y.y")) return __LINE__;
    }
    for (auto& buffer : storage)
        for (size_t i = 8; i < buffer.size(); ++i) if (buffer[i] != '#') return __LINE__;
    return 0;
}

int prefix_and_invalid_names_case() {
    VkFFTSpecializationConstantsLayout sc{};
    char storage[16] = "";
    std::memset(storage + 8, '#', 8);
    PfContainer value{};
    value.type = 101;
    value.name = storage;
    value.size = 8;
    PfPrefixContainerName(&sc, &value, "consts.");
    if (sc.res != VKFFT_SUCCESS || std::strcmp(storage, "consts.")) return __LINE__;
    PfPrefixContainerName(&sc, &value, "x");
    if (sc.res != VKFFT_ERROR_MATH_FAILED || std::strcmp(storage, "consts.")) return __LINE__;
    sc.res = VKFFT_SUCCESS;
    std::memset(storage, 'x', 8);
    PfCopyContainer(&sc, &value, &value);
    if (sc.res != VKFFT_ERROR_MATH_FAILED) return __LINE__;
    sc.res = VKFFT_SUCCESS;
    PfPrefixContainerName(&sc, &value, "prefix");
    if (sc.res != VKFFT_ERROR_MATH_FAILED) return __LINE__;
    for (int i = 8; i < 16; ++i) if (storage[i] != '#') return __LINE__;
    for (int size : {0, -1}) {
        sc.res = VKFFT_SUCCESS;
        value.size = size;
        PfFormatName(&sc, &value, "%s", "x");
        if (sc.res != VKFFT_ERROR_MATH_FAILED) return __LINE__;
    }
    sc.res = VKFFT_SUCCESS;
    value.size = 8;
    value.name = nullptr;
    PfFormatName(&sc, &value, "%s", "x");
    return sc.res == VKFFT_ERROR_MATH_FAILED ? 0 : __LINE__;
}

int numeric_double_copy_case(bool complex) {
    state = {};
    VkFFTSpecializationConstantsLayout sc{};
    PfContainer left{}, right{};
    left.type = right.type = complex ? 33 : 32;
    PfAllocateContainerFlexible(&sc, &left, 50);
    PfAllocateContainerFlexible(&sc, &right, 50);
    if (sc.res != VKFFT_SUCCESS) return __LINE__;
    auto* left_owner = left.data.dd;
    auto* right_owner = right.data.dd;
    auto* left_parts = complex ? left.data.c[0].data.dd : left.data.dd;
    auto* right_parts = complex ? right.data.c[0].data.dd : right.data.dd;
    right_parts[0].data.d = 12;
    right_parts[1].data.d = 0.25;
    PfCopyContainer(&sc, &left, &right);
    if (sc.res != VKFFT_SUCCESS || left.data.dd != left_owner || right.data.dd != right_owner ||
        (complex && (left.data.c[0].data.dd != left_parts || right.data.c[0].data.dd != right_parts)) ||
        left_parts[0].data.d != 12 || left_parts[1].data.d != 0.25) return __LINE__;
    left_parts[0].data.d = 27;
    PfSwapContainers(&sc, &left, &right);
    if (sc.res != VKFFT_SUCCESS || left.data.dd != left_owner || right.data.dd != right_owner ||
        (complex && (left.data.c[0].data.dd != left_parts || right.data.c[0].data.dd != right_parts)) ||
        left_parts[0].data.d != 12 || right_parts[0].data.d != 27) return __LINE__;
    PfCopyContainer(&sc, &left, &left);
    PfSwapContainers(&sc, &right, &right);
    PfDeallocateContainer(&sc, &left);
    PfDeallocateContainer(&sc, &right);
    return live() || state.invalid ? __LINE__ : 0;
}

int opaque_name_copy_case(int type) {
    state = {};
    VkFFTSpecializationConstantsLayout sc{};
    PfContainer left{}, right{};
    left.type = right.type = type;
    PfAllocateContainerFlexible(&sc, &left, 50);
    PfAllocateContainerFlexible(&sc, &right, 50);
    PfSetContainerName(&sc, &left, "left");
    PfSetContainerName(&sc, &right, "right");
    PfSwapContainers(&sc, &left, &right);
    if (sc.res != VKFFT_SUCCESS || std::strcmp(left.name, "right") ||
        std::strcmp(right.name, "left")) return __LINE__;
    PfCopyContainer(&sc, &right, &left);
    if (sc.res != VKFFT_SUCCESS || std::strcmp(right.name, "right")) return __LINE__;
    PfDeallocateContainer(&sc, &left);
    PfDeallocateContainer(&sc, &right);
    return live() || state.invalid ? __LINE__ : 0;
}

int parameter_name_failure_case() {
    state = {};
    VkFFTApplication app{};
    VkFFTAxis axis{};
    auto& sc = axis.specializationConstants;
    sc.maxTempLength = 5000;
    sc.swapComputeWorkGroupID = 1;
    sc.localSize[0].data.i = std::numeric_limits<pfINT>::max();
    if (initMemoryParametersAPI(&app, &sc) != VKFFT_SUCCESS) return __LINE__;
    sc.inputMemoryCode = sc.vecTypeInputMemoryCode;
    sc.outputMemoryCode = sc.vecTypeOutputMemoryCode;
    if (initParametersAPI(&app, &sc) != VKFFT_ERROR_MATH_FAILED) return __LINE__;
    deleteAxis(&app, &axis, 0);
    return live() || state.invalid ? __LINE__ : 0;
}

int kernel_name_allocation_failure_case() {
    state = {};
    VkFFTApplication app{};
    VkFFTAxis axis{};
    auto& sc = axis.specializationConstants;
    if (initialize_parameters(&app, &axis) != VKFFT_SUCCESS) return __LINE__;
    const int baseline = live();
    state.fail_at = state.attempts + 1;
    appendSinCos20(&sc);
    if (sc.res != VKFFT_ERROR_MALLOC_FAILED || live() != baseline) return __LINE__;
    deleteAxis(&app, &axis, 0);
    return live() || state.invalid ? __LINE__ : 0;
}

} // namespace codegen_fake

extern "C" int vkfft_test_codegen_names() {
    using namespace codegen_fake;
    for (int mode : {0, 1, 2}) {
        for (const char* input : {"", "1234567", "12345678", "a much longer container name"}) {
            const auto expected = std::strlen(input) < 8 ? VKFFT_SUCCESS : VKFFT_ERROR_MATH_FAILED;
            if (int line = name_buffer_case(mode, input, expected)) return line;
        }
    }
    if (int line = component_names_case("abc", VKFFT_SUCCESS)) return line;
    if (int line = component_names_case("abcd", VKFFT_ERROR_MATH_FAILED)) return line;
    if (int line = prefix_and_invalid_names_case()) return line;
    if (int line = parameter_name_failure_case()) return line;
    return kernel_name_allocation_failure_case();
}

extern "C" int vkfft_test_codegen_copy_ownership() {
    using namespace codegen_fake;
    for (bool complex : {false, true})
        if (int line = numeric_double_copy_case(complex)) return line;
    for (int type : {201, 213, 232, 233, 300, 313, 323, 332, 333})
        if (int line = opaque_name_copy_case(type)) return line;
    VkFFTSpecializationConstantsLayout sc{};
    PfContainer left{}, right{};
    left.type = 132;
    right.type = 101;
    PfCopyContainer(&sc, &left, &right);
    if (sc.res != VKFFT_ERROR_MATH_FAILED) return __LINE__;
    sc.res = VKFFT_SUCCESS;
    PfSwapContainers(&sc, &left, &right);
    return sc.res == VKFFT_ERROR_MATH_FAILED ? 0 : __LINE__;
}
