// Reuse and teardown tests against the native builders and deleteAxis.
#ifndef VKFFT_RS_LUT_SHARING_TESTS_HPP
#define VKFFT_RS_LUT_SHARING_TESTS_HPP

namespace lut_fake {
struct SharingFixture {
    VkFFTApplication app{};
    std::unique_ptr<VkFFTPlan> plan = std::make_unique<VkFFTPlan>();
    std::unique_ptr<VkFFTPlan> inverse = std::make_unique<VkFFTPlan>();
    VkDevice device = handle<VkDevice>(9);
    VkFFTAxis* target = nullptr;
    VkFFTAxis* source = nullptr;
    pfUINT bytes = 0;
    int route;

    // Routes exercise inverse, Bluestein, earlier-axis and R2C table reuse.
    SharingFixture(int path, int precision) : route(path) {
        state = {};
        app.configuration.device = &device;
        app.configuration.useLUT = 1;
        app.configuration.makeForwardPlanOnly = path == 1 || path == 2;
        app.configuration.doublePrecision = precision == 1;
        app.configuration.doublePrecisionFloatMemory = precision == 2;
        app.configuration.quadDoubleDoublePrecision = precision == 3;
        app.configuration.quadDoubleDoublePrecisionDoubleMemory = precision == 4;
        for (auto& size : app.configuration.size)
            size = 16;
        app.localFFTPlan_inverse = inverse.get();
        target = path == 1   ? &plan->inverseBluesteinAxes[0][1]
                 : path == 2 ? &plan->axes[2][1]
                 : path == 3 ? &plan->R2Cdecomposition
                             : &plan->axes[0][0];
        source = path == 1   ? &plan->axes[0][1]
                 : path == 2 ? &plan->axes[1][1]
                 : path == 3 ? &inverse->R2Cdecomposition
                             : &inverse->axes[0][0];
        auto& sc = target->specializationConstants;
        sc.axis_id = path == 2 ? 2 : 0;
        sc.axis_upload_id = path == 1 || path == 2 ? 1 : 0;
        sc.numAxisUploads = sc.axis_upload_id + 1;
        sc.reverseBluesteinMultiUpload = path == 1;
        sc.registerBoost = 1;
        sc.numStages = 2;
        sc.stageRadix[0] = sc.stageRadix[1] = 4;
        sc.fft_dim_full.data.i = sc.fftDim.data.i = 16;
        source->specializationConstants = sc;
        auto& other = source->specializationConstants;
        other.axis_id = path == 2 ? 1 : 0;
        other.reverseBluesteinMultiUpload = 0;
        other.maxStageSumLUT = 8;
        plan->numAxisUploads[other.axis_id] = other.numAxisUploads;
        // Slot zero has the expected shape even when the selected slot differs.
        if (path == 2)
            plan->axes[1][0].specializationConstants = other;
        bytes = 8 * (precision >= 3 ? 4 : 2) * (precision ? sizeof(double) : sizeof(float));
        source->bufferLUT = handle<VkBuffer>(5);
        source->bufferLUTDeviceMemory = handle<VkDeviceMemory>(6);
        source->bufferLUTSize = bytes;
    }
    VkFFTResult build(bool missing_plan = false) {
        auto* current = missing_plan ? nullptr : plan.get();
        return route == 3 ? VkFFT_AllocateLUT_R2C(&app, current, target, 0)
                          : VkFFT_AllocateLUT(&app, current, target, 0);
    }
};
} // namespace lut_fake

#define SHARING_CHECK(condition)                                                                   \
    do {                                                                                           \
        if (!(condition))                                                                          \
            return __LINE__;                                                                       \
    } while (false)

extern "C" int vkfft_test_lut_sharing_compatibility() {
    using namespace lut_fake;
    for (int route = 0; route < 4; ++route) {
        for (int precision = 0; precision < 5; ++precision) {
            for (int variant = 0; variant < 14; ++variant) {
                // R2C content depends on the application's length/precision,
                // not the complex FFT's stage metadata.
                if (route == 3 && variant >= 5 && variant != 10 && variant != 12)
                    continue;
                SharingFixture fixture(route, precision);
                auto& source = *fixture.source;
                auto& sc = source.specializationConstants;
                switch (variant) {
                    case 1:
                        source.bufferLUT = VK_NULL_HANDLE;
                        break;
                    case 2:
                        source.bufferLUTDeviceMemory = VK_NULL_HANDLE;
                        break;
                    case 3:
                        --source.bufferLUTSize;
                        break;
                    case 4:
                        ++source.bufferLUTSize;
                        break;
                    case 5:
                        sc.numStages = 3;
                        sc.stageRadix[2] = 2;
                        break;
                    case 6:
                        sc.stageRadix[0] = 2;
                        break;
                    case 7:
                        sc.rader_generator[1] = 3;
                        break;
                    case 8:
                        sc.maxStageSumLUT = 4;
                        break;
                    case 9:
                        sc.fft_dim_full.data.i = 32;
                        break;
                    case 10:
                        sc.res = VKFFT_ERROR_MALLOC_FAILED;
                        break;
                    case 11:
                        sc.numRaderPrimes = 1;
                        sc.raderContainer = nullptr;
                        break;
                    case 12:
                        fixture.app.localFFTPlan_inverse = nullptr;
                        break;
                    case 13:
                        sc.numAxisUploads = 0;
                        break;
                }
                state.failure = variant ? None : Host;
                const bool missing_plan = variant == 12 && (route == 1 || route == 2);
                SHARING_CHECK(fixture.build(missing_plan) == VKFFT_SUCCESS);
                SHARING_CHECK(!live() && !state.invalid);
                SHARING_CHECK(fixture.target->bufferLUTSize == fixture.bytes);
                if (variant == 0) {
                    SHARING_CHECK(fixture.target->referenceLUT == 1 &&
                                  fixture.target->bufferLUT == source.bufferLUT);
                    SHARING_CHECK(fixture.target->bufferLUTDeviceMemory ==
                                  source.bufferLUTDeviceMemory);
                    SHARING_CHECK(!state.attempts && !state.device_calls && !state.transfer_calls);
                } else {
                    SHARING_CHECK(!fixture.target->referenceLUT &&
                                  fixture.target->bufferLUT == handle<VkBuffer>(1));
                    SHARING_CHECK(state.attempts == 1 && state.device_calls == 1 &&
                                  state.transfer_calls == 1);
                    SHARING_CHECK(state.uploaded == fixture.bytes);
                }
            }
        }
    }
    // Rader metadata and cached kernel identity are part of the table's content.
    for (int variant = 0; variant < 10; ++variant) {
        SharingFixture fixture(0, 0);
        std::array<float, 32> kernel{}, alternate{};
        VkFFTRaderContainer target_rader{}, source_rader{};
        target_rader.prime = 17;
        target_rader.generator = 3;
        target_rader.numStages = 1;
        target_rader.stageRadix[0] = 16;
        target_rader.raderFFTkernel = kernel.data();
        source_rader = target_rader;
        auto& target = fixture.target->specializationConstants;
        auto& source = fixture.source->specializationConstants;
        target.useRader = source.useRader = 1;
        target.numRaderPrimes = source.numRaderPrimes = 1;
        target.raderContainer = &target_rader;
        source.raderContainer = &source_rader;
        fixture.source->bufferLUTSize = 24 * 2 * sizeof(float);
        switch (variant) {
            case 1:
                source.raderContainer = nullptr;
                break;
            case 2:
                source.numRaderPrimes = 0;
                break;
            case 3:
                source_rader.prime = 19;
                break;
            case 4:
                source_rader.generator = 4;
                break;
            case 5:
                source_rader.type = 1;
                break;
            case 6:
                source_rader.numStages = 2;
                source_rader.stageRadix[1] = 2;
                break;
            case 7:
                source_rader.stageRadix[0] = 8;
                break;
            case 8:
                source_rader.raderFFTkernel = alternate.data();
                break;
            case 9:
                source_rader.numStages = 21;
                break;
        }
        SHARING_CHECK(fixture.build() == VKFFT_SUCCESS);
        SHARING_CHECK(fixture.target->referenceLUT == (variant == 0));
        SHARING_CHECK(state.attempts == (variant != 0) && !live() && !state.invalid);
    }
    SharingFixture unavailable(0, 0);
    unavailable.source->bufferLUTSize = 1;
    state.failure = Host;
    SHARING_CHECK(unavailable.build() == VKFFT_ERROR_MALLOC_FAILED);
    SHARING_CHECK(!unavailable.target->referenceLUT && state.attempts == 1 && !state.device_calls &&
                  !live());

    // Matching buffer sizes alone do not establish matching four-step geometry.
    SharingFixture four_step(1, 0);
    four_step.app.configuration.useLUT_4step = 1;
    auto& a = four_step.target->specializationConstants;
    auto& b = four_step.source->specializationConstants;
    a.stageStartSize.data.i = b.stageStartSize.data.i = 4;
    a.fftDim.data.i = 4;
    b.fftDim.data.i = 8;
    four_step.source->bufferLUTSize = 24 * 2 * sizeof(float);
    SHARING_CHECK(four_step.build() == VKFFT_SUCCESS && !four_step.target->referenceLUT);
    SHARING_CHECK(state.uploaded == 24 * 2 * sizeof(float) && !live() && !state.invalid);

    // Reject a self-reference without changing ownership or the existing handle.
    auto& axis = *four_step.target;
    SHARING_CHECK(!vkfft_lut::borrow_table(axis, &axis) && !axis.referenceLUT);
    return 0;
}

extern "C" int vkfft_test_lut_shared_teardown() {
    using namespace lut_fake;
    for (int route = 0; route < 4; ++route) {
        for (bool owner_first : {false, true}) {
            SharingFixture fixture(route, 0);
            SHARING_CHECK(fixture.build() == VKFFT_SUCCESS && fixture.target->referenceLUT);
            auto* first = owner_first ? fixture.source : fixture.target;
            auto* last = owner_first ? fixture.target : fixture.source;
            deleteAxis(&fixture.app, first, route == 1 && !owner_first);
            SHARING_CHECK(state.destroyed_buffers == int(owner_first) &&
                          state.freed_memories == int(owner_first));
            SHARING_CHECK(!first->bufferLUT && !first->bufferLUTDeviceMemory &&
                          !first->bufferLUTSize && !first->referenceLUT);
            deleteAxis(&fixture.app, first, route == 1 && !owner_first);
            SHARING_CHECK(state.destroyed_buffers == int(owner_first) &&
                          state.freed_memories == int(owner_first));
            if (!owner_first) {
                SHARING_CHECK(fixture.source->bufferLUT == handle<VkBuffer>(5));
                SHARING_CHECK(fixture.source->bufferLUTSize == fixture.bytes);
            }
            deleteAxis(&fixture.app, last, route == 1 && owner_first);
            deleteAxis(&fixture.app, last, route == 1 && owner_first);
            SHARING_CHECK(state.destroyed_buffers == 1 && state.freed_memories == 1 &&
                          !state.invalid && !live());
            SHARING_CHECK(!last->bufferLUT && !last->bufferLUTDeviceMemory &&
                          !last->bufferLUTSize && !last->referenceLUT);
        }
    }
    return 0;
}

#undef SHARING_CHECK
#endif
