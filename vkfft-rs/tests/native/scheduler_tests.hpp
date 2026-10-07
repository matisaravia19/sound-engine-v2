// Exercise the actual scheduler without creating devices, shaders or GPU allocations.
#include "scheduler/stages.hpp"
#include "vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_Scheduler.h"

namespace codegen_fake {

template <size_t N> int scheduler_capacity_case() {
    struct Guarded {
        int radices[N]{};
        int radix_guard = 123456;
        int generators[N]{};
        int generator_guard = 123456;
    } stages;
    VkFFTRaderContainer rader{};
    rader.prime = 17;
    rader.generator = 3;
    rader.multiplier = 1;
    rader.type = 1;
    for (int count : {-1, static_cast<int>(N), static_cast<int>(N + 1)}) {
        int cursor = count;
        if (VkFFTGetRaderFFTStages(&rader, 1, &cursor, stages.radices, stages.generators) !=
                VKFFT_ERROR_MATH_FAILED ||
            cursor != count || rader.multiplier != 1 || stages.radix_guard != 123456 ||
            stages.generator_guard != 123456)
            return __LINE__;
    }
    int count = static_cast<int>(N) - 1;
    rader.multiplier = 2;
    if (VkFFTGetRaderFFTStages(&rader, 1, &count, stages.radices, stages.generators) !=
            VKFFT_ERROR_MATH_FAILED ||
        count != N || rader.multiplier != 1 || stages.radices[N - 1] != 17 ||
        stages.generators[N - 1] != 3 || stages.radix_guard != 123456 ||
        stages.generator_guard != 123456)
        return __LINE__;
    for (int i = 0; i < static_cast<int>(N) - 1; ++i)
        if (stages.radices[i] || stages.generators[i])
            return __LINE__;
    return 0;
}

int scheduler_nested_capacity_case() {
    VkFFTRaderContainer rader{};
    rader.prime = 17;
    rader.generator = 3;
    rader.multiplier = 1;
    rader.numStages = 19;
    rader.numSubPrimes = 0;
    rader.stage_rader_generator[0] = 123456;
    rader.loc_multipliers[2] = 2;
    int radices[33]{};
    int generators[33]{};
    int count = 0;
    if (VkFFTGetRaderFFTStages(&rader, 1, &count, radices, generators) != VKFFT_ERROR_MATH_FAILED ||
        rader.numStages != 20 || rader.loc_multipliers[2] != 1 || rader.stageRadix[19] != 2 ||
        rader.stage_rader_generator[19] || rader.stage_rader_generator[0] != 123456 || count != 1)
        return __LINE__;
    rader = {};
    rader.prime = 17;
    rader.generator = 3;
    rader.numSubPrimes = 1;
    rader.container = &rader;
    count = 0;
    if (VkFFTGetRaderFFTStages(&rader, 1, &count, radices, generators) != VKFFT_ERROR_MATH_FAILED)
        return __LINE__;
    rader.container = nullptr;
    return 0;
}

// Each node's stages have different register demands so an outer/inner index
// mix-up cannot accidentally produce the same maximum as a correct traversal.
struct NestedSchedulerFixture {
    int radices[33]{};
    int generators[33]{};
    int registers[33]{};
    VkFFTRaderContainer parent{};
    VkFFTRaderContainer child{};

    NestedSchedulerFixture() {
        radices[0] = 2;
        radices[1] = 23;
        generators[1] = 5;
        registers[2] = 2;
        parent.prime = 23;
        parent.generator = 5;
        parent.numStages = 2;
        parent.stageRadix[0] = 11;
        parent.stageRadix[1] = 2;
        parent.stage_rader_generator[0] = 2;
        parent.registers_per_thread_per_radix[2] = 4;
        parent.container = &child;
        parent.numSubPrimes = 1;
        parent.containerFFTDim = 22;
        parent.containerFFTNum = 1;
        child.prime = 11;
        child.generator = 2;
        child.numStages = 2;
        child.stageRadix[0] = 5;
        child.stageRadix[1] = 2;
        child.registers_per_thread_per_radix[5] = 5;
        child.registers_per_thread_per_radix[2] = 6;
        child.containerFFTDim = 10;
        child.containerFFTNum = 1;
    }
};

int scheduler_register_case(int mode) {
    NestedSchedulerFixture fixture;
    auto& f = fixture;
    int count = 2;
    int container_count = 1;
    switch (mode) {
        case 1:
            count = -1;
            break;
        case 2:
            count = 34;
            break;
        case 3:
            f.radices[0] = 33;
            break;
        case 4:
            f.registers[2] = -1;
            break;
        case 5:
            f.generators[1] = -1;
            break;
        case 6:
            f.parent.generator = 3;
            break;
        case 7:
            f.parent.type = 2;
            break;
        case 8:
            f.parent.numStages = 21;
            break;
        case 9:
            f.parent.numStages = 0;
            break;
        case 10:
            f.parent.numSubPrimes = 0;
            break;
        case 11:
            f.parent.container = nullptr;
            break;
        case 12:
            f.child.stageRadix[0] = -1;
            break;
        case 13:
            container_count = -1;
            break;
        case 14:
            container_count = 34;
            break;
        case 15:
            f.child.stageRadix[0] = 11;
            f.child.stage_rader_generator[0] = 2;
            f.child.container = &f.child;
            f.child.numSubPrimes = 1;
            break;
    }
    int minimum = 1000;
    int maximum = 0;
    auto result = VkFFTMinMaxRegisterCheck(count, f.radices, &minimum, &maximum, f.registers,
                                           &f.parent, container_count, f.generators);
    if (mode == 0)
        return result == VKFFT_SUCCESS && minimum == 2 && maximum == 6 ? 0 : __LINE__;
    return result == VKFFT_ERROR_MATH_FAILED && minimum == 1000 && maximum == 0 ? 0 : __LINE__;
}

int scheduler_threads_case(int mode) {
    NestedSchedulerFixture f;
    switch (mode) {
        case 1:
            f.parent.containerFFTDim = 0;
            break;
        case 2:
            f.parent.containerFFTNum = -1;
            break;
        case 3:
            f.parent.numStages = 21;
            break;
        case 4:
            f.parent.stageRadix[1] = 33;
            break;
        case 5:
            f.parent.registers_per_thread_per_radix[2] = 0;
            break;
        case 6:
            f.parent.registers_per_thread_per_radix[2] = -1;
            break;
        case 7:
            f.parent.containerFFTNum = INT_MAX;
            break;
        case 8:
            f.parent.numSubPrimes = 1;
            f.parent.container = nullptr;
            break;
        case 9:
            f.parent.stage_rader_generator[0] = -1;
            break;
        case 10:
            f.parent.type = 2;
            break;
        case 11:
            f.child.container = &f.parent;
            f.child.numSubPrimes = 1;
            break;
    }
    int threads = 0;
    const auto result = VkFFTGetRaderFFTThreadsNum(&f.parent, 1, &threads);
    if (mode == 0)
        return result == VKFFT_SUCCESS && threads == 6 ? 0 : __LINE__;
    return result == VKFFT_ERROR_MATH_FAILED && threads == 0 ? 0 : __LINE__;
}

int scheduler_reorder_case() {
    VkFFTSpecializationConstantsLayout sc{};
    sc.numStages = 2;
    sc.stageRadix[0] = 41;
    sc.rader_generator[0] = 6;
    sc.stageRadix[1] = 3;
    sc.registers_per_thread_per_radix[3] = 3;
    // This unrelated field lies beyond the radix table; it must never influence selection.
    sc.useCoalescedLUTUploadToSM = 3;
    if (vkfft_scheduler::prioritize_register_stage(sc, 3) != VKFFT_SUCCESS ||
        sc.stageRadix[0] != 3 || sc.rader_generator[0] || sc.stageRadix[1] != 41 ||
        sc.rader_generator[1] != 6)
        return __LINE__;
    sc.numStages = 34;
    if (vkfft_scheduler::prioritize_register_stage(sc, 3) != VKFFT_ERROR_MATH_FAILED)
        return __LINE__;
    sc.numStages = 1;
    sc.stageRadix[0] = 33;
    return vkfft_scheduler::prioritize_register_stage(sc, 3) == VKFFT_ERROR_MATH_FAILED ? 0
                                                                                        : __LINE__;
}

// Own every scheduler allocation through the same axis teardown used by native plans.
struct SchedulerFixture {
    VkFFTApplication app{};
    std::unique_ptr<VkFFTPlan> plan = std::make_unique<VkFFTPlan>();
    pfUINT temporary_bytes = 0;

    SchedulerFixture(int length, bool double_precision, bool strided, int boost) {
        auto& c = app.configuration;
        c.FFTdim = 1;
        c.size[0] = length;
        c.numberBatches = c.coordinateFeatures = c.numberKernels = 1;
        c.doublePrecision = double_precision;
        c.considerAllAxesStrided = strided;
        c.registerBoost = boost;
        c.registerBoost4Step = 1;
        c.coalescedMemory = 32;
        c.sharedMemorySize = 49152;
        c.sharedMemorySizePow2 = 32768;
        c.maxThreadsNum = 1024;
        c.warpSize = 32;
        c.fixMinRaderPrimeMult = c.fixMinRaderPrimeFFT = 17;
        c.fixMaxRaderPrimeMult = 89;
        c.fixMaxRaderPrimeFFT = 16384;
        c.vendorID = 0x10de;
        c.swapTo3Stage4Step = 4194305;
        c.reorderFourStep = 1;
        c.useRaderUintLUT = 1;
        c.tempBufferSize = &temporary_bytes;
    }
    ~SchedulerFixture() {
        for (auto& uploads : plan->axes)
            for (auto& axis : uploads)
                deleteAxis(&app, &axis, 0);
    }
};

int scheduler_plan_case(int length, bool precision, bool strided, int boost, int fail_at,
                        bool throws, int& attempts) {
    state = {};
    state.fail_at = throws ? 0 : fail_at;
    state.throw_at = throws ? fail_at : 0;
    try {
        SchedulerFixture fixture(length, precision, strided, boost);
        const auto result = VkFFTScheduler(&fixture.app, fixture.plan.get(), 0);
        if (result != (fail_at ? VKFFT_ERROR_MALLOC_FAILED : VKFFT_SUCCESS))
            return __LINE__;
        if (!fail_at) {
            const auto uploads = fixture.plan->numAxisUploads[0];
            if (!uploads || uploads > 4)
                return __LINE__;
            for (pfUINT upload = 0; upload < uploads; ++upload) {
                const auto& sc = fixture.plan->axes[0][upload].specializationConstants;
                if (sc.numStages < 0 || sc.numStages > 33 || sc.registers_per_thread <= 0)
                    return __LINE__;
                pfINT product = 1;
                for (int stage = 0; stage < sc.numStages; ++stage) {
                    if (sc.stageRadix[stage] < 2 ||
                        product > sc.fftDim.data.i / sc.stageRadix[stage])
                        return __LINE__;
                    product *= sc.stageRadix[stage];
                }
                if (product != sc.fftDim.data.i)
                    return __LINE__;
            }
        }
    } catch (const std::bad_alloc&) {
        if (!throws || !fail_at)
            return __LINE__;
    }
    attempts = state.attempts;
    return live() || state.invalid || (fail_at && attempts != fail_at) ? __LINE__ : 0;
}

} // namespace codegen_fake

extern "C" int vkfft_test_scheduler_metadata() {
    using namespace codegen_fake;
    if (int line = scheduler_capacity_case<33>())
        return line;
    if (int line = scheduler_capacity_case<20>())
        return line;
    if (int line = scheduler_nested_capacity_case())
        return line;
    for (int mode = 0; mode < 16; ++mode)
        if (int line = scheduler_register_case(mode))
            return line;
    for (int mode = 0; mode < 12; ++mode)
        if (int line = scheduler_threads_case(mode))
            return line;
    return scheduler_reorder_case();
}

extern "C" int vkfft_test_scheduler_plans() {
    using namespace codegen_fake;
    for (bool precision : {false, true})
        for (bool strided : {false, true})
            for (int boost : {1, 4})
                for (int length :
                     {1, 2, 16, 17, 23, 41, 257, 1009, 48, 51, 85, 123, 771, 1285, 4096, 65537}) {
                    int attempts = 0;
                    if (int line = scheduler_plan_case(length, precision, strided, boost, 0, false,
                                                       attempts))
                        return line;
                    for (int fail_at = 1; fail_at <= attempts; ++fail_at)
                        for (bool throws : {false, true}) {
                            int ignored = 0;
                            if (int line = scheduler_plan_case(length, precision, strided, boost,
                                                               fail_at, throws, ignored))
                                return line;
                        }
                }
    return 0;
}
