// LUT reuse is confined to axes owned by the same native application.
#ifndef VKFFT_RS_LUT_SHARING_HPP
#define VKFFT_RS_LUT_SHARING_HPP

#include "lut/layout.hpp"

namespace vkfft_lut {
namespace {

// Numeric layout fields survive shader-parameter cleanup. Compare actual upload
// slots, including the nested Rader layout, before reading their buffer handles.
static inline bool same_complex_layout(const VkFFTConfiguration& config,
                                       const VkFFTSpecializationConstantsLayout& left,
                                       const VkFFTSpecializationConstantsLayout& right) {
    const int stage_capacity = sizeof(left.stageRadix) / sizeof(left.stageRadix[0]);
    if (left.res != VKFFT_SUCCESS || right.res != VKFFT_SUCCESS || !valid_axis(left) ||
        !valid_axis(right) || left.numStages < 0 || left.numStages > stage_capacity ||
        left.numStages != right.numStages || left.maxStageSumLUT != right.maxStageSumLUT ||
        left.fft_dim_full.data.i != right.fft_dim_full.data.i || left.useRader != right.useRader ||
        left.inline_rader_kernel != right.inline_rader_kernel ||
        left.numRaderPrimes != right.numRaderPrimes || !valid_rader_list(left) ||
        !valid_rader_list(right))
        return false;
    for (int i = 0; i < left.numStages; ++i) {
        if (left.stageRadix[i] != right.stageRadix[i] ||
            left.rader_generator[i] != right.rader_generator[i])
            return false;
    }
    const bool left_four_step = config.useLUT_4step == 1 && left.axis_upload_id > 0;
    const bool right_four_step = config.useLUT_4step == 1 && right.axis_upload_id > 0;
    if (left_four_step != right_four_step ||
        (left_four_step && (left.stageStartSize.data.i != right.stageStartSize.data.i ||
                            left.fftDim.data.i != right.fftDim.data.i)))
        return false;
    if (left.performDCT != right.performDCT || left.performDST != right.performDST)
        return false;
    if ((left.performDCT || left.performDST) &&
        (left.actualInverse != right.actualInverse || left.axis_upload_id != right.axis_upload_id ||
         left.numAxisUploads != right.numAxisUploads ||
         left.reverseBluesteinMultiUpload != right.reverseBluesteinMultiUpload ||
         config.size[left.axis_id] != config.size[right.axis_id]))
        return false;
    for (int i = 0; i < left.numRaderPrimes; ++i) {
        const auto& a = left.raderContainer[i];
        const auto& b = right.raderContainer[i];
        if (a.prime != b.prime || a.generator != b.generator || a.type != b.type)
            return false;
        if (a.type == 0) {
            const int rader_capacity = sizeof(a.stageRadix) / sizeof(a.stageRadix[0]);
            if (a.numStages < 1 || a.numStages > rader_capacity || a.numStages != b.numStages)
                return false;
            for (int stage = 0; stage < a.numStages; ++stage) {
                if (a.stageRadix[stage] != b.stageRadix[stage])
                    return false;
            }
            // Cached FFT kernels remain owned by this application. Pointer identity
            // avoids assuming independently generated kernels have identical data.
            if (left.useRader && !left.inline_rader_kernel && a.raderFFTkernel != b.raderFFTkernel)
                return false;
        }
    }
    return true;
}

// Both axes belong to the same application; its owner outlives all submissions.
// referenceLUT prevents the borrowing axis from freeing this allocation at teardown.
static inline bool borrow_table(VkFFTAxis& target, const VkFFTAxis* source) {
    if (!source || source == &target || source->specializationConstants.res != VKFFT_SUCCESS ||
        !source->bufferLUT || source->bufferLUTSize != target.bufferLUTSize)
        return false;
#if (VKFFT_BACKEND == 0)
    if (!source->bufferLUTDeviceMemory)
        return false;
    target.bufferLUTDeviceMemory = source->bufferLUTDeviceMemory;
#endif
    target.bufferLUT = source->bufferLUT;
    target.referenceLUT = 1;
    return true;
}

// Commit borrowed handles only after the complete layout comparison succeeds.
static inline bool borrow_complex_table(const VkFFTConfiguration& config, VkFFTAxis& target,
                                        const VkFFTAxis* source) {
    return source && source != &target &&
           same_complex_layout(config, target.specializationConstants,
                               source->specializationConstants) &&
           borrow_table(target, source);
}

// Resolve reuse before any host allocation or table generation. An unavailable
// or incompatible source falls back to an independently owned, correctly sized LUT.
static inline bool reuse_complex_table(VkFFTApplication& app, VkFFTPlan* plan, VkFFTAxis& axis,
                                       pfUINT inverse, bool real_table_extension) {
    axis.referenceLUT = 0;
    if (real_table_extension)
        return false;
    const auto& sc = axis.specializationConstants;
    if (sc.reverseBluesteinMultiUpload == 1 && plan &&
        borrow_complex_table(app.configuration, axis, &plan->axes[sc.axis_id][sc.axis_upload_id]))
        return true;
    if (!inverse && !app.configuration.makeForwardPlanOnly && app.localFFTPlan_inverse &&
        borrow_complex_table(app.configuration, axis,
                             &app.localFFTPlan_inverse->axes[sc.axis_id][sc.axis_upload_id]))
        return true;
    if (plan) {
        for (int earlier = 0; earlier < sc.axis_id; ++earlier) {
            if (plan->numAxisUploads[earlier] > static_cast<pfUINT>(sc.axis_upload_id) &&
                borrow_complex_table(app.configuration, axis,
                                     &plan->axes[earlier][sc.axis_upload_id]))
                return true;
        }
    }
    return false;
}

// R2C tables depend only on the application's immutable length and precision;
// their expected byte count is computed before checking this candidate.
static inline bool reuse_r2c_table(VkFFTApplication& app, VkFFTAxis& axis, pfUINT inverse) {
    axis.referenceLUT = 0;
    return !inverse && !app.configuration.makeForwardPlanOnly && app.localFFTPlan_inverse &&
           borrow_table(axis, &app.localFFTPlan_inverse->R2Cdecomposition);
}

} // namespace
} // namespace vkfft_lut
#endif
