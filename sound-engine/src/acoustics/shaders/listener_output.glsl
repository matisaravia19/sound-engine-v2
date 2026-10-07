// -----------------------------------------------------------------------------
// Visibility, stereo panning, and IR deposition
// -----------------------------------------------------------------------------

// Atomically deposits one ear contribution with a linear fractional delay.
// Boundary clipping intentionally drops any portion outside the configured IR.
void depositChannel(float sample_position, float amplitude, uint channel) {
    if (!finiteFloat(sample_position)
        || !finiteFloat(amplitude)
        || sample_position <= -1.0
        || sample_position >= float(pc.sample_count)) {
        return;
    }

    int base = int(floor(sample_position));
    int next = base + 1;
    float fraction = sample_position - float(base);
    // Linear fractional delay preserves the total signed pressure of an event:
    // the two weights always sum to one unless the IR boundary clips the tail.
    if (base >= 0 && uint(base) < pc.sample_count) {
        if (channel == 0u) {
            atomicAdd(ir.samples[uint(base)].left, amplitude * (1.0 - fraction));
        } else {
            atomicAdd(ir.samples[uint(base)].right, amplitude * (1.0 - fraction));
        }
    }
    if (fraction > 0.0 && next >= 0 && uint(next) < pc.sample_count) {
        if (channel == 0u) {
            atomicAdd(ir.samples[uint(next)].left, amplitude * fraction);
        } else {
            atomicAdd(ir.samples[uint(next)].right, amplitude * fraction);
        }
    }
}

// The deterministic direct arrival dominates perceived level in a visible
// region, so its fractional delay must not inherit the two-tap interpolator's
// 3 dB phase-dependent energy ripple. Use its normalized Lanczos kernel,
// or a compact pressure-preserving cubic kernel for numerous edge arrivals.
void depositStableChannel(float sample_position, float amplitude, uint channel, bool cubic_edge) {
    if (!finiteFloat(sample_position)
        || !finiteFloat(amplitude)
        || sample_position <= -1.0
        || sample_position >= float(pc.sample_count)) {
        return;
    }

    // The four-tap edge kernel preserves signed pressure without per-ray
    // trigonometry. Geometric direct arrivals retain their existing Lanczos.
    if (cubic_edge) {
        int base = int(floor(sample_position));
        float fraction = sample_position - float(base);
        float square = fraction * fraction;
        float cube = square * fraction;
        vec4 weights = vec4(-0.5 * fraction + square - 0.5 * cube,
            1.0 - 2.5 * square + 1.5 * cube,
            0.5 * fraction + 2.0 * square - 1.5 * cube,
            -0.5 * square + 0.5 * cube);
        for (int tap = 0; tap < 4; ++tap) {
            int sample_index = base + tap - 1;
            if (sample_index < 0 || uint(sample_index) >= pc.sample_count) {
                continue;
            }
            float contribution = amplitude * weights[tap];
            if (channel == 0u) {
                atomicAdd(ir.samples[uint(sample_index)].left, contribution);
            } else {
                atomicAdd(ir.samples[uint(sample_index)].right, contribution);
            }
        }
        return;
    }

    const int LANCZOS_RADIUS = 8;
    int base = int(floor(sample_position));
    float weight_sum = 0.0;
    for (int offset = -LANCZOS_RADIUS + 1; offset <= LANCZOS_RADIUS; ++offset) {
        float distance = float(base + offset) - sample_position;
        float sinc_value = abs(distance) < 1.0e-6
            ? 1.0
            : sin(PI * distance) / (PI * distance);
        float window_value = abs(distance) < 1.0e-6
            ? 1.0
            : sin(PI * distance / float(LANCZOS_RADIUS))
                / (PI * distance / float(LANCZOS_RADIUS));
        weight_sum += sinc_value * window_value;
    }
    if (!finiteFloat(weight_sum) || abs(weight_sum) <= 1.0e-6) {
        return;
    }

    for (int offset = -LANCZOS_RADIUS + 1; offset <= LANCZOS_RADIUS; ++offset) {
        int sample_index = base + offset;
        if (sample_index < 0 || uint(sample_index) >= pc.sample_count) {
            continue;
        }
        float distance = float(sample_index) - sample_position;
        float sinc_value = abs(distance) < 1.0e-6
            ? 1.0
            : sin(PI * distance) / (PI * distance);
        float window_value = abs(distance) < 1.0e-6
            ? 1.0
            : sin(PI * distance / float(LANCZOS_RADIUS))
                / (PI * distance / float(LANCZOS_RADIUS));
        float contribution = amplitude * sinc_value * window_value / weight_sum;
        if (channel == 0u) {
            atomicAdd(ir.samples[uint(sample_index)].left, contribution);
        } else {
            atomicAdd(ir.samples[uint(sample_index)].right, contribution);
        }
    }
}

// Applies listener-space spatialization with pressure-preserving interpolation.
// Scalar edge arrivals select cubic interpolation; direct arrivals use Lanczos.
void depositStablePressure(
    float total_path_length,
    float amplitude,
    vec3 listener_arrival_direction,
    bool cubic_edge
) {
    if (!finiteFloat(total_path_length)
        || !finiteFloat(amplitude)
        || total_path_length < 0.0
        || total_path_length > pc.maximum_path_length) {
        return;
    }

    float center_sample_position =
        total_path_length / max(pc.speed_of_sound, RAY_EPSILON) * float(pc.sample_rate);
    if (!finiteFloat(center_sample_position) || center_sample_position < 0.0) {
        return;
    }

    if (pc.output_channels == OUTPUT_CHANNELS_MONO) {
        depositStableChannel(center_sample_position, amplitude, 0u, cubic_edge);
        return;
    }

    float lateral = clamp(
        dot(normalizeOrZero(listener_arrival_direction), normalizeOrZero(pc.listener_right)),
        -1.0,
        1.0
    );
    if (pc.output_channels == OUTPUT_CHANNELS_STEREO) {
        float left_factor = sqrt(0.5 * (1.0 - lateral));
        float right_factor = sqrt(0.5 * (1.0 + lateral));
        depositStableChannel(center_sample_position, amplitude * left_factor, 0u, cubic_edge);
        depositStableChannel(center_sample_position, amplitude * right_factor, 1u, cubic_edge);
        return;
    }

    float absolute_lateral = abs(lateral);
    float itd_seconds = pc.head_radius_meters
        / max(pc.speed_of_sound, RAY_EPSILON)
        * (asin(absolute_lateral) + absolute_lateral);
    itd_seconds *= sign(lateral);
    float half_itd_samples = 0.5 * itd_seconds * float(pc.sample_rate);
    float far_to_near = pow(10.0, -pc.max_ild_db * absolute_lateral / 20.0);
    float near_factor = inversesqrt(1.0 + far_to_near * far_to_near);
    float far_factor = far_to_near * near_factor;
    float left_factor = lateral < 0.0 ? near_factor : far_factor;
    float right_factor = lateral > 0.0 ? near_factor : far_factor;
    if (lateral == 0.0) {
        left_factor = near_factor;
        right_factor = near_factor;
    }

    depositStableChannel(center_sample_position + half_itd_samples, amplitude * left_factor, 0u, cubic_edge);
    depositStableChannel(center_sample_position - half_itd_samples, amplitude * right_factor, 1u, cubic_edge);
}

// The single geometric arrival retains its established 16-tap kernel.
// Edge arrivals use the compact cubic path through depositStablePressure.
void depositDirectPressure(
    float total_path_length,
    float amplitude,
    vec3 listener_arrival_direction
) {
    depositStablePressure(total_path_length, amplitude, listener_arrival_direction, false);
}

// Converts geometric length to a fractional sample position and applies the
// output mode's listener-space spatialization before IR deposition.
void depositPressure(
    float total_path_length,
    float amplitude,
    vec3 listener_arrival_direction
) {
    if (!finiteFloat(total_path_length)
        || !finiteFloat(amplitude)
        || total_path_length < 0.0
        || total_path_length > pc.maximum_path_length) {
        return;
    }

    float center_sample_position =
        total_path_length / max(pc.speed_of_sound, RAY_EPSILON) * float(pc.sample_rate);
    if (!finiteFloat(center_sample_position) || center_sample_position < 0.0) {
        return;
    }

    if (pc.output_channels == OUTPUT_CHANNELS_MONO) {
        depositChannel(center_sample_position, amplitude, 0u);
        return;
    }

    // The initial backward path direction points toward the apparent source.
    // Its right-axis projection is positive for arrivals on the listener's right.
    float lateral = clamp(
        dot(normalizeOrZero(listener_arrival_direction), normalizeOrZero(pc.listener_right)),
        -1.0,
        1.0
    );
    if (pc.output_channels == OUTPUT_CHANNELS_STEREO) {
        // Preserve the existing equal-power stereo panner exactly.
        float left_factor = sqrt(0.5 * (1.0 - lateral));
        float right_factor = sqrt(0.5 * (1.0 + lateral));
        depositChannel(center_sample_position, amplitude * left_factor, 0u);
        depositChannel(center_sample_position, amplitude * right_factor, 1u);
        return;
    }

    // Woodworth's spherical-head delay expressed by the lateral sine. Using
    // asin(lateral) makes front/back and elevation share the expected cone of
    // confusion while preserving the delay sign between the two ears.
    float absolute_lateral = abs(lateral);
    float itd_seconds = pc.head_radius_meters
        / max(pc.speed_of_sound, RAY_EPSILON)
        * (asin(absolute_lateral) + absolute_lateral);
    itd_seconds *= sign(lateral);
    float half_itd_samples = 0.5 * itd_seconds * float(pc.sample_rate);

    // Scalar ILD cannot reproduce frequency-dependent head shadow, but bounding
    // the far ear avoids the hard null produced by extreme stereo panning.
    float far_to_near = pow(10.0, -pc.max_ild_db * absolute_lateral / 20.0);
    float near_factor = inversesqrt(1.0 + far_to_near * far_to_near);
    float far_factor = far_to_near * near_factor;
    float left_factor = lateral < 0.0 ? near_factor : far_factor;
    float right_factor = lateral > 0.0 ? near_factor : far_factor;
    if (lateral == 0.0) {
        left_factor = near_factor;
        right_factor = near_factor;
    }

    depositChannel(center_sample_position + half_itd_samples, amplitude * left_factor, 0u);
    depositChannel(center_sample_position - half_itd_samples, amplitude * right_factor, 1u);
}

// Evaluates the deterministic zero-vertex path listener -> source.

void evaluateDirectPath() {
    vec3 to_source = pc.source_position - pc.listener_position;
    float distance_to_source = length(to_source);
    if (!finiteFloat(distance_to_source)) {
        return;
    }
    vec3 source_direction = distance_to_source > DIRECTION_EPSILON
        ? to_source / distance_to_source
        : vec3(0.0);
    if (!traceVisibility(
        pc.listener_position + source_direction * RAY_EPSILON,
        source_direction,
        distance_to_source
    )) {
        return;
    }
    // The direct path is evaluated exactly once by invocation zero and is not a
    // listener-sphere Monte Carlo sample, so it does not receive the 4*PI/N term.
    float spreading = 1.0 / max(distance_to_source, RAY_EPSILON);
    depositDirectPressure(distance_to_source, pc.source_amplitude * spreading, source_direction);
}
