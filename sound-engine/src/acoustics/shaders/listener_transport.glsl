// Transport helpers issue only depth-one traversal calls. The iterative path
// loop and branch policy are defined in listener_path.glsl and ray generation.
// Miss and closest-hit stages return compact data; they never continue a path
// or write the IR. These helpers evaluate local responses and sampling weights;
// the caller owns accumulated path state and termination.
//
// Direction convention:
// - `direction` points away from the listener along the backward-traced path.
// - `incoming_direction` is that same listener-to-vertex direction at a hit.
// - `source_direction` points from the current vertex toward the source.
// - `initial_direction` is retained for listener-space panning even after bends.
//
// Each edge carries its dimensionless incident capture measure K*a/cos(i).
// Subsequent spreading starts at that edge: an edge-source connection uses 1/b,
// while a surface connection uses the unfolded suffix length. Acoustic delay
// always uses the full path. Paths without diffraction keep their original
// 1/total_length spreading. Initial spherical normalization is applied once.
// Mixed surface/edge paths retain the finite-lobe surface approximation.

const float PI = 3.14159265358979323846;
const float TWO_PI = 6.28318530717958647692;
// Origins are moved off accepted geometry to avoid immediately re-hitting the
// same primitive. This is a geometric tolerance, not an acoustic distance.
const float RAY_EPSILON = 0.001;
const float DIRECTION_EPSILON = 0.00001;
const uint HIT_NONE = 0;
const uint HIT_SURFACE = 1;
const uint HIT_DIFFRACTION_EDGE = 2;
const uint OUTPUT_CHANNELS_MONO = 0;
const uint OUTPUT_CHANNELS_STEREO = 1;
const uint OUTPUT_CHANNELS_BINAURAL = 2;
// Visibility rays see physical surfaces only. Continuation rays additionally
// see virtual diffraction strips, whose instances use the disjoint 0x02 mask.
const uint SURFACE_GEOMETRY_MASK = 0x01;
const uint CONTINUATION_GEOMETRY_MASK = 0x03;

// -----------------------------------------------------------------------------
// Scene inputs and traversal payloads
// -----------------------------------------------------------------------------

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;

struct IrSample {
    float left;
    float right;
};

// Float storage-buffer atomics are a required device feature. Every completed
// path can therefore deposit pressure without an intermediate contribution
// list, fixed-point encoding, or a later hit-to-IR compute pass.
layout(std430, set = 0, binding = 1) buffer IrBuffer {
    IrSample samples[];
} ir;

struct SceneMaterial {
    float specular_pressure;
    float rough_pressure;
    float transmission_pressure;
    float roughness;
};

layout(std430, set = 0, binding = 3) readonly buffer SceneMaterialBuffer {
    SceneMaterial materials[];
} scene_materials;

struct DiffractionEdge {
    vec3 start;
    float diffraction_radius;
    vec3 end;
    float _pad_end;
    vec3 edge_dir;
    float edge_length;
    vec3 bisector_dir;
    float edge_angle_radians;
    vec3 positive_plane_normal;
    float _pad0;
    vec3 positive_shadow_dir;
    float _pad1;
    vec3 negative_shadow_dir;
    float solid_wedge_cos;
};

layout(std430, set = 0, binding = 4) readonly buffer DiffractionEdgeBuffer {
    DiffractionEdge edges[];
} diffraction_edges;

struct HitPayload {
    // HIT_NONE is also the continuation-miss result.
    uint hit_type;
    // Parametric distance along the current world-space ray.
    float t;
    uint instance_id;
    uint primitive_id;
    // Triangle barycentrics for surfaces; (distance_to_edge, edge_t) for edges.
    vec2 barycentrics;
    vec3 world_normal;
    // Material index for a surface or edge index for a diffraction proxy.
    uint data_id;
};

struct VisibilityPayload {
    // The visibility hit group only changes this flag; it returns no geometry.
    uint occluded;
};

// Separate payload locations prevent the compact visibility query from
// clobbering the continuation result while nested traceRayEXT calls are issued
// sequentially by the same ray-generation invocation.
layout(location = 0) rayPayloadEXT HitPayload hit_payload;
layout(location = 1) rayPayloadEXT VisibilityPayload visibility_payload;

// Explicit offsets mirror ListenerTracePushConstants on the host. vec3 values
// occupy twelve bytes here, so relying on implicit layout would be error-prone.
layout(push_constant) uniform PushConstants {
    layout(offset = 0) vec3 source_position;
    layout(offset = 12) float source_amplitude;
    layout(offset = 16) vec3 listener_position;
    layout(offset = 28) float speed_of_sound;
    layout(offset = 32) vec3 listener_right;
    layout(offset = 44) float maximum_path_length;
    layout(offset = 48) uint listener_ray_count;
    layout(offset = 52) uint max_bounces;
    layout(offset = 56) uint sample_rate;
    layout(offset = 60) uint sample_count;
    layout(offset = 64) uint output_channels;
    layout(offset = 68) uint sampling_seed;
    layout(offset = 72) float minimum_throughput;
    layout(offset = 76) float head_radius_meters;
    layout(offset = 80) float max_ild_db;
} pc;

// -----------------------------------------------------------------------------
// Numerical guards and deterministic sampling
// -----------------------------------------------------------------------------

// GLSL normalization of a zero or non-finite vector can poison every later
// calculation with NaNs. Returning zero lets callers terminate that path.
bool finiteFloat(float value) {
    return !isnan(value) && !isinf(value);
}

bool finiteVector(vec3 value) {
    return !any(isnan(value)) && !any(isinf(value));
}

vec3 normalizeOrZero(vec3 value) {
    float length_squared = dot(value, value);
    if (!finiteFloat(length_squared) || length_squared <= DIRECTION_EPSILON * DIRECTION_EPSILON) {
        return vec3(0.0);
    }
    return value * inversesqrt(length_squared);
}

uint hashState(uint value) {
    // This integer avalanche is used both to initialize independent per-ray
    // streams and to advance them. Reproducibility is more important here than
    // cryptographic quality.
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

float randomFloat(inout uint state) {
    state = hashState(state + 0x9e3779b9u);
    return float(state >> 8) * (1.0 / 16777216.0);
}

// The Fibonacci lattice covers the sphere with nearly equal-area directions.
// A seed-dependent global azimuth rotation keeps the distribution stable for a
// fixed seed while allowing callers to choose a different deterministic set.
vec3 fibonacciSphereDirection(uint ray_index, uint ray_count, uint azimuth_rotation) {
    float index = float(ray_index);
    float count = float(max(ray_count, 1u));
    float z = 1.0 - 2.0 * ((index + 0.5) / count);
    float radius = sqrt(max(0.0, 1.0 - z * z));
    // Computing index * golden_angle directly in float loses nearly all
    // azimuth precision once the launch reaches a few million rays: the
    // unreduced product is millions of radians, so adjacent indices quantize
    // to the same handful of angles. Multiplication by 0x61c88647 performs the
    // same golden-ratio rotation modulo 2^32 using exact integer wraparound.
    // Taking the high 24 bits is exactly representable by float and retains a
    // stable, uniformly rotated lattice throughout the supported launch range.
    uint phase = ray_index * 0x61c88647u + azimuth_rotation;
    float phi = TWO_PI * float(phase >> 8) * (1.0 / 16777216.0);
    return vec3(cos(phi) * radius, sin(phi) * radius, z);
}

void orthonormalBasis(vec3 axis, out vec3 tangent, out vec3 bitangent) {
    // Choose a helper that is not almost parallel to the lobe axis; otherwise
    // the cross product loses precision for directions near the world Z axis.
    vec3 helper = abs(axis.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
    tangent = normalize(cross(helper, axis));
    bitangent = cross(axis, tangent);
}

float powerCosinePdf(float cosine, float exponent) {
    if (!finiteFloat(cosine) || !finiteFloat(exponent) || cosine <= 0.0 || exponent < 0.0) {
        return 0.0;
    }
    // This constant normalizes cos(theta)^exponent over one hemisphere.
    return (exponent + 1.0) * pow(cosine, exponent) / (2.0 * PI);
}

// Samples the same normalized finite lobe used by source connections. Because
// response contains this PDF, response / direction_pdf cancels for a sampled
// branch; retaining both terms makes the Monte Carlo estimator and guards clear.
bool samplePowerCosine(
    vec3 axis,
    float exponent,
    inout uint random_state,
    out vec3 sampled_direction,
    out float direction_pdf
) {
    vec3 normalized_axis = normalizeOrZero(axis);
    if (dot(normalized_axis, normalized_axis) <= 0.0 || exponent < 0.0) {
        return false;
    }

    float u0 = randomFloat(random_state);
    float u1 = randomFloat(random_state);
    // Invert the marginal CDF of the power-cosine distribution. Clamping u0
    // avoids an exact grazing sample whose PDF may underflow for sharp lobes.
    float cosine = pow(max(u0, DIRECTION_EPSILON), 1.0 / (exponent + 1.0));
    float sine = sqrt(max(0.0, 1.0 - cosine * cosine));
    float phi = TWO_PI * u1;
    vec3 tangent;
    vec3 bitangent;
    orthonormalBasis(normalized_axis, tangent, bitangent);
    sampled_direction = normalizeOrZero(
        tangent * (cos(phi) * sine)
        + bitangent * (sin(phi) * sine)
        + normalized_axis * cosine
    );
    direction_pdf = powerCosinePdf(max(dot(sampled_direction, normalized_axis), 0.0), exponent);
    return finiteVector(sampled_direction) && direction_pdf > 0.0 && finiteFloat(direction_pdf);
}

float materialSpecularExponent(float roughness) {
    // Match the Phong lobe's width to the authored RMS slope approximation.
    // Capping the smooth limit keeps it finite without making ordinary
    // nonzero roughness so narrow that a fixed ray lattice spatially aliases it.
    float alpha = max(clamp(roughness, 0.0, 1.0), 0.0001);
    return clamp(2.0 / (alpha * alpha) - 2.0, 1.0, 4096.0);
}

float specularExponent(float roughness) {
    // Point-source evaluation needs a finite numerical footprint. At the thesis
    // 65,536-ray budget, 4096 aliases even a single plane (see r1-d2-validation.md).
    // This cap regularizes source connections, not smooth material continuation.
    return min(materialSpecularExponent(roughness), 1024.0);
}

float roughExponent(float roughness) {
    // The rough branch stays much wider than the specular branch at every
    // roughness, reaching a cosine-weighted shape at roughness one.
    return mix(64.0, 1.0, clamp(roughness, 0.0, 1.0));
}

// -----------------------------------------------------------------------------
// Visibility, stereo panning, and IR deposition
// -----------------------------------------------------------------------------

// Returns true when the open segment from `origin` to the source is clear.
// The caller is responsible for offsetting origin onto the intended side of the
// current surface or edge.
bool traceVisibility(vec3 origin, vec3 direction, float distance_to_source) {
    if (distance_to_source <= 2.0 * RAY_EPSILON) {
        return true;
    }
    visibility_payload.occluded = 0u;
    // SBT offset 1 selects the surface-visibility hit record. Stride zero is
    // intentional: diffraction instances are excluded by the surface-only mask.
    // Miss index 1 selects visibility.rmiss.glsl.
    // tmin skips the offset origin's immediate neighborhood. Reducing tmax by
    // the same tolerance prevents geometry at the source endpoint from
    // incorrectly classifying the connection as blocked.
    traceRayEXT(
        tlas,
        gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT,
        SURFACE_GEOMETRY_MASK,
        1,
        0,
        1,
        origin,
        RAY_EPSILON,
        direction,
        max(distance_to_source - RAY_EPSILON, RAY_EPSILON),
        1
    );
    return visibility_payload.occluded == 0u;
}

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

// Folds the part of an axis-centered lobe that crosses a physical surface back
// into the valid outgoing hemisphere. The sum of the original and mirrored
// densities integrates to one there, so grazing lobes do not randomly lose
// energy through rejected continuation samples.
float foldedPowerCosinePdf(vec3 direction, vec3 axis, vec3 normal, float exponent) {
    vec3 mirrored_direction = reflect(direction, normal);
    return powerCosinePdf(max(dot(direction, axis), 0.0), exponent)
        + powerCosinePdf(max(dot(mirrored_direction, axis), 0.0), exponent);
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

// Keep the established interpolation for the single geometric direct arrival.
void depositDirectChannel(float sample_position, float amplitude, uint channel) {
    depositStableChannel(sample_position, amplitude, channel, false);
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

// -----------------------------------------------------------------------------
// Surface next-event estimation and continuation
// -----------------------------------------------------------------------------

// Evaluates all surface branches that can connect this vertex to the source.
// This is evaluation, not sampling, so valid specular and rough reflection
// responses are summed. Transmission is evaluated on the opposite hemisphere.
float surfaceSourceResponse(SceneMaterial material, vec3 incoming_direction, vec3 normal, vec3 source_direction, float source_fraction) {

    // These are pressure-domain local responses. The normalized lobe value
    // includes directional density, with pressure gains converted before scene upload.
    float response = 0.0;
    // Because incoming_direction points toward the surface, reflect() produces
    // the backward path's ideal outgoing reflection direction.
    vec3 reflection_axis = normalizeOrZero(reflect(incoming_direction, normal));
    // A short final connection magnifies angular mismatch by 1/source_fraction.
    // Bound the numerical footprint in listener angle, while retaining limits
    // on local blur. Authored nonzero roughness keeps its existing response.
    float connection_exponent = material.roughness <= 0.0
        ? clamp(16384.0 * source_fraction * source_fraction, 64.0, 1024.0)
        : specularExponent(material.roughness);
    if (dot(source_direction, normal) > DIRECTION_EPSILON) {
        response += material.specular_pressure
            * foldedPowerCosinePdf(
                source_direction,
                reflection_axis,
                normal,
                connection_exponent
            );
        response += material.rough_pressure
            * foldedPowerCosinePdf(
                source_direction,
                reflection_axis,
                normal,
                roughExponent(material.roughness)
            );
    }
    if (dot(source_direction, normal) < -DIRECTION_EPSILON) {
        // Straight transmission remains centered on incoming_direction: the
        // backward path continues through the surface without refraction.
        response += material.transmission_pressure
            * foldedPowerCosinePdf(
                source_direction,
                normalizeOrZero(incoming_direction),
                normal,
                connection_exponent
            );
    }
    return finiteFloat(response) ? response : 0.0;
}

void connectSurfaceToSource(
    vec3 hit_position,
    vec3 incoming_direction,
    vec3 normal,
    SceneMaterial material,
    float traveled_distance,
    float spreading_origin_distance,
    float path_throughput,
    float monte_carlo_normalization,
    vec3 listener_arrival_direction
) {
    // Next-event estimation explicitly connects every accepted surface vertex
    // to the point source. It preserves this path's accumulated listener-side
    // length instead of merging vertices into a common delay.
    vec3 to_source = pc.source_position - hit_position;
    float source_distance = length(to_source);
    if (!finiteFloat(source_distance) || source_distance <= DIRECTION_EPSILON) {
        return;
    }
    float total_path_length = traveled_distance + source_distance;
    if (total_path_length > pc.maximum_path_length) {
        return;
    }

    vec3 source_direction = to_source / source_distance;
    // Reflection footprint and spreading use the same post-edge origin.
    // With no previous edge this is the complete listener-to-source length.
    float connection_path_length = total_path_length - spreading_origin_distance;
    float local_response = surfaceSourceResponse(material, incoming_direction, normal, source_direction,
                                                 source_distance / max(connection_path_length, RAY_EPSILON));
    if (local_response <= 0.0) {
        return;
    }
    // Offset toward the same hemisphere as the source connection. Always using
    // +normal would make transmission rays start on the wrong side.
    vec3 offset_normal = dot(source_direction, normal) >= 0.0 ? normal : -normal;
    if (!traceVisibility(hit_position + offset_normal * RAY_EPSILON, source_direction, source_distance)) {
        return;
    }

    // The previous edge already carries the incident capture measure.
    // Apply suffix spreading once; the acoustic delay still uses the full path.
    float spreading = 1.0 / max(total_path_length - spreading_origin_distance, RAY_EPSILON);
    // A normalized reflection lobe is a density in source-connection solid
    // angle, while this estimator integrates uniformly sampled listener solid
    // angle. For a planar first reflection, perturbing the listener direction
    // changes the source-connection mismatch by total_length/source_distance
    // in each of its two angular dimensions. Unfolding a sequence of planar
    // mirror continuations gives the same narrow-lobe determinant. Smooth rays
    // therefore must not acquire the numerical lobe's spread at earlier bounces.
    // Finite lobes, rough paths and edge/surface mixtures remain approximations;
    // this is not an accumulated differential for those paths.
    float angular_scale = connection_path_length / max(source_distance, RAY_EPSILON);
    float source_connection_jacobian = angular_scale * angular_scale;
    float amplitude = pc.source_amplitude
        * path_throughput
        * local_response
        * source_connection_jacobian
        * spreading
        * monte_carlo_normalization;
    depositPressure(total_path_length, amplitude, listener_arrival_direction);
}

// Samples exactly one of:
//   branch 0: specular reflection
//   branch 1: rough reflection
//   branch 2: straight transmission
// On success, path_throughput is updated with the unbiased discrete/continuous
// importance weight and continuation_direction lies on the required hemisphere.
bool sampleSurfaceContinuation(
    SceneMaterial material,
    vec3 incoming_direction,
    vec3 normal,
    inout uint random_state,
    inout float path_throughput,
    out vec3 continuation_direction,
    out bool mirror_delta
) {
    mirror_delta = false;
    // Branch selection operates in the pressure domain. Choosing one branch,
    // rather than emitting all three, keeps one continuation ray per invocation.
    float weights[3] = float[3](material.specular_pressure, material.rough_pressure, material.transmission_pressure);
    float weight_sum = weights[0] + weights[1] + weights[2];
    if (!finiteFloat(weight_sum) || weight_sum <= 0.0) {
        return false;
    }

    float branch_sample = randomFloat(random_state) * weight_sum;
    uint branch = branch_sample < weights[0] ? 0u : (branch_sample < weights[0] + weights[1] ? 1u : 2u);
    float branch_probability = weights[branch] / weight_sum;
    if (branch_probability <= 0.0 || !finiteFloat(branch_probability)) {
        return false;
    }

    // Both reflection branches share the ideal reflection axis but use
    // different widths. Transmission is centered on straight propagation.
    vec3 axis = branch == 2u
        ? normalizeOrZero(incoming_direction)
        : normalizeOrZero(reflect(incoming_direction, normal));
    // A smooth material has a delta continuation. The finite source-connection
    // lobe regularizes point-source sampling; it must not scatter the ray again
    // at every smooth bounce. Only the discrete branch probability remains.
    if (branch != 1u && material.roughness <= 0.0) {
        mirror_delta = branch == 0u;
        continuation_direction = axis;
        path_throughput *= weights[branch] / branch_probability;
        return finiteFloat(path_throughput) && finiteVector(continuation_direction);
    }
    float exponent = branch == 1u ? roughExponent(material.roughness) : materialSpecularExponent(material.roughness);
    float direction_pdf;
    if (!samplePowerCosine(axis, exponent, random_state, continuation_direction, direction_pdf)) {
        return false;
    }
    // Fold wrong-side samples into the physical hemisphere instead of rejecting
    // them. Rejection without renormalizing the PDF was a position-dependent
    // energy loss, especially for rough or grazing lobes.
    bool wrong_side = branch == 2u
        ? dot(continuation_direction, normal) >= 0.0
        : dot(continuation_direction, normal) <= 0.0;
    if (wrong_side) {
        continuation_direction = normalizeOrZero(reflect(continuation_direction, normal));
    }

    // Divide by both discrete branch probability and continuous direction PDF.
    // For the matched power-cosine sampler the PDF cancels analytically, but
    // writing the full estimator avoids hiding that invariant.
    direction_pdf = foldedPowerCosinePdf(continuation_direction, axis, normal, exponent);
    if (direction_pdf <= 0.0 || !finiteFloat(direction_pdf)) {
        return false;
    }
    float response = weights[branch] * direction_pdf;
    path_throughput *= response / (branch_probability * direction_pdf);
    return finiteFloat(path_throughput) && finiteVector(continuation_direction);
}

// -----------------------------------------------------------------------------
// Diffraction geometry and next-event estimation
// -----------------------------------------------------------------------------

vec3 rotateAroundAxis(vec3 value, vec3 axis, float angle) {
    // Rotate the radial frame around the edge. The continuation sampler
    // selects the axial component separately.
    float sine = sin(angle);
    float cosine = cos(angle);
    return value * cosine + cross(axis, value) * sine + axis * dot(axis, value) * (1.0 - cosine);
}

bool insideSolidWedge(DiffractionEdge edge, vec3 perpendicular_direction) {
    return dot(perpendicular_direction, -edge.bisector_dir) > edge.solid_wedge_cos + DIRECTION_EPSILON;
}

// Shadow-only support; this is not a full boundary-continuity treatment.
// Angles are measured from the authored positive face through the exterior air
// wedge. The incident-side radial vector is -incoming_direction. The geometric
// shadow consists of outgoing angles separated from it by more than PI.
// Include the entire shadow interval: a forward-hemisphere cutoff excludes
// valid deep-shadow connections at wedges with a large exterior angle.
bool diffractionShadowDomain(
    DiffractionEdge edge,
    vec3 incoming_direction,
    out vec3 incoming_perpendicular,
    out float perpendicular_length,
    out vec3 parallel_component,
    out float valid_angle_measure,
    out float angle_start,
    out float angle_end
) {
    parallel_component = dot(incoming_direction, edge.edge_dir) * edge.edge_dir;
    vec3 perpendicular = incoming_direction - parallel_component;
    perpendicular_length = length(perpendicular);
    if (perpendicular_length <= DIRECTION_EPSILON) {
        return false;
    }
    incoming_perpendicular = perpendicular / perpendicular_length;
    vec3 incident_side = -incoming_perpendicular;
    float incident_angle = mod(atan(
        dot(edge.edge_dir, cross(edge.positive_shadow_dir, incident_side)),
        dot(edge.positive_shadow_dir, incident_side)
    ) + TWO_PI, TWO_PI);
    if (incident_angle > edge.edge_angle_radians + DIRECTION_EPSILON) {
        return false;
    }
    if (incident_angle < edge.edge_angle_radians - PI) {
        angle_start = 0.0;
        angle_end = edge.edge_angle_radians - PI - incident_angle;
    } else if (incident_angle > PI) {
        angle_start = PI - incident_angle;
        angle_end = 0.0;
    } else {
        // Both wedge faces are visible from this incidence.
        return false;
    }
    valid_angle_measure = angle_end - angle_start;
    return valid_angle_measure > DIRECTION_EPSILON;
}

// Broadband scalar approximation, not an exact BTM or frequency-domain UTD
// coefficient. The four rational terms include both wedge-face distances;
// axial mismatch narrows the response near the incident shadow boundary.
// A 0.02-radian floor bounds variance at that boundary. See the derivation and
// limitations in references/scalar-diffraction-validation.md.
const float DIFFRACTION_ANGULAR_FLOOR = 0.02;

float diffractionScalarResponse(
    DiffractionEdge edge,
    vec3 incoming_direction,
    vec3 outgoing_direction,
    bool shadow_only
) {
    float incoming_axial = dot(incoming_direction, edge.edge_dir);
    float outgoing_axial = dot(outgoing_direction, edge.edge_dir);
    vec3 incoming_radial = incoming_direction - incoming_axial * edge.edge_dir;
    vec3 outgoing_radial = outgoing_direction - outgoing_axial * edge.edge_dir;
    float incoming_radius = length(incoming_radial);
    float outgoing_radius = length(outgoing_radial);
    if (min(incoming_radius, outgoing_radius) <= DIRECTION_EPSILON) {
        return 0.0;
    }
    vec3 incident_side = -incoming_radial / incoming_radius;
    vec3 outgoing_side = outgoing_radial / outgoing_radius;
    float incident_angle = mod(atan(
        dot(edge.edge_dir, cross(edge.positive_shadow_dir, incident_side)),
        dot(edge.positive_shadow_dir, incident_side)
    ) + TWO_PI, TWO_PI);
    float outgoing_angle = mod(atan(
        dot(edge.edge_dir, cross(edge.positive_shadow_dir, outgoing_side)),
        dot(edge.positive_shadow_dir, outgoing_side)
    ) + TWO_PI, TWO_PI);
    float near_angle = min(incident_angle, outgoing_angle);
    float far_angle = max(incident_angle, outgoing_angle);
    float bend = far_angle - near_angle - PI;
    float exterior = edge.edge_angle_radians;
    if (far_angle > exterior + DIRECTION_EPSILON || (shadow_only && bend <= 0.0)) {
        return 0.0;
    }
    float face_a = near_angle;
    float face_b = max(exterior - far_angle, 0.0);
    vec4 angle = vec4(bend, bend + 2.0 * face_a, bend + 2.0 * face_b,
                      2.0 * (exterior - PI) - bend);
    float axial_mismatch = incoming_axial / incoming_radius
        - outgoing_axial / outgoing_radius;
    vec4 angle_squared = angle * angle;
    vec4 taper = max(vec4(0.0), vec4(1.0) - angle_squared / (exterior * exterior));
    // Only the shadow-boundary term needs its numerator regularized.
    // Keep the other three face terms unchanged away from that boundary.
    vec4 regularized_angle = angle;
    regularized_angle.x = sqrt(angle_squared.x
        + DIFFRACTION_ANGULAR_FLOOR * DIFFRACTION_ANGULAR_FLOOR);
    // Extend the same four-term approximation across the shadow boundary.
    // The leading term changes sign; no separately windowed lit correction is
    // added. This local approximation does not guarantee field continuity.
    if (bend <= 0.0) {
        regularized_angle.x = -regularized_angle.x;
    }
    vec4 terms = regularized_angle * taper / (angle_squared + vec4(
        axial_mismatch * axial_mismatch
        + DIFFRACTION_ANGULAR_FLOOR * DIFFRACTION_ANGULAR_FLOOR));
    return dot(terms, vec4(1.0)) / TWO_PI;
}

// Maps a normalized proxy strip to edge-line measure. K has units 1/m,
// so K*a/cos(i) is dimensionless. With the projected solid-angle density,
// this leaves incident 1/a spreading per unit edge length. Both estimators
// must use this same factor; inserting K alone would break scene-scale laws.
float diffractionCaptureWeight(
    DiffractionEdge edge,
    vec3 incoming_direction,
    float distance_to_edge,
    float incoming_distance
) {
    float projected_incidence = abs(dot(incoming_direction, edge.positive_plane_normal));
    if (projected_incidence <= DIRECTION_EPSILON) {
        return 0.0;
    }

    // Only the exterior half-strip is accessible, hence normalization on [0,R].
    // Keep projection distances comparable to the incident path length.
    // The triangular kernel still integrates to one for every edge point;
    // this changes radial sampling support, not the edge response or gain.
    float capture_radius = min(edge.diffraction_radius, incoming_distance);
    if (capture_radius <= RAY_EPSILON) {
        return 0.0;
    }
    float radial_kernel = 2.0
        * clamp(1.0 - distance_to_edge / capture_radius, 0.0, 1.0)
        / capture_radius;
    return radial_kernel * incoming_distance / projected_incidence;
}

// Source connections retain the bilateral scalar response. Continued edge
// interactions retain shadow-only response, evaluated at their physical endpoint.
bool diffractionSourceResponse(
    DiffractionEdge edge,
    vec3 incoming_direction,
    vec3 source_direction,
    float distance_to_edge,
    float incoming_distance,
    out float response
) {
    vec3 incoming_perpendicular;
    float perpendicular_length;
    vec3 parallel_component;
    float valid_angle_measure;
    float angle_start;
    float angle_end;
    if (!diffractionShadowDomain(
        edge,
        incoming_direction,
        incoming_perpendicular,
        perpendicular_length,
        parallel_component,
        valid_angle_measure,
        angle_start,
        angle_end
    )) {
        return false;
    }

    // Source connections must stay outside the solid wedge.
    vec3 source_perpendicular = source_direction
        - dot(source_direction, edge.edge_dir) * edge.edge_dir;
    source_perpendicular = normalizeOrZero(source_perpendicular);
    if (dot(source_perpendicular, source_perpendicular) <= 0.0
        || insideSolidWedge(edge, source_perpendicular)) {
        return false;
    }

    float capture_weight = diffractionCaptureWeight(
        edge, incoming_direction, distance_to_edge, incoming_distance
    );
    float directional_shape = diffractionScalarResponse(
        edge, incoming_direction, source_direction, false
    );
    response = capture_weight * directional_shape;
    return response != 0.0 && finiteFloat(response);
}

// Performs next-event estimation for listener -> ... -> edge -> source.
void connectDiffractionToSource(
    DiffractionEdge edge,
    vec3 edge_hit_position,
    vec3 incoming_direction,
    float distance_to_edge,
    float edge_path_distance,
    float spreading_origin_distance,
    float path_throughput,
    float monte_carlo_normalization,
    vec3 listener_arrival_direction
) {
    // Interaction preparation already reached the physical edge. The source
    // connection evaluates its local response using that same path state.
    vec3 to_source = pc.source_position - edge_hit_position;
    float source_distance = length(to_source);
    if (!finiteFloat(source_distance) || source_distance <= DIRECTION_EPSILON) {
        return;
    }
    float total_path_length = edge_path_distance + source_distance;
    if (total_path_length > pc.maximum_path_length) {
        return;
    }

    vec3 source_direction = to_source / source_distance;
    float local_response;
    if (!diffractionSourceResponse(
        edge,
        incoming_direction,
        source_direction,
        distance_to_edge,
        edge_path_distance - spreading_origin_distance,
        local_response
    )) {
        return;
    }
    // Visibility skips the immediate endpoint along the connection direction;
    // the proxy itself is invisible to this surface-only trace.
    if (!traceVisibility(edge_hit_position, source_direction, source_distance)) {
        return;
    }
    // The local capture measure contains incident spreading. Only the final
    // edge-to-source leg remains; using total length here would count it twice.
    float spreading = 1.0 / source_distance;
    float amplitude = pc.source_amplitude
        * path_throughput
        * local_response
        * spreading
        * monte_carlo_normalization;
    // Cubic interpolation reduces phase-dependent cancellation error using
    // four atomics per channel and no per-connection trigonometric kernel.
    depositStablePressure(total_path_length, amplitude, listener_arrival_direction, true);
}

// Samples a solid-angle proposal for one continued edge event. Finite axial
// spread matches the scalar response's dimensionality; a delta Keller cone
// cannot represent the edge-line integral after another proxy relocation.
// Full proposal support lets the next physical endpoint determine validity.
bool sampleDiffractionBaseProposal(
    DiffractionEdge edge,
    vec3 incoming_direction,
    inout uint random_state,
    out vec3 continuation_direction,
    out float pdf
) {
    vec3 incoming_perpendicular;
    float perpendicular_length;
    vec3 parallel_component;
    float valid_angle_measure;
    float angle_start;
    float angle_end;
    if (!diffractionShadowDomain(
        edge,
        incoming_direction,
        incoming_perpendicular,
        perpendicular_length,
        parallel_component,
        valid_angle_measure,
        angle_start,
        angle_end
    )) {
        return false;
    }

    // Integrating the regularized axial response removes its 1/bend peak.
    // Sample azimuth uniformly in shadow, with a full-circle component to
    // cover directions whose next proxy hit will relocate into that shadow.
    // This mixture only controls variance; its complete PDF is divided out.
    const float shadow_probability = 0.9;
    float azimuth_sample = randomFloat(random_state);
    float angle;
    if (azimuth_sample < shadow_probability) {
        angle = mix(angle_start, angle_end, azimuth_sample / shadow_probability);
    } else {
        angle = TWO_PI * (azimuth_sample - shadow_probability) / (1.0 - shadow_probability) - PI;
    }

    pdf = (1.0 - shadow_probability) / TWO_PI;
    if (angle >= angle_start && angle <= angle_end) {
        pdf += shadow_probability / valid_angle_measure;
    }
    float bend = abs(angle);
    vec3 sampled_perpendicular = normalizeOrZero(
        rotateAroundAxis(incoming_perpendicular, edge.edge_dir, angle)
    );
    // The evaluated edge response has finite axial spread. Sample that second
    // dimension too: a Cauchy proposal matches its regularized denominator.
    // With axial slope t, solid angle is d(phi)*dt/(1+t*t)^(3/2).
    float axial_scale = sqrt(bend * bend
        + DIFFRACTION_ANGULAR_FLOOR * DIFFRACTION_ANGULAR_FLOOR);
    float incoming_slope = dot(incoming_direction, edge.edge_dir) / perpendicular_length;
    // Relocation can turn a narrow sampled bend into a broad physical one.
    // Mix its local Cauchy proposal with a unit-width Cauchy so those paths
    // do not receive huge weights from an almost-zero proposal density.
    float axial_sample = randomFloat(random_state);
    float proposal_scale = axial_sample < 0.5 ? axial_scale : 1.0;
    float axial_angle = PI * (fract(2.0 * axial_sample) - 0.5);
    float slope_offset = proposal_scale * tan(axial_angle);
    float axial_slope = incoming_slope + slope_offset;
    float outgoing_radius = inversesqrt(1.0 + axial_slope * axial_slope);
    float slope_pdf = 0.5 / PI * (
        axial_scale / (axial_scale * axial_scale + slope_offset * slope_offset)
        + 1.0 / (1.0 + slope_offset * slope_offset)
    );
    pdf *= slope_pdf / (outgoing_radius * outgoing_radius * outgoing_radius);

    continuation_direction = normalizeOrZero(
        outgoing_radius * (sampled_perpendicular + axial_slope * edge.edge_dir)
    );
    if (!finiteFloat(pdf) || pdf <= 0.0) {
        return false;
    }
    if (!finiteVector(continuation_direction)) {
        return false;
    }

    return true;
}

// Evaluate the original full-support solid-angle proposal at any direction.
float diffractionProposalPdf(DiffractionEdge edge, vec3 incoming, vec3 outgoing) {
    vec3 incoming_radial;
    vec3 parallel_component;
    float incoming_radius;
    float measure;
    float start;
    float end;
    if (!diffractionShadowDomain(edge, incoming, incoming_radial, incoming_radius,
                                 parallel_component, measure, start, end)) {
        return 0.0;
    }
    float outgoing_axial = dot(outgoing, edge.edge_dir);
    vec3 outgoing_radial = outgoing - outgoing_axial * edge.edge_dir;
    float outgoing_radius = length(outgoing_radial);
    // The base proposal has support arbitrarily close to the edge axis.
    // A geometric angular cutoff here would omit part of the mixture PDF.
    if (outgoing_radius <= 0.0) {
        return 0.0;
    }
    outgoing_radial /= outgoing_radius;
    float angle = atan(dot(edge.edge_dir, cross(incoming_radial, outgoing_radial)),
                       dot(incoming_radial, outgoing_radial));
    float azimuth_pdf = 0.1 / TWO_PI;
    if (angle >= start && angle <= end) {
        azimuth_pdf += 0.9 / measure;
    }
    float width = sqrt(angle * angle + DIFFRACTION_ANGULAR_FLOOR * DIFFRACTION_ANGULAR_FLOOR);
    float offset = outgoing_axial / outgoing_radius - dot(incoming, edge.edge_dir) / incoming_radius;
    float slope_pdf = 0.5 / PI * (width / (width * width + offset * offset)
                                + 1.0 / (1.0 + offset * offset));
    return azimuth_pdf * slope_pdf / (outgoing_radius * outgoing_radius * outgoing_radius);
}

// Select a proposal, not an extra acoustic path. Both lobes contribute to the
// PDF even when only one of them generated the direction.
bool sampleDiffractionContinuation(
    DiffractionEdge edge,
    vec3 edge_position,
    vec3 incoming_direction,
    float distance_to_edge,
    float incident_distance,
    inout uint random_state,
    inout float path_throughput,
    out vec3 continuation_direction
) {
    vec3 incoming_radial;
    vec3 parallel_component;
    float radius;
    float measure;
    float start;
    float end;
    if (!diffractionShadowDomain(edge, incoming_direction, incoming_radial, radius,
                                 parallel_component, measure, start, end)) {
        return false;
    }
    // These proposal constants control variance, not acoustic gain. Retain
    // full base support, and fall back to it when the source has no direction.
    const float guide_exponent = 32.0;
    vec3 guide_axis = normalizeOrZero(pc.source_position - edge_position);
    float guide_probability = dot(guide_axis, guide_axis) > 0.0 ? 0.8 : 0.0;
    float base_pdf;
    float guide_pdf;
    if (randomFloat(random_state) < guide_probability) {
        if (!samplePowerCosine(guide_axis, guide_exponent, random_state,
                              continuation_direction, guide_pdf)) {
            return false;
        }
        base_pdf = diffractionProposalPdf(edge, incoming_direction, continuation_direction);
    } else {
        if (!sampleDiffractionBaseProposal(edge, incoming_direction, random_state,
                                          continuation_direction, base_pdf)) {
            return false;
        }
        guide_pdf = powerCosinePdf(dot(guide_axis, continuation_direction), guide_exponent);
    }
    float pdf = (1.0 - guide_probability) * base_pdf + guide_probability * guide_pdf;
    float capture_weight = diffractionCaptureWeight(edge, incoming_direction,
                                                    distance_to_edge, incident_distance);
    if (!finiteFloat(pdf) || pdf <= 0.0 || capture_weight <= 0.0) {
        return false;
    }
    path_throughput *= capture_weight / pdf;
    return finiteFloat(path_throughput);
}
