#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_shader_atomic_float : require

// This shader is deliberately iterative. Each launch invocation owns one
// listener-originated acoustic path and issues only depth-one traversal calls.
// Miss and closest-hit stages return compact data; they never continue a path
// or write the IR. Keeping estimator state here makes branch probabilities,
// direction PDFs, path lengths, and termination rules explicit in one place.
//
// Direction convention:
// - `direction` points away from the listener along the backward-traced path.
// - `incoming_direction` is that same listener-to-vertex direction at a hit.
// - `source_direction` points from the current vertex toward the source.
// - `initial_direction` is retained for listener-space panning even after bends.
//
// For every non-direct completed path the pressure estimator is:
//
//   source_amplitude * path_throughput * local_source_response
//       * (4 * PI / listener_ray_count) / total_path_length
//
// `path_throughput` contains only sampled continuation responses. It never
// includes spreading or the initial spherical sampling normalization.

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
    float absorption;
    float scattering;
    float transmission;
    float roughness;
};

layout(std430, set = 0, binding = 3) readonly buffer SceneMaterialBuffer {
    SceneMaterial materials[];
} scene_materials;

struct DiffractionEdge {
    vec3 start;
    float diffraction_radius;
    vec3 end;
    float base_strength;
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

float specularExponent(float roughness) {
    // Even a perfectly smooth material uses a finite lobe. This avoids delta
    // distributions, which would require a separate estimator and PDF rules.
    return mix(4096.0, 128.0, clamp(roughness, 0.0, 1.0));
}

float roughExponent(float roughness) {
    // The rough branch stays much wider than the specular branch at every
    // roughness, reaching a cosine-weighted shape at roughness one.
    return mix(64.0, 1.0, clamp(roughness, 0.0, 1.0));
}

// Material coefficients are authored as energy fractions. Reflection is the
// energy remaining after absorption and straight transmission; scattering
// divides only that reflected portion into specular and rough components.
void materialEnergies(
    SceneMaterial material,
    out float specular_energy,
    out float rough_energy,
    out float transmission_energy
) {
    float absorption = clamp(material.absorption, 0.0, 1.0);
    transmission_energy = min(clamp(material.transmission, 0.0, 1.0), 1.0 - absorption);
    float reflection_energy = max(0.0, 1.0 - absorption - transmission_energy);
    float rough_fraction = clamp(material.scattering, 0.0, 1.0);
    specular_energy = reflection_energy * (1.0 - rough_fraction);
    rough_energy = reflection_energy * rough_fraction;
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

// Converts geometric length to a fractional sample position and applies the
// output mode's listener-space spatialization before IR deposition.
void depositPressure(float total_path_length, float amplitude, vec3 listener_arrival_direction) {
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
    depositPressure(distance_to_source, pc.source_amplitude * spreading, source_direction);
}

// -----------------------------------------------------------------------------
// Surface next-event estimation and continuation
// -----------------------------------------------------------------------------

// Evaluates all surface branches that can connect this vertex to the source.
// This is evaluation, not sampling, so valid specular and rough reflection
// responses are summed. Transmission is evaluated on the opposite hemisphere.
float surfaceSourceResponse(SceneMaterial material, vec3 incoming_direction, vec3 normal, vec3 source_direction) {
    float specular_energy;
    float rough_energy;
    float transmission_energy;
    materialEnergies(material, specular_energy, rough_energy, transmission_energy);

    // These are pressure-domain local responses. The normalized lobe value
    // includes directional density, while sqrt converts energy to pressure.
    float response = 0.0;
    // Because incoming_direction points toward the surface, reflect() produces
    // the backward path's ideal outgoing reflection direction.
    vec3 reflection_axis = normalizeOrZero(reflect(incoming_direction, normal));
    if (dot(source_direction, normal) > DIRECTION_EPSILON) {
        float reflection_cosine = max(dot(source_direction, reflection_axis), 0.0);
        response += sqrt(specular_energy)
            * powerCosinePdf(reflection_cosine, specularExponent(material.roughness));
        response += sqrt(rough_energy)
            * powerCosinePdf(reflection_cosine, roughExponent(material.roughness));
    }
    if (dot(source_direction, normal) < -DIRECTION_EPSILON) {
        // Straight transmission remains centered on incoming_direction: the
        // backward path continues through the surface without refraction.
        response += sqrt(transmission_energy)
            * powerCosinePdf(
                max(dot(source_direction, normalizeOrZero(incoming_direction)), 0.0),
                specularExponent(material.roughness)
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
    float local_response = surfaceSourceResponse(material, incoming_direction, normal, source_direction);
    if (local_response <= 0.0) {
        return;
    }
    // Offset toward the same hemisphere as the source connection. Always using
    // +normal would make transmission rays start on the wrong side.
    vec3 offset_normal = dot(source_direction, normal) >= 0.0 ? normal : -normal;
    if (!traceVisibility(hit_position + offset_normal * RAY_EPSILON, source_direction, source_distance)) {
        return;
    }

    // Apply spreading once to the complete listener-to-vertex-to-source path.
    // path_throughput excludes the initial spherical Monte Carlo normalization.
    float spreading = 1.0 / max(total_path_length, RAY_EPSILON);
    // A normalized reflection lobe is a density in source-connection solid
    // angle, while this estimator integrates uniformly sampled listener solid
    // angle. For a planar first reflection, perturbing the listener direction
    // changes the source-connection mismatch by total_length/source_distance
    // in each of its two angular dimensions. This determinant converts between
    // those measures and makes the narrow-lobe limit reproduce the exact image
    // source. It is intentionally evaluated here as an experiment: curved
    // geometry and later vertices require the full accumulated path Jacobian.
    float angular_scale = total_path_length / max(source_distance, RAY_EPSILON);
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
    out vec3 continuation_direction
) {
    float energies[3];
    materialEnergies(material, energies[0], energies[1], energies[2]);
    // Branch selection operates in the pressure domain. Choosing one branch,
    // rather than emitting all three, keeps one continuation ray per invocation.
    float weights[3] = float[3](sqrt(energies[0]), sqrt(energies[1]), sqrt(energies[2]));
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
    float exponent = branch == 1u ? roughExponent(material.roughness) : specularExponent(material.roughness);
    float direction_pdf;
    if (!samplePowerCosine(axis, exponent, random_state, continuation_direction, direction_pdf)) {
        return false;
    }
    // Finite lobes have tails that can cross the surface plane. Reject those
    // directions instead of launching a branch into the physically wrong side.
    if ((branch == 2u && dot(continuation_direction, normal) >= -DIRECTION_EPSILON)
        || (branch != 2u && dot(continuation_direction, normal) <= DIRECTION_EPSILON)) {
        return false;
    }

    // Divide by both discrete branch probability and continuous direction PDF.
    // For the matched power-cosine sampler the PDF cancels analytically, but
    // writing the full estimator avoids hiding that invariant.
    float response = sqrt(energies[branch]) * direction_pdf;
    path_throughput *= response / (branch_probability * direction_pdf);
    return finiteFloat(path_throughput) && finiteVector(continuation_direction);
}

// -----------------------------------------------------------------------------
// Diffraction geometry and next-event estimation
// -----------------------------------------------------------------------------

// Chooses which authored shadow boundary bounds the diffraction interval.
bool chooseShadowDirection(DiffractionEdge edge, vec3 perpendicular_direction, out vec3 shadow_direction) {
    // The proxy plane is not itself the diffraction domain. Reject incidence
    // inside the solid wedge, then select the authored shadow boundary from the
    // side of the edge on which the projected incoming direction lies.
    if (dot(perpendicular_direction, -edge.bisector_dir) >= edge.solid_wedge_cos - DIRECTION_EPSILON) {
        return false;
    }
    float side = dot(perpendicular_direction, edge.positive_plane_normal);
    if (abs(side) <= DIRECTION_EPSILON) {
        return false;
    }
    shadow_direction = side > 0.0 ? edge.positive_shadow_dir : edge.negative_shadow_dir;
    return true;
}

vec3 rotateAroundAxis(vec3 value, vec3 axis, float angle) {
    // Rodrigues' formula rotates the cross-section direction around the edge.
    float sine = sin(angle);
    float cosine = cos(angle);
    return value * cosine + cross(axis, value) * sine + axis * dot(axis, value) * (1.0 - cosine);
}

// Builds the valid one-dimensional angular sampling domain in the plane
// perpendicular to the edge. Returns false for solid-wedge incidence, nearly
// edge-parallel rays, and proxy-boundary hits whose interval has collapsed.
bool diffractionDomain(
    DiffractionEdge edge,
    vec3 incoming_direction,
    float distance_to_edge,
    out vec3 incoming_perpendicular,
    out float perpendicular_length,
    out vec3 parallel_component,
    out float rotation_sign,
    out float maximum_angle
) {
    // Work in the cross-section perpendicular to the edge. The edge-parallel
    // component is stored separately and restored after sampling a bend.
    parallel_component = dot(incoming_direction, edge.edge_dir) * edge.edge_dir;
    vec3 perpendicular = incoming_direction - parallel_component;
    perpendicular_length = length(perpendicular);
    if (perpendicular_length <= DIRECTION_EPSILON || edge.diffraction_radius <= DIRECTION_EPSILON) {
        return false;
    }
    incoming_perpendicular = perpendicular / perpendicular_length;

    vec3 shadow_direction;
    if (!chooseShadowDirection(edge, incoming_perpendicular, shadow_direction)) {
        return false;
    }
    // The interval starts at straight passage and rotates toward the selected
    // shadow boundary. Large authored wedges are capped to a 90-degree bend.
    float full_angle = min(
        acos(clamp(dot(incoming_perpendicular, shadow_direction), -1.0, 1.0)),
        0.5 * PI
    );
    // The interval collapses continuously at the proxy-radius boundary so the
    // virtual strip does not introduce a discontinuous diffraction event.
    float bend = clamp(1.0 - distance_to_edge / edge.diffraction_radius, 0.0, 1.0);
    maximum_angle = full_angle * bend;
    float orientation = dot(cross(incoming_perpendicular, shadow_direction), edge.edge_dir);
    rotation_sign = orientation >= 0.0 ? 1.0 : -1.0;
    return maximum_angle > DIRECTION_EPSILON && finiteFloat(maximum_angle);
}

// Evaluates whether the edge-to-source connection belongs to the same angular
// domain available to a sampled continuation and, if so, returns its pressure
// response including radial normalization and angular density.
bool diffractionSourceResponse(
    DiffractionEdge edge,
    vec3 incoming_direction,
    vec3 source_direction,
    float distance_to_edge,
    out float response
) {
    vec3 incoming_perpendicular;
    float perpendicular_length;
    vec3 parallel_component;
    float rotation_sign;
    float maximum_angle;
    if (!diffractionDomain(
        edge,
        incoming_direction,
        distance_to_edge,
        incoming_perpendicular,
        perpendicular_length,
        parallel_component,
        rotation_sign,
        maximum_angle
    )) {
        return false;
    }

    // Project the source connection into the same cross-section and accept it
    // only when it falls inside the same oriented interval used by continuation.
    vec3 source_perpendicular = source_direction
        - dot(source_direction, edge.edge_dir) * edge.edge_dir;
    source_perpendicular = normalizeOrZero(source_perpendicular);
    // Directions on the authored wall boundary are valid diffraction limits;
    // reject only directions measurably inside the solid wedge.
    if (dot(source_perpendicular, source_perpendicular) <= 0.0
        || dot(source_perpendicular, -edge.bisector_dir) > edge.solid_wedge_cos + DIRECTION_EPSILON) {
        return false;
    }

    // A dot product gives only the unsigned angle. The cross-product sign is
    // also required so a source across the opposite shadow boundary is rejected.
    float source_angle = acos(clamp(dot(incoming_perpendicular, source_perpendicular), -1.0, 1.0));
    float source_orientation = dot(cross(incoming_perpendicular, source_perpendicular), edge.edge_dir);
    float source_sign = source_orientation >= 0.0 ? 1.0 : -1.0;
    if (source_sign != rotation_sign || source_angle > maximum_angle + DIRECTION_EPSILON) {
        return false;
    }

    // Dividing the triangular radial kernel by the proxy radius compensates for
    // the number of listener rays intercepted as the strip width changes.
    float radial_kernel = clamp(1.0 - distance_to_edge / edge.diffraction_radius, 0.0, 1.0)
        / edge.diffraction_radius;
    // The cross-section angle is sampled uniformly, hence this normalized PDF.
    float angular_pdf = 1.0 / maximum_angle;
    response = sqrt(clamp(edge.base_strength, 0.0, 1.0)) * radial_kernel * angular_pdf;
    return response > 0.0 && finiteFloat(response);
}

// Performs next-event estimation for listener -> ... -> edge -> source.
void connectDiffractionToSource(
    DiffractionEdge edge,
    vec3 edge_origin,
    vec3 incoming_direction,
    float distance_to_edge,
    float edge_path_distance,
    float path_throughput,
    float monte_carlo_normalization,
    vec3 listener_arrival_direction
) {
    // The virtual-plane hit is projected onto the physical finite edge. The
    // listener-to-edge distance and this edge-to-source distance remain local
    // to the current path so nearby proxy hits retain distinct delays.
    vec3 to_source = pc.source_position - edge_origin;
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
        local_response
    )) {
        return;
    }
    // Offset along the connection rather than an arbitrary proxy normal. The
    // proxy itself is invisible to this surface-only visibility trace.
    if (!traceVisibility(edge_origin + source_direction * RAY_EPSILON, source_direction, source_distance)) {
        return;
    }
    float spreading = 1.0 / max(total_path_length, RAY_EPSILON);
    float amplitude = pc.source_amplitude
        * path_throughput
        * local_response
        * spreading
        * monte_carlo_normalization;
    depositPressure(total_path_length, amplitude, listener_arrival_direction);
}

// Chooses either transparent traversal of the virtual strip or one diffracted
// continuation. The two origins and distance updates intentionally differ:
// straight traversal stays on the proxy-plane intersection, while diffraction
// detours to the closest physical edge point.
bool sampleDiffractionContinuation(
    DiffractionEdge edge,
    vec3 incoming_origin,
    vec3 plane_hit_position,
    vec3 edge_origin,
    vec3 incoming_direction,
    float incoming_path_distance,
    float hit_distance,
    float distance_to_edge,
    inout uint random_state,
    inout float path_throughput,
    out vec3 continuation_origin,
    out vec3 continuation_direction,
    out float continuation_path_distance
) {
    vec3 incoming_perpendicular;
    float perpendicular_length;
    vec3 parallel_component;
    float rotation_sign;
    float maximum_angle;
    bool valid_diffraction = diffractionDomain(
        edge,
        incoming_direction,
        distance_to_edge,
        incoming_perpendicular,
        perpendicular_length,
        parallel_component,
        rotation_sign,
        maximum_angle
    );

    float bend = clamp(1.0 - distance_to_edge / edge.diffraction_radius, 0.0, 1.0);
    // bend is an unnormalized interaction strength used to reserve the
    // remaining energy for transparent passage. The pressure response below
    // separately uses bend/radius to compensate for proxy interception width.
    float diffraction_energy = valid_diffraction
        ? clamp(edge.base_strength, 0.0, 1.0) * bend
        : 0.0;
    float straight_energy = max(0.0, 1.0 - diffraction_energy);
    // Transparent passage and diffraction are mutually exclusive branches.
    // Their probabilities are proportional to pressure response, matching the
    // surface branch policy while keeping exactly one continuation.
    float straight_weight = sqrt(straight_energy);
    float radial_kernel = valid_diffraction
        ? bend / edge.diffraction_radius
        : 0.0;
    float diffraction_weight = sqrt(clamp(edge.base_strength, 0.0, 1.0))
        * radial_kernel;
    float weight_sum = straight_weight + diffraction_weight;
    if (weight_sum <= 0.0 || !finiteFloat(weight_sum)) {
        return false;
    }

    float branch_sample = randomFloat(random_state) * weight_sum;
    if (branch_sample < straight_weight || diffraction_energy <= 0.0) {
        float branch_probability = straight_weight / weight_sum;
        // Straight passage is deterministic once selected, so it has no
        // continuous direction PDF—only the discrete branch probability.
        path_throughput *= sqrt(straight_energy) / branch_probability;
        // Straight passage advances from the virtual plane and accounts for the
        // ordinary ray parameter, since it never detours to the physical edge.
        continuation_origin = plane_hit_position + incoming_direction * RAY_EPSILON;
        continuation_direction = incoming_direction;
        continuation_path_distance = incoming_path_distance + hit_distance;
        return finiteFloat(path_throughput);
    }

    float branch_probability = diffraction_weight / weight_sum;
    // Uniform sampling over [0, maximum_angle] gives PDF 1/maximum_angle.
    float sampled_angle = randomFloat(random_state) * maximum_angle;
    float angular_pdf = 1.0 / maximum_angle;
    // Diffraction starts at the closest physical edge point. Only the
    // cross-section component bends; the component along the edge is conserved.
    vec3 bent_perpendicular = rotateAroundAxis(
        incoming_perpendicular,
        edge.edge_dir,
        rotation_sign * sampled_angle
    );
    continuation_direction = normalizeOrZero(
        parallel_component + bent_perpendicular * perpendicular_length
    );
    if (!finiteVector(continuation_direction) || dot(continuation_direction, continuation_direction) <= 0.0) {
        return false;
    }
    // As with surface lobes, retain response / PDF explicitly even though the
    // matched uniform angular PDF cancels in this throughput update.
    float response = diffraction_weight * angular_pdf;
    path_throughput *= response / (branch_probability * angular_pdf);
    continuation_origin = edge_origin + continuation_direction * RAY_EPSILON;
    // This branch travels from the previous origin to the edge, not merely to
    // the proxy plane. That distinction is required for correct IR delay.
    continuation_path_distance = incoming_path_distance + length(edge_origin - incoming_origin);
    return finiteFloat(path_throughput) && finiteFloat(continuation_path_distance);
}

// -----------------------------------------------------------------------------
// Per-listener-path state machine
// -----------------------------------------------------------------------------

void main() {
    uint ray_index = gl_LaunchIDEXT.x;
    // Direct sound is a deterministic visibility query, so exactly one
    // invocation performs it regardless of listener_ray_count.
    if (ray_index == 0u) {
        evaluateDirectPath();
    }
    // max_bounces zero disables all vertex paths but intentionally leaves the
    // direct test above active. Zero source amplitude has nothing left to trace.
    if (pc.max_bounces == 0u || pc.source_amplitude == 0.0) {
        return;
    }

    // Each invocation receives an independent deterministic continuation stream.
    // The Fibonacci rotation depends only on the seed, preserving even coverage.
    uint random_state = hashState(ray_index ^ pc.sampling_seed ^ 0xa511e9b3u);
    uint azimuth_rotation = hashState(pc.sampling_seed);
    vec3 initial_direction = fibonacciSphereDirection(ray_index, pc.listener_ray_count, azimuth_rotation);
    vec3 origin = pc.listener_position;
    vec3 direction = initial_direction;
    // Throughput contains only accumulated local transport. The 4*PI/N
    // estimator weight stays separate and is applied only to completed paths.
    float path_throughput = 1.0;
    float traveled_distance = 0.0;
    float monte_carlo_normalization = 4.0 * PI / float(pc.listener_ray_count);

    for (uint bounce = 0u; bounce < pc.max_bounces; ++bounce) {
        // This is deterministic truncation only. There is no Russian roulette;
        // identical inputs always execute the same stopping policy.
        if (!finiteFloat(path_throughput)
            || abs(path_throughput) < pc.minimum_throughput
            || !finiteVector(origin)
            || !finiteVector(direction)
            || traveled_distance >= pc.maximum_path_length) {
            break;
        }

        hit_payload.hit_type = HIT_NONE;
        hit_payload.t = 0.0;
        // SBT offset 0 selects surface continuation. Surface instances have
        // instance offset 0, diffraction instances offset 2, and stride zero;
        // this routes them to records 0 and 2 respectively. Miss index 0 returns
        // HIT_NONE without recursively continuing the path.
        traceRayEXT(
            tlas,
            gl_RayFlagsOpaqueEXT,
            CONTINUATION_GEOMETRY_MASK,
            0,
            0,
            0,
            origin,
            RAY_EPSILON,
            direction,
            max(pc.maximum_path_length - traveled_distance, RAY_EPSILON),
            0
        );
        // A miss has no environment contribution in the current point-source
        // model, so it terminates the path immediately.
        if (hit_payload.hit_type == HIT_NONE || !finiteFloat(hit_payload.t) || hit_payload.t < 0.0) {
            break;
        }

        vec3 hit_position = origin + direction * hit_payload.t;
        if (hit_payload.hit_type == HIT_SURFACE) {
            // Step 1: account for distance to the physical surface.
            // Step 2: explicitly connect that vertex to the source.
            // Step 3: if another bounce is allowed, sample one continuation.
            float hit_path_distance = traveled_distance + hit_payload.t;
            if (hit_path_distance > pc.maximum_path_length) {
                break;
            }

            vec3 normal = normalizeOrZero(hit_payload.world_normal);
            SceneMaterial material = scene_materials.materials[hit_payload.data_id];
            connectSurfaceToSource(
                hit_position,
                direction,
                normal,
                material,
                hit_path_distance,
                path_throughput,
                monte_carlo_normalization,
                initial_direction
            );
            // Source connection belongs to this accepted vertex and is therefore
            // evaluated before the bounce-limit continuation check.
            if (bounce + 1u >= pc.max_bounces) {
                break;
            }

            vec3 continuation_direction;
            if (!sampleSurfaceContinuation(
                material,
                direction,
                normal,
                random_state,
                path_throughput,
                continuation_direction
            )) {
                break;
            }
            // Commit the new path state only after every sample and finite-state
            // check succeeds. Failed samples terminate without partial updates.
            origin = hit_position + continuation_direction * RAY_EPSILON;
            direction = continuation_direction;
            traveled_distance = hit_path_distance;
        } else if (hit_payload.hit_type == HIT_DIFFRACTION_EDGE) {
            // The traversal hit lies on a virtual plane strip. Recover the
            // closest physical point from the packed intersection attributes.
            DiffractionEdge edge = diffraction_edges.edges[hit_payload.data_id];
            float distance_to_edge = hit_payload.barycentrics.x;
            float edge_t = hit_payload.barycentrics.y;
            vec3 edge_origin = edge.start + edge.edge_dir * edge_t;
            // Use listener-to-edge length for diffraction delays, not the
            // shorter ray parameter that reached the proxy plane.
            float edge_path_distance = traveled_distance + length(edge_origin - origin);
            if (edge_path_distance > pc.maximum_path_length) {
                break;
            }

            connectDiffractionToSource(
                edge,
                edge_origin,
                direction,
                distance_to_edge,
                edge_path_distance,
                path_throughput,
                monte_carlo_normalization,
                initial_direction
            );
            // As for surfaces, this vertex's source connection remains valid at
            // the final permitted bounce; only continuation is suppressed.
            if (bounce + 1u >= pc.max_bounces) {
                break;
            }

            vec3 continuation_origin;
            vec3 continuation_direction;
            float continuation_path_distance;
            if (!sampleDiffractionContinuation(
                edge,
                origin,
                hit_position,
                edge_origin,
                direction,
                traveled_distance,
                hit_payload.t,
                distance_to_edge,
                random_state,
                path_throughput,
                continuation_origin,
                continuation_direction,
                continuation_path_distance
            )) {
                break;
            }
            // The continuation helper selects the correct origin and distance
            // convention for either transparent or diffracted traversal.
            origin = continuation_origin;
            direction = continuation_direction;
            traveled_distance = continuation_path_distance;
        } else {
            // Defensive termination for payload values unknown to this shader.
            break;
        }
    }
}
