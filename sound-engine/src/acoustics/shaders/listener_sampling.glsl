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

// Folds the part of an axis-centered lobe that crosses a physical surface back
// into the valid outgoing hemisphere. The sum of the original and mirrored
// densities integrates to one there, so grazing lobes do not randomly lose
// energy through rejected continuation samples.
float foldedPowerCosinePdf(vec3 direction, vec3 axis, vec3 normal, float exponent) {
    vec3 mirrored_direction = reflect(direction, normal);
    return powerCosinePdf(max(dot(direction, axis), 0.0), exponent)
        + powerCosinePdf(max(dot(mirrored_direction, axis), 0.0), exponent);
}
