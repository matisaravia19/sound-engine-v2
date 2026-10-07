// -----------------------------------------------------------------------------
// Surface next-event estimation and continuation
// -----------------------------------------------------------------------------

// Evaluates all surface branches that can connect this vertex to the source.
// This is evaluation, not sampling, so valid specular and rough reflection
// responses are summed within the reflection hemisphere.
float surfaceSourceResponse(
    SceneMaterial material,
    vec3 incoming_direction,
    vec3 normal,
    vec3 source_direction,
    float source_fraction
) {
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
    float local_response = surfaceSourceResponse(
        material,
        incoming_direction,
        normal,
        source_direction,
        source_distance / max(connection_path_length, RAY_EPSILON)
    );
    if (local_response <= 0.0) {
        return;
    }
    if (!traceVisibility(hit_position + normal * RAY_EPSILON, source_direction, source_distance)) {
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
    // rather than emitting both, keeps one continuation ray per invocation.
    float weights[2] = float[2](material.specular_pressure, material.rough_pressure);
    float weight_sum = weights[0] + weights[1];
    if (!finiteFloat(weight_sum) || weight_sum <= 0.0) {
        return false;
    }

    float branch_sample = randomFloat(random_state) * weight_sum;
    uint branch = branch_sample < weights[0] ? 0u : 1u;
    float branch_probability = weights[branch] / weight_sum;
    if (branch_probability <= 0.0 || !finiteFloat(branch_probability)) {
        return false;
    }

    // Both reflection branches share the ideal reflection axis but use
    // different widths.
    vec3 axis = normalizeOrZero(reflect(incoming_direction, normal));
    // A smooth material has a delta continuation. The finite source-connection
    // lobe regularizes point-source sampling; it must not scatter the ray again
    // at every smooth bounce. Only the discrete branch probability remains.
    if (branch == 0u && material.roughness <= 0.0) {
        mirror_delta = true;
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
    if (dot(continuation_direction, normal) <= 0.0) {
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
