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
