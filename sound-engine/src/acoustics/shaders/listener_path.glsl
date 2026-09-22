// One iterative path, with a sampled straight/diffracted choice at every proxy.
// Local responses and angular sampling live in listener_transport.glsl.
// A straight proxy crossing costs work but is not an acoustic scattering vertex.

// One listener-originated path; no deferred branch or recursive path state.
struct ListenerPath {
    // Epsilon-offset origin used only for traversal.
    vec3 origin;
    // Physical vertex used for acoustic distances, excluding numerical offsets.
    vec3 vertex;
    // Backward propagation direction, from the listener toward the next vertex.
    vec3 direction;
    // Arrival direction retained for listener-space panning.
    vec3 listener_direction;
    // Local responses and sampling corrections; excludes 1/d and 4*PI/N.
    float pressure_weight;
    // Acoustic length from the listener to the current accepted vertex.
    float distance;
    // Accumulated length at the last diffracting vertex; zero at launch.
    // Subsequent spreading uses distance from this vertex, not from the listener.
    float spreading_origin_distance;
    // Independent continuation stream; source connections do not consume it.
    uint random_state;
    // Work budget used by surfaces and all proxies, including straight crossings.
    uint interactions;
    // Distinguishes the uniform launch for projection and arrival direction.
    // Straight proxy crossings leave the physical scattering state unchanged.
    bool has_scattered;
    // The previous diffraction response awaits the next physical vertex.
    bool pending_diffraction;
    // Edge defining the pending response; valid only while pending_diffraction.
    uint pending_edge_id;
    // Incident direction at that edge, before sampling its outgoing proposal.
    vec3 pending_incoming_direction;
    // One initial ideal reflection can be relocated with the first edge.
    // Finite lobes and longer reflection prefixes retain their approximation.
    bool initial_mirror;
    // Plane and material of that ray-discovered reflection, while valid.
    vec3 mirror_normal;
    uint mirror_material;
};

// A prepared physical interaction. A proxy hit is relocated before this record
// is used for source connection or direction sampling.
struct PathInteraction {
    // Surface or edge tag selecting the local response and sampler.
    uint type;
    // Material or edge index, according to type.
    uint data_id;
    // Physical surface or edge point in world meters.
    vec3 position;
    // Surface normal; unused for edge interactions.
    vec3 normal;
    // Original proxy distance to the edge, retained for kernel normalization.
    float distance_to_edge;
};

// Numerical truncation is shared by every walk; there is no roulette termination.
bool canTracePath(ListenerPath path) {
    return path.interactions < pc.max_bounces
        && finiteFloat(path.pressure_weight)
        && (path.pending_diffraction || abs(path.pressure_weight) >= pc.minimum_throughput)
        && finiteVector(path.origin)
        && finiteVector(path.direction)
        && path.distance + length(path.origin - path.vertex) < pc.maximum_path_length;
}

// Visibility has its own surface-only payload. This query discovers the next
// interaction and never connects to the source or continues from closest-hit.
bool traceNextInteraction(ListenerPath path, out HitPayload hit) {
    hit_payload.hit_type = HIT_NONE;
    hit_payload.t = 0.0;

    // Instance SBT offsets route surfaces and proxies to their hit records.
    traceRayEXT(
        tlas,
        gl_RayFlagsOpaqueEXT,
        CONTINUATION_GEOMETRY_MASK,
        0,
        0,
        0,
        path.origin,
        RAY_EPSILON,
        path.direction,
        max(pc.maximum_path_length - path.distance - length(path.origin - path.vertex), RAY_EPSILON),
        0
    );

    hit = hit_payload;
    return hit.hit_type != HIT_NONE && finiteFloat(hit.t) && hit.t >= 0.0;
}

// Moving P to E also moves the stationary point on a preceding mirror. Trace
// the corrected first leg against real solids: the remembered plane alone
// cannot establish that its finite surface exists or remains unobstructed.
bool relocateInitialMirror(inout ListenerPath path, vec3 edge_position, vec3 proxy_position) {
    vec3 listener = pc.listener_position;
    vec3 normal = path.mirror_normal;
    vec3 virtual_listener = listener - 2.0 * dot(listener - path.vertex, normal) * normal;
    vec3 unfolded_leg = edge_position - virtual_listener;
    float denominator = dot(unfolded_leg, normal);
    if (abs(denominator) <= DIRECTION_EPSILON) {
        return false;
    }

    float fraction = dot(path.vertex - virtual_listener, normal) / denominator;
    if (fraction <= 0.0 || fraction >= 1.0) {
        return false;
    }

    vec3 mirror_position = virtual_listener + fraction * unfolded_leg;
    vec3 first_leg = mirror_position - listener;
    float first_length = length(first_leg);
    if (first_length <= RAY_EPSILON) {
        return false;
    }

    vec3 first_direction = first_leg / first_length;
    hit_payload.hit_type = HIT_NONE;
    traceRayEXT(
        tlas,
        gl_RayFlagsOpaqueEXT,
        SURFACE_GEOMETRY_MASK,
        0, 0, 0,
        listener,
        RAY_EPSILON,
        first_direction,
        first_length + 4.0 * RAY_EPSILON,
        0
    );

    if (hit_payload.hit_type != HIT_SURFACE
        || hit_payload.data_id != path.mirror_material
        || abs(hit_payload.t - first_length) > 4.0 * RAY_EPSILON
        || abs(dot(normalizeOrZero(hit_payload.world_normal), normal)) < 0.9999) {
        return false;
    }

    // Reflection preserves solid angle. Its projection density is measured
    // from the unfolded listener, rather than from the old reflection point.
    float density_ratio = length(proxy_position - virtual_listener) / length(unfolded_leg);
    path.pressure_weight *= density_ratio * density_ratio * density_ratio;
    path.vertex = mirror_position;
    path.distance = first_length;
    path.listener_direction = first_direction;
    return true;
}

// Relocation belongs to interaction preparation, not to the continuation
// sampler. Both the source connection and the next ray use the same edge point.
bool prepareInteraction(
    inout ListenerPath path,
    HitPayload hit,
    out PathInteraction interaction
) {
    interaction.type = hit.hit_type;
    interaction.data_id = hit.data_id;
    interaction.position = path.origin + path.direction * hit.t;
    interaction.normal = vec3(0.0);
    interaction.distance_to_edge = 0.0;

    if (hit.hit_type == HIT_SURFACE) {
        interaction.normal = normalizeOrZero(hit.world_normal);
        path.distance += length(interaction.position - path.vertex);
        return path.distance <= pc.maximum_path_length;
    }

    if (hit.hit_type != HIT_DIFFRACTION_EDGE) {
        return false;
    }

    DiffractionEdge edge = diffraction_edges.edges[hit.data_id];
    vec3 proxy_position = interaction.position;
    float proxy_leg_length = length(interaction.position - path.vertex);
    interaction.position = edge.start + hit.barycentrics.y * edge.edge_dir;
    interaction.distance_to_edge = hit.barycentrics.x;

    if (path.initial_mirror && !relocateInitialMirror(path, interaction.position, proxy_position)) {
        return false;
    }

    vec3 edge_leg = interaction.position - path.vertex;
    float edge_leg_length = length(edge_leg);
    if (edge_leg_length <= RAY_EPSILON) {
        return false;
    }

    path.direction = edge_leg / edge_leg_length;
    path.distance += edge_leg_length;

    // The proxy leg may be clear while the relocated physical leg is blocked.
    if (path.distance > pc.maximum_path_length
        || !traceVisibility(path.vertex, path.direction, edge_leg_length)) {
        return false;
    }

    // Convert proxy area density to the relocated edge direction. For a
    // continued diffraction, its PDF remains at P and its response is evaluated
    // at E below. Surface delta/finite-lobe mixtures keep their approximation;
    // this solid-angle correction does not apply to an ideal mirror delta.
    if (!path.has_scattered || path.pending_diffraction) {
        float density_ratio = proxy_leg_length / edge_leg_length;
        path.pressure_weight *= density_ratio * density_ratio * density_ratio;
    }

    if (!path.has_scattered) {
        path.listener_direction = path.direction;
    }

    return true;
}

// Both local evaluators use surface-only visibility. A blocked connection does
// not terminate the path: its independently sampled continuation can survive.
void connectInteractionToSource(
    ListenerPath path,
    PathInteraction interaction,
    float normalization
) {
    if (interaction.type == HIT_SURFACE) {
        connectSurfaceToSource(
            interaction.position,
            path.direction,
            interaction.normal,
            scene_materials.materials[interaction.data_id],
            path.distance,
            path.spreading_origin_distance,
            path.pressure_weight,
            normalization,
            path.listener_direction
        );
    } else {
        connectDiffractionToSource(
            diffraction_edges.edges[interaction.data_id],
            interaction.position,
            path.direction,
            interaction.distance_to_edge,
            path.distance,
            path.spreading_origin_distance,
            path.pressure_weight,
            normalization,
            path.listener_direction
        );
    }
}

// Only the local sampler differs by interaction type. Position, epsilon offset
// and path-state updates are shared; no distance is added a second time here.
bool continuePath(inout ListenerPath path, PathInteraction interaction) {
    vec3 direction;
    bool sampled;

    if (interaction.type == HIT_SURFACE) {
        bool mirror_delta;
        sampled = sampleSurfaceContinuation(
            scene_materials.materials[interaction.data_id],
            path.direction,
            interaction.normal,
            path.random_state,
            path.pressure_weight,
            direction,
            mirror_delta
        );
        path.initial_mirror = mirror_delta
            && all(equal(path.vertex, pc.listener_position));
        path.mirror_normal = interaction.normal;
        path.mirror_material = interaction.data_id;
    } else {
        sampled = sampleDiffractionContinuation(
            diffraction_edges.edges[interaction.data_id],
            interaction.position,
            path.direction,
            interaction.distance_to_edge,
            path.distance - path.spreading_origin_distance,
            path.random_state,
            path.pressure_weight,
            direction
        );
    }

    if (!sampled) {
        return false;
    }

    if (interaction.type == HIT_DIFFRACTION_EDGE) {
        path.initial_mirror = false;
        path.spreading_origin_distance = path.distance;
        path.pending_diffraction = true;
        path.pending_edge_id = interaction.data_id;
        path.pending_incoming_direction = path.direction;
    }

    path.origin = interaction.position + direction * RAY_EPSILON;
    path.vertex = interaction.position;
    path.direction = direction;

    return finiteVector(path.origin) && finiteFloat(path.distance);
}

// Select one entire edge event, including its source connection. Applying the
// inverse probability before that connection weights both local and future
// contributions. Every proxy uses the same choice within the shared budget.
bool sampleDiffractionEvent(inout ListenerPath path, HitPayload hit) {
    DiffractionEdge edge = diffraction_edges.edges[hit.data_id];
    vec3 edge_position = edge.start + hit.barycentrics.y * edge.edge_dir;
    vec3 incident_leg = edge_position - path.vertex;
    float incident_length = length(incident_leg);
    float capture_weight = diffractionCaptureWeight(
        edge,
        normalizeOrZero(incident_leg),
        hit.barycentrics.x,
        path.distance - path.spreading_origin_distance + incident_length
    );

    // Thin proxies have few hits but a large capture weight. Spend more of
    // those hits on diffraction, retaining both events with bounded variance.
    // This is proposal importance only: each chosen event divides by its PDF.
    if (capture_weight <= 0.0) {
        return false;
    }

    float diffraction_probability = clamp(capture_weight / (1.0 + capture_weight), 0.05, 0.95);
    bool diffract = randomFloat(path.random_state) < diffraction_probability;
    float probability = diffract ? diffraction_probability : 1.0 - diffraction_probability;
    path.pressure_weight /= probability;
    return diffract;
}

// Advance traversal without bending the acoustic path or inserting an IR
// contribution at a transparent proxy. Retaining the physical vertex preserves
// length, first-edge projection and visibility across multiple straight passes.
bool passThroughProxy(inout ListenerPath path, HitPayload hit) {
    vec3 position = path.origin + path.direction * hit.t;
    path.origin = position + path.direction * RAY_EPSILON;
    return canTracePath(path);
}

// Each accepted hit consumes the shared cost budget before either edge choice.
// The final scattering vertex still connects to the source; a final straight
// crossing cannot reach a later surface. Every continuation sees both masks.
bool tracePathStep(inout ListenerPath path, float normalization) {
    if (!canTracePath(path)) {
        return false;
    }

    HitPayload hit;
    if (!traceNextInteraction(path, hit)) {
        return false;
    }

    ++path.interactions;

    if (hit.hit_type == HIT_DIFFRACTION_EDGE && !sampleDiffractionEvent(path, hit)) {
        return passThroughProxy(path, hit);
    }

    PathInteraction interaction;
    if (!prepareInteraction(path, hit, interaction)) {
        return false;
    }

    // Evaluate the response along the realized physical leg, after P becomes E.
    // Straight proxy crossings preserve the pending response and its proposal.
    if (path.pending_diffraction) {
        vec3 outgoing = normalizeOrZero(interaction.position - path.vertex);
        path.pressure_weight *= diffractionScalarResponse(
            diffraction_edges.edges[path.pending_edge_id],
            path.pending_incoming_direction,
            outgoing,
            true
        );
        path.pending_diffraction = false;
    }

    connectInteractionToSource(path, interaction, normalization);
    path.has_scattered = true;

    if (path.interactions >= pc.max_bounces || !continuePath(path, interaction)) {
        return false;
    }

    return true;
}
