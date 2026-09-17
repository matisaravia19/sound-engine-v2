// One iterative path, with a sampled straight/diffracted choice at every proxy.
// Local responses and angular sampling live in listener_transport.glsl.
// A straight proxy crossing costs work but is not an acoustic scattering vertex.

// Importance-sampling probability, not a physical energy split. Both event
// weights include the reciprocal probability, so neither contribution is halved.
const float DIFFRACTION_EVENT_PROBABILITY = 0.5;

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
    // Independent continuation stream; source connections do not consume it.
    uint random_state;
    // Work budget used by surfaces and all proxies, including straight crossings.
    uint interactions;
    // Distinguishes the uniform launch from later scattering. Straight proxy
    // crossings must not disable its projection correction or arrival direction.
    bool has_scattered;
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
        && abs(path.pressure_weight) >= pc.minimum_throughput
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
    float proxy_leg_length = length(interaction.position - path.vertex);
    interaction.position = edge.start + hit.barycentrics.y * edge.edge_dir;
    interaction.distance_to_edge = hit.barycentrics.x;

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

    // Relocation changes dOmega/dA by (proxy distance / edge distance)^3.
    // Only the initial uniform launch has this correction; later directional
    // PDFs require their own treatment, unchanged by this reorganization.
    if (!path.has_scattered) {
        float density_ratio = proxy_leg_length / edge_leg_length;
        path.pressure_weight *= density_ratio * density_ratio * density_ratio;
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
        sampled = sampleSurfaceContinuation(
            scene_materials.materials[interaction.data_id],
            path.direction,
            interaction.normal,
            path.random_state,
            path.pressure_weight,
            direction
        );
    } else {
        sampled = sampleDiffractionContinuation(
            diffraction_edges.edges[interaction.data_id],
            path.direction,
            path.random_state,
            path.pressure_weight,
            direction
        );
    }

    if (!sampled) {
        return false;
    }

    path.origin = interaction.position + direction * RAY_EPSILON;
    path.vertex = interaction.position;
    path.direction = direction;

    return finiteVector(path.origin) && finiteFloat(path.distance);
}

// Select one entire edge event, including its source connection. Applying the
// inverse probability before that connection weights both local and future
// contributions. Every proxy uses the same choice within the shared budget.
bool sampleDiffractionEvent(inout ListenerPath path) {
    bool diffract = randomFloat(path.random_state) < DIFFRACTION_EVENT_PROBABILITY;
    float probability = diffract
        ? DIFFRACTION_EVENT_PROBABILITY
        : 1.0 - DIFFRACTION_EVENT_PROBABILITY;
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

    if (hit.hit_type == HIT_DIFFRACTION_EDGE && !sampleDiffractionEvent(path)) {
        return passThroughProxy(path, hit);
    }

    PathInteraction interaction;
    if (!prepareInteraction(path, hit, interaction)) {
        return false;
    }

    connectInteractionToSource(path, interaction, normalization);
    path.has_scattered = true;

    if (path.interactions >= pc.max_bounces || !continuePath(path, interaction)) {
        return false;
    }

    return true;
}
