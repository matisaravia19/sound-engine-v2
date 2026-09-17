// Iterative path mechanics shared by ordinary and diffracted walks.
// The branch policy lives in ray generation; local responses and direction
// sampling live in listener_transport.glsl. No helper recursively traces paths.

// Two edge events bound the current higher-order approximation. They also
// consume the common interaction budget, just like surface hits.
const uint MAX_DIFFRACTION_EVENTS = 2u;

// Yield before processing the first proxy so the caller can preserve the
// ordinary suffix. Resuming the cached hit neither retraces nor counts it twice.
const uint WALK_TO_FIRST_EDGE = 0u;

// Ordinary suffixes must not fork again or receive diffraction contributions.
const uint WALK_SURFACES_ONLY = 1u;

// Resume the diffracted branch, subject to the common and edge-event budgets.
const uint WALK_WITH_DIFFRACTION = 2u;

// Step outcomes separate path termination from the first-proxy policy decision.
const uint PATH_FINISHED = 0u;
const uint PATH_CONTINUED = 1u;
const uint PATH_REACHED_FIRST_EDGE = 2u;

// One listener-originated path. Copying it preserves the pressure prefix and
// RNG stream when the current policy keeps both ordinary and diffracted walks.
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
    // Number of accepted vertices, including both surfaces and edges.
    uint interactions;
    // Number of accepted edge events within the common interaction budget.
    uint diffraction_events;
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
        && path.distance < pc.maximum_path_length;
}

// Visibility has its own surface-only payload. This query discovers the next
// interaction and never connects to the source or continues from closest-hit.
bool traceNextInteraction(ListenerPath path, uint walk_mode, out HitPayload hit) {
    bool surfaces_only = walk_mode == WALK_SURFACES_ONLY
        || path.diffraction_events >= MAX_DIFFRACTION_EVENTS;

    hit_payload.hit_type = HIT_NONE;
    hit_payload.t = 0.0;

    // Instance SBT offsets route surfaces and proxies to their hit records.
    traceRayEXT(
        tlas,
        gl_RayFlagsOpaqueEXT,
        surfaces_only ? SURFACE_GEOMETRY_MASK : CONTINUATION_GEOMETRY_MASK,
        0,
        0,
        0,
        path.origin,
        RAY_EPSILON,
        path.direction,
        max(pc.maximum_path_length - path.distance, RAY_EPSILON),
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
    if (path.interactions == 0u) {
        float density_ratio = proxy_leg_length / edge_leg_length;
        path.pressure_weight *= density_ratio * density_ratio * density_ratio;
        path.listener_direction = path.direction;
    }

    ++path.diffraction_events;
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

// One interaction of the shared loop. A nonempty pending_hit resumes a cached
// proxy. Yielding at the first edge leaves the prefix and vertex count unchanged.
uint tracePathStep(
    inout ListenerPath path,
    uint walk_mode,
    float normalization,
    inout HitPayload pending_hit
) {
    if (!canTracePath(path)) {
        return PATH_FINISHED;
    }

    if (pending_hit.hit_type == HIT_NONE
        && !traceNextInteraction(path, walk_mode, pending_hit)) {
        return PATH_FINISHED;
    }

    if (walk_mode == WALK_TO_FIRST_EDGE && pending_hit.hit_type == HIT_DIFFRACTION_EDGE) {
        return PATH_REACHED_FIRST_EDGE;
    }

    PathInteraction interaction;
    if (!prepareInteraction(path, pending_hit, interaction)) {
        return PATH_FINISHED;
    }

    pending_hit.hit_type = HIT_NONE;
    connectInteractionToSource(path, interaction, normalization);

    // The final allowed vertex still connects to the source. Only its
    // continuation is suppressed when the common budget is exhausted.
    ++path.interactions;
    if (path.interactions >= pc.max_bounces || !continuePath(path, interaction)) {
        return PATH_FINISHED;
    }

    return PATH_CONTINUED;
}
