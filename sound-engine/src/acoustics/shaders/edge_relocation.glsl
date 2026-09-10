// A continuous angular proposal followed by zero or more delta surfaces.
// Mirror unfolding relocates the entire delta suffix, not just its last leg.
// State belongs to one invocation and holds no external resources.
struct EdgeRelocationState {
    uint kind; // 0: uniform launch, 1: finite surface, 2: scalar edge
    vec3 anchor;
    float prefix_distance;
    vec3 virtual_anchor;
    mat3 rotation;
    uint delta_count;
    uint replay_rng;
    uint signature;
    vec3 surface_normal;
    SurfaceContinuationSample surface_proposal;
    uint edge_id;
    vec3 edge_incoming;
    float original_response;
};

void resetRelocationAnchor(inout EdgeRelocationState state, uint kind,
    vec3 anchor, float distance, uint rng) {
    state.kind = kind;
    state.anchor = anchor;
    state.prefix_distance = distance;
    state.virtual_anchor = anchor;
    state.rotation = mat3(1.0);
    state.delta_count = 0u;
    state.replay_rng = rng;
    state.signature = 0u;
}

uint relocationSignature(uint signature, uint material_id, uint branch) {
    return hashState(signature ^ hashState(material_id) ^ (branch + 0x917823u));
}

void unfoldRelocationSurface(inout vec3 virtual_anchor, inout mat3 rotation,
    vec3 hit, vec3 normal, uint branch) {
    if (branch == 2u) return; // Straight transmission leaves the image unchanged.
    virtual_anchor -= 2.0 * dot(virtual_anchor - hit, normal) * normal;
    rotation = (mat3(1.0) - 2.0 * outerProduct(normal, normal)) * rotation;
}

void recordSurfaceRelocation(inout EdgeRelocationState state,
    SurfaceContinuationSample proposal, vec3 hit, vec3 normal,
    uint material_id, float distance, uint rng) {
    if (proposal.delta) {
        unfoldRelocationSurface(state.virtual_anchor, state.rotation, hit, normal, proposal.branch);
        state.signature = relocationSignature(state.signature, material_id, proposal.branch);
        ++state.delta_count;
    } else {
        resetRelocationAnchor(state, 1u, hit, distance, rng);
        state.surface_normal = normal;
        state.surface_proposal = proposal;
        state.original_response = proposal.directional_pdf;
    }
}

float relocatedAngularResponse(EdgeRelocationState state, vec3 direction) {
    if (state.kind == 0u) return 1.0;
    if (state.kind == 2u) {
        return diffractionScalarResponse(diffraction_edges.edges[state.edge_id],
            state.edge_incoming, direction);
    }
    bool allowed = state.surface_proposal.branch == 2u
        ? dot(direction, state.surface_normal) < -DIRECTION_EPSILON
        : dot(direction, state.surface_normal) > DIRECTION_EPSILON;
    return allowed ? foldedPowerCosinePdf(direction, state.surface_proposal.axis,
        state.surface_normal, state.surface_proposal.exponent) : 0.0;
}

// Reconstruct the delta suffix using its original random stream. Every moved
// reflection must still hit real geometry and select the same material/branch
// sequence. End-image checks reject a change of mirror planes or path topology.
bool relocateEdgePrefix(EdgeRelocationState state, vec3 proxy, vec3 edge,
    inout float throughput, out vec3 incoming, out float distance,
    inout vec3 listener_direction) {
    float proxy_length = length(proxy - state.virtual_anchor);
    float edge_length = length(edge - state.virtual_anchor);
    if (min(proxy_length, edge_length) <= RAY_EPSILON) return false;
    incoming = (edge - state.virtual_anchor) / edge_length;
    vec3 anchor_direction = normalizeOrZero(transpose(state.rotation) * incoming);
    float original = state.kind == 0u ? 1.0 : state.original_response;
    float corrected = relocatedAngularResponse(state, anchor_direction);
    if (original <= 0.0 || corrected <= 0.0) return false;

    vec3 vertex = state.anchor;
    vec3 direction = anchor_direction;
    vec3 image = state.anchor;
    mat3 rotation = mat3(1.0);
    float walked = 0.0;
    uint rng = state.replay_rng;
    uint signature = 0u;
    for (uint step = 0u; step < state.delta_count; ++step) {
        hit_payload.hit_type = HIT_NONE;
        float remaining = edge_length - walked;
        if (remaining <= 2.0 * RAY_EPSILON) return false;
        traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, SURFACE_GEOMETRY_MASK,
            0, 0, 0, vertex + direction * RAY_EPSILON, RAY_EPSILON,
            direction, remaining - RAY_EPSILON, 0);
        if (hit_payload.hit_type != HIT_SURFACE) return false;
        vec3 hit = vertex + direction * (hit_payload.t + RAY_EPSILON);
        vec3 normal = normalizeOrZero(hit_payload.world_normal);
        uint material_id = hit_payload.data_id;
        SceneMaterial material = scene_materials.materials[material_id];
        SurfaceContinuationSample proposal;
        vec3 outgoing;
        float dummy_weight = 1.0;
        if (!sampleSurfaceContinuationTracked(material, direction, normal, rng,
                dummy_weight, outgoing, proposal) || !proposal.delta) return false;
        signature = relocationSignature(signature, material_id, proposal.branch);
        unfoldRelocationSurface(image, rotation, hit, normal, proposal.branch);
        walked += length(hit - vertex);
        vertex = hit;
        direction = outgoing;
    }
    float tolerance = max(8.0 * RAY_EPSILON, edge_length * 1.0e-5);
    if (signature != state.signature || length(image - state.virtual_anchor) > tolerance) return false;
    for (uint column = 0u; column < 3u; ++column) {
        if (length(rotation[column] - state.rotation[column]) > 1.0e-4) return false;
    }
    vec3 leg = edge - vertex;
    float leg_length = length(leg);
    if (leg_length <= RAY_EPSILON || length(leg - direction * leg_length) > tolerance
        || !traceVisibility(vertex, leg / leg_length, leg_length)) return false;
    distance = state.prefix_distance + walked + leg_length;
    if (distance > pc.maximum_path_length) return false;
    incoming = leg / leg_length;
    // Weight already contains f(old)/q(old). The proposal remains q(old):
    // multiply by f(new)/f(old), never replace it with q(new).
    float ratio = proxy_length / edge_length;
    throughput *= ratio * ratio * ratio * corrected / original;
    if (state.kind == 0u) listener_direction = anchor_direction;
    return finiteFloat(throughput) && finiteFloat(distance);
}
