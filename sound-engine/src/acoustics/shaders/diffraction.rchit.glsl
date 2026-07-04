#version 460
#extension GL_EXT_ray_tracing : require

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;

const float RAY_EPSILON = 0.001;
const float MIN_RAY_ENERGY = 0.000001;
const float DIRECTION_EPSILON = 0.00001;

// Payload is mutated before each child trace. Because traceRayEXT writes back
// into this same payload, every branch must restore the distance, energy, and
// depth that should be visible to the child ray it is about to launch.
struct AcousticPayload {
    float ray_energy;
    float path_distance;
    uint path_depth;
};

// GPU mirror of the CPU-preprocessed diffraction edge record. Directions are
// normalized in scene upload code so the shader can avoid per-hit basis setup:
// edge_dir is the finite segment axis, bisector_dir points into the open wedge,
// positive_plane_normal = cross(bisector_dir, edge_dir), and the shadow dirs are
// the two precomputed boundary directions for side-dependent bending.
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

layout(location = 0) rayPayloadInEXT AcousticPayload payload;

// Produced by the intersection shader. x is the virtual plane hit distance to
// the closest edge point, used for bend falloff. y is distance along the edge
// segment, used to recover the physical origin of the diffracted child ray.
hitAttributeEXT vec2 diffraction_hit;

layout(push_constant) uniform PushConstants {
    layout(offset = 0) vec3 source_position;
    layout(offset = 12) float source_energy;
    layout(offset = 16) vec3 listener_position;
    layout(offset = 28) float speed_of_sound;
    layout(offset = 32) float listener_radius;
    layout(offset = 36) float listener_volume;
    layout(offset = 44) uint ray_count;
    layout(offset = 48) uint max_bounces;
    layout(offset = 64) vec3 listener_right;
    layout(offset = 76) uint sample_rate;
    layout(offset = 80) uint sample_count;
    layout(offset = 84) uint output_channels;
} pc;

bool chooseShadowDir(DiffractionEdge edge, vec3 d_perpendicular_dir, out vec3 shadow_dir) {
    // The solid wedge is centered on -bisector_dir. Rejecting directions inside
    // it prevents a virtual diffraction plane from creating paths that travel
    // through the occluding wall volume.
    if (dot(d_perpendicular_dir, -edge.bisector_dir) >= edge.solid_wedge_cos - DIRECTION_EPSILON) {
        return false;
    }

    // The sign convention must match CPU preprocessing: positive_plane_normal
    // was built as cross(bisector_dir, edge_dir), so positive side selects s+.
    float side = dot(d_perpendicular_dir, edge.positive_plane_normal);
    if (abs(side) <= DIRECTION_EPSILON) {
        return false;
    }

    shadow_dir = side > 0.0 ? edge.positive_shadow_dir : edge.negative_shadow_dir;
    return true;
}

bool diffractionDirection(DiffractionEdge edge, vec3 ray_direction, float distance_to_edge, out vec3 result) {
    // Diffraction is a 2D decision in the cross-section perpendicular to the
    // edge. Keep this decomposition shared by side selection and final
    // reconstruction so the perpendicular projection is computed once.
    vec3 d_parallel = dot(ray_direction, edge.edge_dir) * edge.edge_dir;
    vec3 d_perpendicular = ray_direction - d_parallel;
    float perpendicular_len = length(d_perpendicular);
    if (perpendicular_len <= DIRECTION_EPSILON) {
        return false;
    }

    vec3 d_perpendicular_dir = d_perpendicular / perpendicular_len;
    vec3 shadow_dir;
    if (!chooseShadowDir(edge, d_perpendicular_dir, shadow_dir)) {
        return false;
    }

    // The finite diffraction plane doubles as the bend radius. A hit at the
    // edge bends fully toward the selected shadow boundary; a hit at R still
    // counts as detected but leaves the direction unchanged.
    float bend = clamp(1.0 - distance_to_edge / edge.diffraction_radius, 0.0, 1.0);
    vec3 mixed = mix(d_perpendicular_dir, shadow_dir, bend);
    float mixed_len = length(mixed);
    if (mixed_len <= DIRECTION_EPSILON) {
        return false;
    }

    vec3 bent_perpendicular = mixed / mixed_len;

    // Reattach the original edge-parallel component so diffraction changes only
    // the cross-section direction and does not erase propagation along the edge.
    result = normalize(d_parallel + bent_perpendicular * perpendicular_len);
    return !any(isnan(result)) && !any(isinf(result));
}

void main() {
    // The AABB/plane intersection is a virtual event. The straight child uses
    // the plane hit distance, while the diffracted child will recompute its
    // origin and accumulated distance from the closest physical edge point.
    float hit_distance = gl_HitTEXT;
    float incoming_path_distance = payload.path_distance;
    float hit_path_distance = incoming_path_distance + hit_distance;

    if (payload.path_depth >= pc.max_bounces) {
        return;
    }

    DiffractionEdge edge = diffraction_edges.edges[gl_PrimitiveID];
    float distance_to_edge = diffraction_hit.x;
    float edge_t = diffraction_hit.y;

    vec3 hit_position = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * hit_distance;
    vec3 incoming_direction = normalize(gl_WorldRayDirectionEXT);
    float incoming_energy = payload.ray_energy;
    uint next_depth = payload.path_depth + 1;

    // base_strength is the authored diffraction energy split. Distance affects
    // direction bending only. If diffraction is rejected, the virtual plane must
    // stay transparent and the straight continuation keeps the full energy.
    float diffraction_factor = clamp(edge.base_strength, 0.0, 1.0);
    float continued_energy = incoming_energy;
    float diffracted_energy = 0.0;
    vec3 diffracted_direction;
    if (diffraction_factor > 0.0 && diffractionDirection(edge, incoming_direction, distance_to_edge, diffracted_direction)) {
        diffracted_energy = incoming_energy * diffraction_factor;
        if (diffracted_energy > MIN_RAY_ENERGY) {
            continued_energy = incoming_energy - diffracted_energy;
        } else {
            diffracted_energy = 0.0;
        }
    }

    if (continued_energy > MIN_RAY_ENERGY) {
        // Preserve non-diffracted energy by continuing from the virtual plane hit
        // along the original ray. This keeps the detector plane transparent.
        payload.ray_energy = continued_energy;
        payload.path_distance = hit_path_distance;
        payload.path_depth = next_depth;
        traceRayEXT(
            tlas,
            gl_RayFlagsOpaqueEXT,
            0xff,
            0,
            0,
            0,
            hit_position + incoming_direction * RAY_EPSILON,
            RAY_EPSILON,
            incoming_direction,
            10000.0,
            0
        );
    }

    if (diffracted_energy > MIN_RAY_ENERGY) {
        // The plane hit only detects diffraction. The diffracted child starts from
        // the projected point on the finite edge, so the secondary path behaves like
        // the edge itself is the diffracting source.
        vec3 edge_origin = edge.start + edge.edge_dir * edge_t;
        float edge_path_distance = incoming_path_distance + length(edge_origin - gl_WorldRayOriginEXT);

        // Emit the diffracted child from the physical edge, offset along the new
        // direction to avoid immediately intersecting the same diffraction AABB.
        payload.ray_energy = diffracted_energy;
        payload.path_distance = edge_path_distance;
        payload.path_depth = next_depth;
        traceRayEXT(
            tlas,
            gl_RayFlagsOpaqueEXT,
            0xff,
            0,
            0,
            0,
            edge_origin + diffracted_direction * RAY_EPSILON,
            RAY_EPSILON,
            diffracted_direction,
            10000.0,
            0
        );
    }
}
