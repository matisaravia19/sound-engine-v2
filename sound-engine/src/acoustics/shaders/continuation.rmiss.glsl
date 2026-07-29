#version 460
#extension GL_EXT_ray_tracing : require

// Continuation misses end the iterative path. Ray generation initializes the
// payload the same way before traversal; writing it here makes the miss contract
// explicit and independent of any closest-hit data from a previous bounce.

struct HitPayload {
    uint hit_type;
    float t;
    uint instance_id;
    uint primitive_id;
    vec2 barycentrics;
    vec3 world_normal;
    uint data_id;
};

layout(location = 0) rayPayloadInEXT HitPayload payload;

void main() {
    payload.hit_type = 0u;
}
