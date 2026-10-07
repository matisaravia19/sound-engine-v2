#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#include "traversal_payloads.glsl"

// Continuation misses end the iterative path. Ray generation initializes the
// payload the same way before traversal; writing it here makes the miss contract
// explicit and independent of any closest-hit data from a previous bounce.

layout(location = 0) rayPayloadInEXT HitPayload payload;

void main() {
    payload.hit_type = HIT_NONE;
}
