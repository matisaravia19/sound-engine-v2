#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#include "traversal_payloads.glsl"

// Reaching this miss shader proves that no masked physical surface was found
// before the source distance supplied as the ray's tmax.

layout(location = 1) rayPayloadInEXT VisibilityPayload payload;

void main() {
    payload.occluded = 0u;
}
