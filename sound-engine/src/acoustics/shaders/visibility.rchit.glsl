#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#include "traversal_payloads.glsl"

// Visibility rays use a separate one-word payload and terminate on the first
// physical surface. Diffraction proxies are excluded by the ray mask, so they
// can redirect acoustic paths without behaving like solid occluders.

layout(location = 1) rayPayloadInEXT VisibilityPayload payload;

void main() {
    // Ray generation initialized this to zero immediately before traceRayEXT.
    payload.occluded = 1u;
}
