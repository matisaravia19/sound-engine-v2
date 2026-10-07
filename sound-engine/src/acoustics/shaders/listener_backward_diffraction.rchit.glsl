#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#include "traversal_payloads.glsl"

// The procedural intersection stage has already found the virtual strip hit.
// This closest-hit stage labels the event and forwards its radial distance and
// finite-edge coordinate. Ray generation verifies and advances the incoming
// segment to the physical edge before updating the ray's pressure weight.

layout(location = 0) rayPayloadInEXT HitPayload payload;
hitAttributeEXT vec2 diffraction_hit;

void main() {
    payload.hit_type = HIT_DIFFRACTION_EDGE;
    payload.t = gl_HitTEXT;
    payload.edge_coordinates = diffraction_hit;
    payload.world_normal = vec3(0.0);
    // One procedural BLAS contains every edge in storage-buffer order.
    payload.data_id = gl_PrimitiveID;
}
