#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_shader_atomic_float : require
#extension GL_GOOGLE_include_directive : require
#define UTD_TIME_SURFACES
#define UTD_BANDS
#include "utd_transport.glsl"
#include "utd_two_edges.glsl"
void main() {
    traceZeroAndOneEdge(false);
    if ((pc.source_surface_paths & 1u) != 0u) traceZeroAndOneEdge(true);
    traceTwoEdges();
    if ((pc.source_surface_paths & 2u) != 0u) {
        traceSurfaceTwoEdges(false);
        traceSurfaceTwoEdges(true);
    }
}
