#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_shader_atomic_float : require
#extension GL_GOOGLE_include_directive : require
#include "utd_transport.glsl"
void main() { traceZeroAndOneEdge(false); }
