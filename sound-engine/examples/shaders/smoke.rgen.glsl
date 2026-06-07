#version 460
#extension GL_EXT_ray_tracing : require

layout(set = 0, binding = 0) uniform accelerationStructureEXT top_level_as;
layout(set = 0, binding = 1) buffer ResultBuffer {
    uint values[];
} result;

layout(location = 0) rayPayloadEXT uint payload;

void main() {
    uint idx = gl_LaunchIDEXT.x;
    vec3 origin = idx == 0 ? vec3(0.0, 0.0, 1.0) : vec3(2.0, 2.0, 1.0);

    payload = 0;
    traceRayEXT(
        top_level_as,
        gl_RayFlagsOpaqueEXT,
        0xff,
        0,
        1,
        0,
        origin,
        0.0,
        vec3(0.0, 0.0, -1.0),
        10.0,
        0
    );
    result.values[idx] = payload;
}
