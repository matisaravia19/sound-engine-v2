#version 460
#extension GL_EXT_ray_tracing : require

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;

struct AcousticPayload {
    vec4 throughput_distance;
    uvec4 control;
};

layout(location = 0) rayPayloadEXT AcousticPayload payload;

layout(push_constant) uniform PushConstants {
    vec4 source;
    vec4 listener;
    vec4 listener_half_extent;
    uvec4 ray_config;
} pc;

vec3 fibonacciSphereDirection(uint ray_index, uint ray_count) {
    float i = float(ray_index);
    float n = float(max(ray_count, 1));
    float z = 1.0 - 2.0 * ((i + 0.5) / n);
    float radius = sqrt(max(0.0, 1.0 - z * z));
    float phi = i * 2.39996322972865332;
    return vec3(cos(phi) * radius, sin(phi) * radius, z);
}

void main() {
    vec3 origin = pc.source.xyz;
    uint ray_count = max(pc.ray_config.x, 1);
    vec3 ray_direction = fibonacciSphereDirection(gl_LaunchIDEXT.x, ray_count);

    payload.throughput_distance = vec4(pc.source.w / float(ray_count), 0.0, 0.0, 0.0);
    payload.control = uvec4(0, 0, 0, 0);
    traceRayEXT(
        tlas,
        gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT,
        0xff,
        0,
        0,
        0,
        origin,
        0.001,
        ray_direction,
        10000.0,
        0
    );
}
