#version 460
#extension GL_EXT_ray_tracing : require

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;

struct AcousticPayload {
    float ray_gain;
    float path_distance;
    uint reflection_order;
};

layout(location = 0) rayPayloadEXT AcousticPayload payload;

layout(push_constant) uniform PushConstants {
    layout(offset = 0) vec3 source_position;
    layout(offset = 12) float source_gain;
    layout(offset = 16) vec3 listener_position;
    layout(offset = 28) float speed_of_sound;
    layout(offset = 32) vec3 listener_half_extent;
    layout(offset = 44) uint ray_count;
    layout(offset = 48) uint max_bounces;
    layout(offset = 52) uint max_contributions;
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
    vec3 origin = pc.source_position;
    uint ray_count = max(pc.ray_count, 1);
    vec3 ray_direction = fibonacciSphereDirection(gl_LaunchIDEXT.x, ray_count);

    // Split the source gain evenly across all sampled directions.
    payload.ray_gain = pc.source_gain / float(ray_count);
    payload.path_distance = 0.0;
    payload.reflection_order = 0;
    traceRayEXT(
        tlas,
        gl_RayFlagsOpaqueEXT,
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
