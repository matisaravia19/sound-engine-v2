#version 460
#extension GL_EXT_ray_tracing : require

const uint MAX_CONTRIBUTIONS = 64;

struct Contribution {
    vec4 timing_gain;
    vec4 direction_order;
};

layout(std430, set = 0, binding = 1) buffer ContributionBuffer {
    uvec4 header;
    Contribution records[];
} contributions;

layout(location = 0) rayPayloadInEXT uint occluded;

layout(push_constant) uniform PushConstants {
    vec4 source;
    vec4 listener;
    vec4 listener_half_extent;
} pc;

void main() {
    uint record_index = atomicAdd(contributions.header.x, 1);
    if (record_index >= MAX_CONTRIBUTIONS) {
        return;
    }

    float distance = gl_HitTEXT;
    float source_gain = pc.source.w;
    float speed_of_sound = pc.listener.w;

    contributions.records[record_index].timing_gain = vec4(
        distance / max(speed_of_sound, 0.001),
        source_gain / max(distance, 1.0),
        0.0,
        0.0
    );
    contributions.records[record_index].direction_order = vec4(normalize(-gl_WorldRayDirectionEXT), 0.0);
    occluded = 0;
}
