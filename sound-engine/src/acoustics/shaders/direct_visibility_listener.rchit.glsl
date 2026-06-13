#version 460
#extension GL_EXT_ray_tracing : require

const uint MAX_CONTRIBUTIONS = 4096;

struct Contribution {
    vec4 timing_gain;
    vec4 direction_order;
};

layout(std430, set = 0, binding = 1) buffer ContributionBuffer {
    uvec4 header;
    Contribution records[];
} contributions;

struct AcousticPayload {
    vec4 throughput_distance;
    uvec4 control;
};

layout(location = 0) rayPayloadInEXT AcousticPayload payload;

layout(push_constant) uniform PushConstants {
    vec4 source;
    vec4 listener;
    vec4 listener_half_extent;
    uvec4 ray_config;
} pc;

void main() {
    uint record_index = atomicAdd(contributions.header.x, 1);
    if (record_index >= MAX_CONTRIBUTIONS) {
        return;
    }

    float distance = payload.throughput_distance.y + gl_HitTEXT;
    float ray_gain = payload.throughput_distance.x;
    float speed_of_sound = pc.listener.w;

    contributions.records[record_index].timing_gain = vec4(
        distance / max(speed_of_sound, 0.001),
        ray_gain / max(distance, 1.0),
        0.0,
        0.0
    );
    contributions.records[record_index].direction_order = vec4(normalize(-gl_WorldRayDirectionEXT), float(payload.control.x));
}
