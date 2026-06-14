#version 460
#extension GL_EXT_ray_tracing : require

struct Contribution {
    float arrival_time_seconds;
    float linear_gain;
    float _pad0;
    float _pad1;
    vec3 incoming_direction;
    float path_order;
};

layout(std430, set = 0, binding = 1) buffer ContributionBuffer {
    uint contribution_count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
    Contribution records[];
} contributions;

struct AcousticPayload {
    float ray_gain;
    float path_distance;
    uint reflection_order;
};

layout(location = 0) rayPayloadInEXT AcousticPayload payload;

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

void main() {
    uint record_index = atomicAdd(contributions.contribution_count, 1);
    if (record_index >= pc.max_contributions) {
        return;
    }

    float distance = payload.path_distance + gl_HitTEXT;
    float ray_gain = payload.ray_gain;

    // Store arrival time and distance-attenuated gain for CPU IR binning.
    contributions.records[record_index].arrival_time_seconds = distance / max(pc.speed_of_sound, 0.001);
    contributions.records[record_index].linear_gain = ray_gain;
    contributions.records[record_index].incoming_direction = normalize(-gl_WorldRayDirectionEXT);
    contributions.records[record_index].path_order = float(payload.reflection_order);
}
