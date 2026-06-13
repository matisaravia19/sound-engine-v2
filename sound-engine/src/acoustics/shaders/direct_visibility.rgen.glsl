#version 460
#extension GL_EXT_ray_tracing : require

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;

struct Contribution {
    vec4 timing_gain;
    vec4 direction_order;
};

layout(std430, set = 0, binding = 1) buffer ContributionBuffer {
    uvec4 header;
    Contribution records[];
} contributions;

layout(location = 0) rayPayloadEXT uint occluded;

layout(push_constant) uniform PushConstants {
    vec4 source;
    vec4 listener;
    vec4 listener_half_extent;
} pc;

void recordContribution(float distance, float source_gain, float speed_of_sound, vec3 incoming_direction) {
    contributions.header.x = 1;
    contributions.records[0].timing_gain = vec4(
        distance / max(speed_of_sound, 0.001),
        source_gain / max(distance, 1.0),
        0.0,
        0.0
    );
    contributions.records[0].direction_order = vec4(incoming_direction, 0.0);
}

void main() {
    vec3 origin = pc.source.xyz;
    vec3 target = pc.listener.xyz;
    vec3 listener_half_extent = pc.listener_half_extent.xyz;
    float source_gain = pc.source.w;
    float speed_of_sound = pc.listener.w;
    vec3 delta = target - origin;
    float center_distance = length(delta);
    contributions.header.x = 0;

    if (center_distance <= 0.001) {
        recordContribution(0.0, source_gain, speed_of_sound, vec3(0.0));
        return;
    }

    vec3 ray_direction = normalize(delta);
    float listener_extent_radius = length(listener_half_extent);

    occluded = 0;
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
        center_distance + listener_extent_radius + 0.001,
        0
    );
}
