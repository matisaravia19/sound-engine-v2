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
} pc;

void main() {
    vec3 origin = pc.source.xyz;
    vec3 target = pc.listener.xyz;
    float source_gain = pc.source.w;
    float speed_of_sound = pc.listener.w;
    vec3 delta = target - origin;
    float distance = length(delta);
    contributions.header.x = 0;

    if (distance <= 0.001) {
        contributions.header.x = 1;
        contributions.records[0].timing_gain = vec4(0.0, source_gain, 0.0, 0.0);
        contributions.records[0].direction_order = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }

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
        normalize(delta),
        max(distance - 0.002, 0.001),
        0
    );

    if (occluded == 0) {
        contributions.header.x = 1;
        contributions.records[0].timing_gain = vec4(
            distance / max(speed_of_sound, 0.001),
            source_gain / max(distance, 1.0),
            0.0,
            0.0
        );
        contributions.records[0].direction_order = vec4(normalize(-delta), 0.0);
    }
}
