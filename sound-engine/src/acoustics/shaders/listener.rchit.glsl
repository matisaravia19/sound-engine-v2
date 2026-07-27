#version 460
#extension GL_EXT_ray_tracing : require

struct AcousticHit {
    // Unit vectors use octahedral encoding plus packSnorm2x16 so each one
    // occupies 4 bytes in the hit buffer.
    uint packed_incoming_direction;
    uint packed_listener_hit_direction;
    float arrival_time_seconds;
    float energy;
};

layout(std430, set = 0, binding = 1) buffer HitBuffer {
    uint count;
    uint overflow_count;
    uvec2 _pad0;
    AcousticHit hits[];
} hit_buffer;

struct AcousticPayload {
    float ray_energy;
    float path_distance;
    uint path_depth;
};

layout(location = 0) rayPayloadInEXT AcousticPayload payload;
// x is the path length through the listener sphere; y is the entry distance
// from the object-space ray origin, used to reconstruct the sphere hit point.
hitAttributeEXT vec2 listener_hit;

vec2 signNotZero(vec2 value) {
    return vec2(value.x >= 0.0 ? 1.0 : -1.0, value.y >= 0.0 ? 1.0 : -1.0);
}

vec2 octEncode(vec3 direction) {
    // Octahedral encoding stores a unit vector as two normalized components
    // with no spherical-coordinate seam at the poles.
    vec3 normalized = normalize(direction);
    normalized /= max(abs(normalized.x) + abs(normalized.y) + abs(normalized.z), 0.000001);
    if (normalized.z < 0.0) {
        normalized.xy = (1.0 - abs(normalized.yx)) * signNotZero(normalized.xy);
    }
    return clamp(normalized.xy, vec2(-1.0), vec2(1.0));
}

layout(push_constant) uniform PushConstants {
    layout(offset = 0) vec3 source_position;
    layout(offset = 12) float source_energy;
    layout(offset = 16) vec3 listener_position;
    layout(offset = 28) float speed_of_sound;
    layout(offset = 32) float listener_radius;
    layout(offset = 36) float listener_volume;
    layout(offset = 44) uint ray_count;
    layout(offset = 48) uint max_bounces;
    layout(offset = 64) vec3 listener_right;
    layout(offset = 76) uint sample_rate;
    layout(offset = 80) uint sample_count;
    layout(offset = 84) uint output_channels;
    layout(offset = 88) uint hit_capacity;
} pc;

void main() {
    // The listener is only a capture volume; arrival time is measured to
    // the listener center so IR timing is stable across the listener extent.
    float listener_center_distance = length(pc.listener_position - gl_WorldRayOriginEXT);
    float distance = payload.path_distance + listener_center_distance;
    float arrival_time_seconds = distance / max(pc.speed_of_sound, 0.001);

    float listener_distance = listener_hit.x;
    float ray_intensity = payload.ray_energy * listener_distance / max(pc.listener_volume, 0.000001);
    vec3 incoming_direction = normalize(-gl_WorldRayDirectionEXT);
    // The listener BLAS is translated to the listener position, so object-space
    // hit position is also the direction from the listener center.
    vec3 listener_hit_direction = normalize(gl_ObjectRayOriginEXT + gl_ObjectRayDirectionEXT * listener_hit.y);

    uint index = atomicAdd(hit_buffer.count, 1);
    if (index >= pc.hit_capacity) {
        atomicAdd(hit_buffer.overflow_count, 1);
        return;
    }

    // Keep this layout in sync with GpuAcousticHit in pipeline.rs.
    hit_buffer.hits[index].packed_incoming_direction = packSnorm2x16(octEncode(incoming_direction));
    hit_buffer.hits[index].packed_listener_hit_direction = packSnorm2x16(octEncode(listener_hit_direction));
    hit_buffer.hits[index].arrival_time_seconds = arrival_time_seconds;
    hit_buffer.hits[index].energy = ray_intensity;
}
