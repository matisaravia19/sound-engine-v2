#version 460
#extension GL_EXT_ray_tracing : require

const float IR_FIXED_POINT_SCALE = 1048576.0;
const uint OUTPUT_CHANNELS_MONO = 0;

struct IrSample {
    int left;
    int right;
};

layout(std430, set = 0, binding = 1) buffer IrBuffer {
    IrSample samples[];
} ir;

struct AcousticPayload {
    float ray_energy;
    float path_distance;
    uint reflection_order;
};

layout(location = 0) rayPayloadInEXT AcousticPayload payload;
hitAttributeEXT vec2 listener_hit;

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
} pc;

int toFixedPoint(float value) {
    float clamped = clamp(value * IR_FIXED_POINT_SCALE, -2147483008.0, 2147483007.0);
    return int(round(clamped));
}

void main() {
    // The listener is only a capture volume; arrival time is measured to
    // the listener center so IR timing is stable across the listener extent.
    float listener_center_distance = length(pc.listener_position - gl_WorldRayOriginEXT);
    float distance = payload.path_distance + listener_center_distance;
    float arrival_time_seconds = distance / max(pc.speed_of_sound, 0.001);
    uint sample_index = uint(round(arrival_time_seconds * float(pc.sample_rate)));
    if (sample_index >= pc.sample_count) {
        return;
    }

    float listener_distance = listener_hit.x;
    float ray_intensity = payload.ray_energy * listener_distance / max(pc.listener_volume, 0.000001);
    vec3 incoming_direction = normalize(-gl_WorldRayDirectionEXT);
    float left_factor = 1.0;
    float right_factor = 1.0;
    if (pc.output_channels != OUTPUT_CHANNELS_MONO) {
        float pan = clamp(dot(incoming_direction, normalize(pc.listener_right)), -1.0, 1.0);
        left_factor = sqrt(0.5 * (1.0 - pan));
        right_factor = sqrt(0.5 * (1.0 + pan));
    }

    // Multiple rays can land on the same IR sample, so accumulation is atomic.
    atomicAdd(ir.samples[sample_index].left, toFixedPoint(ray_intensity * left_factor));
    atomicAdd(ir.samples[sample_index].right, toFixedPoint(ray_intensity * right_factor));
}
