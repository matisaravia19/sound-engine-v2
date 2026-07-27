#version 460
#extension GL_EXT_shader_atomic_float : require

const uint OUTPUT_CHANNELS_MONO = 0;

struct AcousticHit {
    uint packed_incoming_direction;
    uint packed_listener_hit_direction;
    float arrival_time_seconds;
    float energy;
};

struct IrSample {
    float left;
    float right;
};

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer HitBuffer {
    uint count;
    uint overflow_count;
    uvec2 _pad0;
    AcousticHit hits[];
} hit_buffer;

layout(std430, set = 0, binding = 1) buffer IrBuffer {
    IrSample samples[];
} ir;

layout(push_constant) uniform PushConstants {
    layout(offset = 0) vec3 listener_right;
    layout(offset = 12) uint sample_rate;
    layout(offset = 16) uint sample_count;
    layout(offset = 20) uint output_channels;
    layout(offset = 24) uint hit_capacity;
} pc;

vec2 signNotZero(vec2 value) {
    return vec2(value.x >= 0.0 ? 1.0 : -1.0, value.y >= 0.0 ? 1.0 : -1.0);
}

vec3 octDecode(vec2 encoded) {
    vec3 direction = vec3(encoded.xy, 1.0 - abs(encoded.x) - abs(encoded.y));
    if (direction.z < 0.0) {
        direction.xy = (1.0 - abs(direction.yx)) * signNotZero(direction.xy);
    }
    return normalize(direction);
}

vec3 normalizeOrZero(vec3 value) {
    float length_squared = dot(value, value);
    if (length_squared <= 0.0) {
        return vec3(0.0);
    }
    return value * inversesqrt(length_squared);
}

void main() {
    uint index = gl_GlobalInvocationID.x;
    uint stored_count = min(hit_buffer.count, pc.hit_capacity);
    if (index >= stored_count) {
        return;
    }

    AcousticHit hit = hit_buffer.hits[index];
    if (isinf(hit.arrival_time_seconds) || isnan(hit.arrival_time_seconds) || hit.arrival_time_seconds < 0.0) {
        return;
    }
    if (isinf(hit.energy) || isnan(hit.energy)) {
        return;
    }

    uint sample_index = uint(round(hit.arrival_time_seconds * float(pc.sample_rate)));
    if (sample_index >= pc.sample_count) {
        return;
    }

    vec2 packed_direction = unpackSnorm2x16(hit.packed_incoming_direction);
    vec3 incoming_direction = octDecode(packed_direction);
    float left_factor = 1.0;
    float right_factor = 0.0;
    if (pc.output_channels != OUTPUT_CHANNELS_MONO) {
        float pan = clamp(dot(incoming_direction, normalizeOrZero(pc.listener_right)), -1.0, 1.0);
        left_factor = sqrt(0.5 * (1.0 - pan));
        right_factor = sqrt(0.5 * (1.0 + pan));
    }

    // Many hits can land on the same IR sample, so final IR writes use
    // VK_EXT_shader_atomic_float buffer float32 atomic add.
    atomicAdd(ir.samples[sample_index].left, hit.energy * left_factor);
    atomicAdd(ir.samples[sample_index].right, hit.energy * right_factor);
}
