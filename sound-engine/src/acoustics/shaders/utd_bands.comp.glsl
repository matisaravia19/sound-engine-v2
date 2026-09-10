#version 460
layout(local_size_x=256) in;
layout(std430, binding=0) readonly buffer Rays { vec2 values[]; } rays;
layout(std430, binding=1) buffer Scratch { vec2 values[]; } scratch;
layout(std430, binding=2) buffer Combined { vec2 values[]; } combined;
layout(std430, binding=3) writeonly buffer Output { vec2 values[]; } result;
layout(push_constant) uniform Constants {
    uint samples; uint fft_size; uint sample_rate; uint channels;
    uint bands; uint stage; float maximum_frequency;
} pc;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.fft_size) return;
    if (pc.stage == 0u) {
        uint plane = gl_GlobalInvocationID.y;
        scratch.values[plane*pc.fft_size+i] = i < pc.samples ? rays.values[plane*pc.samples+i] : vec2(0.0);
    } else if (pc.stage == 1u) {
        uint positive = min(i, pc.fft_size-i);
        float f = float(positive)*float(pc.sample_rate)/float(pc.fft_size);
        if (f == 0.0 || f > pc.maximum_frequency) {
            combined.values[i] = vec2(0.0);
            return;
        }
        float minimum_frequency = float(pc.sample_rate)/float(pc.fft_size);
        float node = log(f/minimum_frequency)/log(pc.maximum_frequency/minimum_frequency)*float(pc.bands-1u);
        node = clamp(node, 0.0, float(pc.bands-1u));
        uint lo = uint(floor(node));
        uint hi = min(lo+1u, pc.bands-1u);
        // Restore Hermitian symmetry per ear, then pack the two real signals
        // into a single complex inverse transform: H_left + i H_right.
        vec2 packed = vec2(0.0);
        for (uint channel=0u; channel<pc.channels; ++channel) {
            vec2 lower = scratch.values[(lo*pc.channels+channel)*pc.fft_size+positive];
            vec2 upper = scratch.values[(hi*pc.channels+channel)*pc.fft_size+positive];
            vec2 value = mix(lower, upper, fract(node));
            if (i > pc.fft_size/2u) value.y = -value.y;
            if (positive == pc.fft_size/2u) value.y = 0.0;
            if (channel == 1u) value = vec2(-value.y, value.x);
            packed += value;
        }
        combined.values[i] = packed;
    } else if (i < pc.samples) {
        result.values[i] = combined.values[i] + rays.values[pc.bands*pc.channels*pc.samples+i];
    }
}
