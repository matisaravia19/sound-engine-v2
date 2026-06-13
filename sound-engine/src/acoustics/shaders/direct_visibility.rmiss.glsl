#version 460
#extension GL_EXT_ray_tracing : require

struct AcousticPayload {
    vec4 throughput_distance;
    uvec4 control;
};

layout(location = 0) rayPayloadInEXT AcousticPayload payload;

void main() {
}
