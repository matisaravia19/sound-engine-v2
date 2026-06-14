#version 460
#extension GL_EXT_ray_tracing : require

struct AcousticPayload {
    float ray_gain;
    float path_distance;
    uint reflection_order;
};

layout(location = 0) rayPayloadInEXT AcousticPayload payload;

void main() {
}
