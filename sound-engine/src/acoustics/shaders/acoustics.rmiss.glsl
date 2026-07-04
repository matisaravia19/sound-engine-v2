#version 460
#extension GL_EXT_ray_tracing : require

struct AcousticPayload {
    float ray_energy;
    float path_distance;
    uint path_depth;
};

layout(location = 0) rayPayloadInEXT AcousticPayload payload;

void main() {
}
