#version 460
#extension GL_EXT_ray_tracing : require

// The procedural intersection stage has already found the virtual strip hit.
// This closest-hit stage labels the event and forwards its radial distance and
// finite-edge coordinate. Ray generation verifies and advances the incoming
// segment to the physical edge before updating the ray's pressure weight.

struct HitPayload {
    uint hit_type;
    float t;
    uint instance_id;
    uint primitive_id;
    vec2 barycentrics;
    vec3 world_normal;
    uint data_id;
};

struct DiffractionEdge {
    vec3 start;
    float diffraction_radius;
    vec3 end;
    float base_strength;
    vec3 edge_dir;
    float edge_length;
    vec3 bisector_dir;
    float edge_angle_radians;
    vec3 positive_plane_normal;
    float _pad0;
    vec3 positive_shadow_dir;
    float _pad1;
    vec3 negative_shadow_dir;
    float solid_wedge_cos;
};

layout(std430, set = 0, binding = 4) readonly buffer DiffractionEdgeBuffer {
    DiffractionEdge edges[];
} diffraction_edges;

layout(location = 0) rayPayloadInEXT HitPayload payload;
hitAttributeEXT vec2 diffraction_hit;

void main() {
    // One identity-transformed procedural BLAS contains all authored edges, so
    // gl_PrimitiveID is also the stable index into the edge storage buffer.
    DiffractionEdge edge = diffraction_edges.edges[gl_PrimitiveID];
    payload.hit_type = 2u;
    payload.t = gl_HitTEXT;
    payload.instance_id = gl_InstanceCustomIndexEXT;
    payload.primitive_id = gl_PrimitiveID;
    // x = perpendicular distance from proxy hit to edge, y = meters from start
    // along the finite edge. These are not triangle barycentric coordinates.
    payload.barycentrics = diffraction_hit;
    payload.world_normal = edge.positive_plane_normal;
    payload.data_id = gl_PrimitiveID;
}
