// Matches the host GpuDiffractionEdge record. Intersection and ray generation
// read the same full record even though they use different fields.

struct DiffractionEdge {
    vec3 start;
    float diffraction_radius;
    vec3 end;
    float _pad_end;
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
