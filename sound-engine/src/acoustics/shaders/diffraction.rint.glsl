#version 460
#extension GL_EXT_ray_tracing : require

const float DIFFRACTION_PLANE_EPSILON = 0.00001;

// Must stay layout-compatible with the closest-hit shader and the CPU
// GpuDiffractionEdge record. The intersection stage only needs the geometric
// fields, but it declares the full record because both stages read the same
// storage buffer.
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

// Passed to closest-hit. x is the virtual plane hit distance to the closest edge
// point, which drives bend falloff. y is distance along the finite edge segment,
// which lets closest-hit launch the diffracted child from the physical edge.
hitAttributeEXT vec2 diffraction_hit;

void main() {
    DiffractionEdge edge = diffraction_edges.edges[gl_PrimitiveID];

    // The diffraction primitive is represented as a finite plane strip around
    // the authored edge. The plane is virtual: it detects nearby rays, then
    // closest-hit projects accepted hits back to the physical edge.
    float denom = dot(gl_ObjectRayDirectionEXT, edge.positive_plane_normal);
    if (abs(denom) <= DIFFRACTION_PLANE_EPSILON) {
        return;
    }

    float hit_t = dot(edge.start - gl_ObjectRayOriginEXT, edge.positive_plane_normal) / denom;
    if (hit_t < gl_RayTminEXT || hit_t > gl_RayTmaxEXT) {
        return;
    }

    vec3 hit_position = gl_ObjectRayOriginEXT + gl_ObjectRayDirectionEXT * hit_t;
    float edge_t = dot(hit_position - edge.start, edge.edge_dir);

    // Clamp the virtual strip to the actual edge segment so an authored edge does
    // not diffract rays past its endpoints.
    if (edge_t < 0.0 || edge_t > edge.edge_length) {
        return;
    }

    vec3 closest = edge.start + edge.edge_dir * edge_t;
    float distance_to_edge = length(hit_position - closest);
    
    // The radius is both the strip half-width and the zero-bend boundary used by
    // closest-hit's linear falloff, so hits outside it are not meaningful.
    if (distance_to_edge > edge.diffraction_radius) {
        return;
    }

    // Closest-hit needs both values because H, the plane hit, and O, the edge
    // origin of the diffracted ray, intentionally differ.
    diffraction_hit = vec2(distance_to_edge, edge_t);
    reportIntersectionEXT(hit_t, 0);
}
