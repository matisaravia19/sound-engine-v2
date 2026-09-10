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
// point and drives the normalized radial response. y records distance along the
// finite edge segment so ray generation can advance to the physical edge.
hitAttributeEXT vec2 diffraction_hit;

void main() {
    DiffractionEdge edge = diffraction_edges.edges[gl_PrimitiveID];
    // A disabled edge must not intercept ordinary transport. Skipping the
    // procedural intersection removes it from both next-event evaluation and
    // continuation without requiring a special branch in ray generation.
    if (edge.base_strength <= 0.0 || edge.diffraction_radius <= 0.0) {
        return;
    }

    // The diffraction primitive is represented as a finite plane strip around
    // the authored edge. The plane is virtual: it detects nearby rays, then
    // ray generation projects accepted hits back to the physical edge. The
    // procedural instance is identity-transformed because edge records are
    // already stored in world coordinates.
    float denom = dot(gl_ObjectRayDirectionEXT, edge.positive_plane_normal);
    if (abs(denom) <= DIFFRACTION_PLANE_EPSILON) {
        return;
    }

    // Intersect the infinite plane first. reportIntersectionEXT below turns it
    // into a finite AABB-contained primitive only after the strip tests pass.
    float hit_t = dot(edge.start - gl_ObjectRayOriginEXT, edge.positive_plane_normal) / denom;
    if (hit_t < gl_RayTminEXT || hit_t > gl_RayTmaxEXT) {
        return;
    }

    vec3 hit_position = gl_ObjectRayOriginEXT + gl_ObjectRayDirectionEXT * hit_t;
    // edge_dir is unit length, so edge_t is both the projection parameter and a
    // physical distance in meters from the authored start point.
    float edge_t = dot(hit_position - edge.start, edge.edge_dir);

    // Clamp the virtual strip to the actual edge segment so an authored edge does
    // not diffract rays past its endpoints.
    if (edge_t < 0.0 || edge_t > edge.edge_length) {
        return;
    }

    vec3 closest = edge.start + edge.edge_dir * edge_t;
    // The negative half-strip lies inside the solid wedge (or in the thin
    // screen). It is not part of the exterior sampling support.
    if (dot(hit_position - closest, edge.bisector_dir) < 0.0) {
        return;
    }
    float distance_to_edge = length(hit_position - closest);
    
    // The exterior strip ends where its triangular sampling kernel reaches zero.
    if (distance_to_edge >= edge.diffraction_radius) {
        return;
    }

    // The accepted ray retains its accumulated weight; this coordinate locates
    // its next physical diffraction vertex, not an independent edge source.
    diffraction_hit = vec2(distance_to_edge, edge_t);
    reportIntersectionEXT(hit_t, 0);
}
