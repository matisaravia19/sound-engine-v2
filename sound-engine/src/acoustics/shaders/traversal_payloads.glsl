// Shared by ray generation and every hit/miss stage. Declarations at each
// call site choose outgoing or incoming payload storage and its location.

const uint HIT_NONE = 0;
const uint HIT_SURFACE = 1;
const uint HIT_DIFFRACTION_EDGE = 2;

struct HitPayload {
    // HIT_NONE terminates continuation without reading the remaining fields.
    uint hit_type;
    // Parametric distance along the current world-space ray.
    float t;
    // Edge hits: perpendicular distance to the edge and meters along it.
    vec2 edge_coordinates;
    // Surface hits: world-space normal facing against the incident ray.
    vec3 world_normal;
    // Material index for a surface or edge index for a diffraction proxy.
    uint data_id;
};

struct VisibilityPayload {
    // Visibility returns no geometry and terminates at the first surface.
    uint occluded;
};
