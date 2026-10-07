const float PI = 3.14159265358979323846;
const float TWO_PI = 6.28318530717958647692;
// Origins are moved off accepted geometry to avoid immediately re-hitting the
// same primitive. This is a geometric tolerance, not an acoustic distance.
const float RAY_EPSILON = 0.001;
const float DIRECTION_EPSILON = 0.00001;
const uint HIT_NONE = 0;
const uint HIT_SURFACE = 1;
const uint HIT_DIFFRACTION_EDGE = 2;
const uint OUTPUT_CHANNELS_MONO = 0;
const uint OUTPUT_CHANNELS_STEREO = 1;
const uint OUTPUT_CHANNELS_BINAURAL = 2;
// Visibility rays see physical surfaces only. Continuation rays additionally
// see virtual diffraction strips, whose instances use the disjoint 0x02 mask.
const uint SURFACE_GEOMETRY_MASK = 0x01;
const uint CONTINUATION_GEOMETRY_MASK = 0x03;

// -----------------------------------------------------------------------------
// Scene inputs and traversal payloads
// -----------------------------------------------------------------------------

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;

struct IrSample {
    float left;
    float right;
};

// Float storage-buffer atomics are a required device feature. Every completed
// path can therefore deposit pressure without an intermediate contribution
// list, fixed-point encoding, or a later hit-to-IR compute pass.
layout(std430, set = 0, binding = 1) buffer IrBuffer {
    IrSample samples[];
} ir;

struct SceneMaterial {
    float specular_pressure;
    float rough_pressure;
    float roughness;
};

layout(std430, set = 0, binding = 3) readonly buffer SceneMaterialBuffer {
    SceneMaterial materials[];
} scene_materials;

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

struct HitPayload {
    // HIT_NONE is also the continuation-miss result.
    uint hit_type;
    // Parametric distance along the current world-space ray.
    float t;
    uint instance_id;
    uint primitive_id;
    // Triangle barycentrics for surfaces; (distance_to_edge, edge_t) for edges.
    vec2 barycentrics;
    vec3 world_normal;
    // Material index for a surface or edge index for a diffraction proxy.
    uint data_id;
};

struct VisibilityPayload {
    // The visibility hit group only changes this flag; it returns no geometry.
    uint occluded;
};

// Separate payload locations prevent the compact visibility query from
// clobbering the continuation result while nested traceRayEXT calls are issued
// sequentially by the same ray-generation invocation.
layout(location = 0) rayPayloadEXT HitPayload hit_payload;
layout(location = 1) rayPayloadEXT VisibilityPayload visibility_payload;

// Explicit offsets mirror ListenerTracePushConstants on the host. vec3 values
// occupy twelve bytes here, so relying on implicit layout would be error-prone.
layout(push_constant) uniform PushConstants {
    layout(offset = 0) vec3 source_position;
    layout(offset = 12) float source_amplitude;
    layout(offset = 16) vec3 listener_position;
    layout(offset = 28) float speed_of_sound;
    layout(offset = 32) vec3 listener_right;
    layout(offset = 44) float maximum_path_length;
    layout(offset = 48) uint listener_ray_count;
    layout(offset = 52) uint max_bounces;
    layout(offset = 56) uint sample_rate;
    layout(offset = 60) uint sample_count;
    layout(offset = 64) uint output_channels;
    layout(offset = 68) uint sampling_seed;
    layout(offset = 72) float minimum_throughput;
    layout(offset = 76) float head_radius_meters;
    layout(offset = 80) float max_ild_db;
} pc;
