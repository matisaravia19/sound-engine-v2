#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_shader_atomic_float : require
#extension GL_GOOGLE_include_directive : require
#include "listener_transport.glsl"
#include "listener_path.glsl"

// Every accepted surface or proxy consumes one interaction. The local step
// selects one continuation; no suffix is saved or traced as a second branch.
void traceListenerPath(ListenerPath path, float normalization) {
    while (true) {
        if (!tracePathStep(path, normalization)) {
            return;
        }
    }
}

void main() {
    uint ray_index = gl_LaunchIDEXT.x;

    // Direct visibility contributes once, independently of the ray budget.
    if (ray_index == 0u) {
        evaluateDirectPath();
    }

    if (pc.max_bounces == 0u || pc.source_amplitude == 0.0) {
        return;
    }

    // Seeded rotation preserves the existing even spherical launch coverage;
    // continuations retain their independent deterministic random streams.
    uint azimuth_rotation = hashState(pc.sampling_seed);
    vec3 direction = fibonacciSphereDirection(
        ray_index,
        pc.listener_ray_count,
        azimuth_rotation
    );

    ListenerPath path;
    path.origin = pc.listener_position;
    path.vertex = pc.listener_position;
    path.direction = direction;
    path.listener_direction = direction;
    path.pressure_weight = 1.0;
    path.distance = 0.0;
    path.spreading_origin_distance = 0.0;
    path.random_state = hashState(ray_index ^ pc.sampling_seed ^ 0xa511e9b3u);
    path.interactions = 0u;
    path.has_scattered = false;
    path.pending_diffraction = false;
    path.pending_edge_id = 0u;
    path.pending_incoming_direction = vec3(0.0);
    path.initial_mirror = false;
    path.mirror_normal = vec3(0.0);
    path.mirror_material = 0u;

    // Keep the estimator's spherical measure separate from local path weights.
    float normalization = 4.0 * PI / float(pc.listener_ray_count);
    traceListenerPath(path, normalization);
}
