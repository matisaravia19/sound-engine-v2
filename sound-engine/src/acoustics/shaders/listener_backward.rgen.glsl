#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_shader_atomic_float : require
#extension GL_GOOGLE_include_directive : require
#include "listener_transport.glsl"
#include "listener_path.glsl"

// Current branch policy: keep the ordinary suffix once, then the diffracted
// suffix. Both use tracePathStep; the prefix, hit and RNG are preserved.
// A future single-continuation estimator belongs at this policy boundary and
// in its local response/PDF sampler, not in a second traversal implementation.
void traceListenerBranches(ListenerPath path, float normalization) {
    uint walk_mode = WALK_TO_FIRST_EDGE;
    HitPayload pending_hit;
    pending_hit.hit_type = HIT_NONE;

    ListenerPath diffracted_path;
    HitPayload diffraction_hit;

    // One iterative loop processes every interaction, with at most one saved
    // branch. Neither local samplers nor closest-hit shaders start another walk.
    while (true) {
        uint outcome = tracePathStep(path, walk_mode, normalization, pending_hit);
        if (outcome == PATH_CONTINUED) {
            continue;
        }

        if (outcome == PATH_REACHED_FIRST_EDGE) {
            // Preserve the prefix and RNG before the ordinary walk changes them.
            diffracted_path = path;
            diffraction_hit = pending_hit;
            pending_hit.hit_type = HIT_NONE;
            walk_mode = WALK_SURFACES_ONLY;
        } else if (walk_mode == WALK_SURFACES_ONLY) {
            path = diffracted_path;
            pending_hit = diffraction_hit;
            walk_mode = WALK_WITH_DIFFRACTION;
        } else {
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
    path.random_state = hashState(ray_index ^ pc.sampling_seed ^ 0xa511e9b3u);
    path.interactions = 0u;
    path.diffraction_events = 0u;

    // Keep the estimator's spherical measure separate from local path weights.
    float normalization = 4.0 * PI / float(pc.listener_ray_count);
    traceListenerBranches(path, normalization);
}
