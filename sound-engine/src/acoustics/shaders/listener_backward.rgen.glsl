#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_shader_atomic_float : require
#extension GL_GOOGLE_include_directive : require
#include "listener_transport.glsl"

// Two edges retain the double-edge benchmark paths while bounding the least
// reliable higher-order approximation. Surface bounces keep their own budget.
const uint MAX_DIFFRACTION_EVENTS = 2u;

// A proxy is sampling geometry, not an occluder. At the first proxy, retain
// the ray's ordinary surface continuation as well as its diffraction branch.
// The prefix has already contributed; trace only the remaining surface suffix.
// This adds at most one bounded surface walk per launch, with no recursive
// tracing, ray queue, extra frequency state, or change to surface pressure.
void traceSurfaceSuffix(
    uint first_bounce, vec3 origin, vec3 path_vertex, vec3 direction,
    float path_throughput, float traveled_distance, uint random_state,
    float normalization, vec3 initial_direction
) {
    for (uint bounce = first_bounce; bounce < pc.max_bounces; ++bounce) {
        if (!finiteFloat(path_throughput)
            || abs(path_throughput) < pc.minimum_throughput
            || !finiteVector(origin) || !finiteVector(direction)
            || traveled_distance >= pc.maximum_path_length) break;

        hit_payload.hit_type = HIT_NONE;
        traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, SURFACE_GEOMETRY_MASK,
                    0, 0, 0, origin, RAY_EPSILON, direction,
                    max(pc.maximum_path_length - traveled_distance, RAY_EPSILON), 0);
        if (hit_payload.hit_type != HIT_SURFACE
            || !finiteFloat(hit_payload.t) || hit_payload.t < 0.0) break;
        vec3 hit = origin + direction * hit_payload.t;
        float distance = traveled_distance + length(hit - path_vertex);
        if (distance > pc.maximum_path_length) break;
        vec3 normal = normalizeOrZero(hit_payload.world_normal);
        SceneMaterial material = scene_materials.materials[hit_payload.data_id];
        connectSurfaceToSource(hit, direction, normal, material, distance,
                               path_throughput, normalization, initial_direction);
        if (bounce + 1u >= pc.max_bounces) break;
        vec3 continuation;
        if (!sampleSurfaceContinuation(material, direction, normal, random_state,
                                       path_throughput, continuation)) break;
        origin = hit + continuation * RAY_EPSILON;
        path_vertex = hit;
        direction = continuation;
        traveled_distance = distance;
    }
}

void main() {
    uint ray_index = pc.ray_offset + gl_LaunchIDEXT.x;
    // Direct sound is a deterministic visibility query, so exactly one
    // invocation performs it regardless of listener_ray_count.
    if (ray_index == 0u) {
        evaluateDirectPath();
    }
    // max_bounces zero disables all vertex paths but intentionally leaves the
    // direct test above active. Zero source amplitude has nothing left to trace.
    if (pc.max_bounces == 0u || pc.source_amplitude == 0.0) {
        return;
    }

    // Each invocation receives an independent deterministic continuation stream.
    // The Fibonacci rotation depends only on the seed, preserving even coverage.
    uint random_state = hashState(ray_index ^ pc.sampling_seed ^ 0xa511e9b3u);
    uint azimuth_rotation = hashState(pc.sampling_seed);
    vec3 initial_direction = fibonacciSphereDirection(ray_index, pc.listener_ray_count, azimuth_rotation);
    vec3 origin = pc.listener_position;
    // Keep acoustic vertices separate from epsilon-offset traversal origins.
    // Numerical self-intersection offsets must not shorten propagation delays.
    vec3 path_vertex = origin;
    vec3 direction = initial_direction;
    // Throughput contains only accumulated local transport. The 4*PI/N
    // estimator weight stays separate and is applied only to completed paths.
    float path_throughput = 1.0;
    float traveled_distance = 0.0;
    float monte_carlo_normalization = 4.0 * PI / float(pc.listener_ray_count);
    uint diffraction_events = 0u;

    for (uint bounce = 0u; bounce < pc.max_bounces; ++bounce) {
        // This is deterministic truncation only. There is no Russian roulette;
        // identical inputs always execute the same stopping policy.
        if (!finiteFloat(path_throughput)
            || abs(path_throughput) < pc.minimum_throughput
            || !finiteVector(origin)
            || !finiteVector(direction)
            || traveled_distance >= pc.maximum_path_length) {
            break;
        }

        hit_payload.hit_type = HIT_NONE;
        hit_payload.t = 0.0;
        // Surface instances use SBT offset zero and diffraction instances use
        // offset two, routing hits to triangle and procedural records.
        traceRayEXT(
            tlas,
            gl_RayFlagsOpaqueEXT,
            diffraction_events < MAX_DIFFRACTION_EVENTS
                ? CONTINUATION_GEOMETRY_MASK : SURFACE_GEOMETRY_MASK,
            0,
            0,
            0,
            origin,
            RAY_EPSILON,
            direction,
            max(pc.maximum_path_length - traveled_distance, RAY_EPSILON),
            0
        );
        // A miss has no environment contribution in the current point-source
        // model, so it terminates the path immediately.
        if (hit_payload.hit_type == HIT_NONE || !finiteFloat(hit_payload.t) || hit_payload.t < 0.0) {
            break;
        }

        vec3 hit_position = origin + direction * hit_payload.t;
        if (hit_payload.hit_type == HIT_SURFACE) {
            // Step 1: account for distance to the physical surface.
            // Step 2: explicitly connect that vertex to the source.
            // Step 3: if another bounce is allowed, sample one continuation.
            float hit_path_distance = traveled_distance + length(hit_position - path_vertex);
            if (hit_path_distance > pc.maximum_path_length) {
                break;
            }

            vec3 normal = normalizeOrZero(hit_payload.world_normal);
            SceneMaterial material = scene_materials.materials[hit_payload.data_id];
            connectSurfaceToSource(
                hit_position,
                direction,
                normal,
                material,
                hit_path_distance,
                path_throughput,
                monte_carlo_normalization,
                initial_direction
            );
            // Source connection belongs to this accepted vertex and is therefore
            // evaluated before the bounce-limit continuation check.
            if (bounce + 1u >= pc.max_bounces) {
                break;
            }

            vec3 continuation_direction;
            if (!sampleSurfaceContinuation(
                material,
                direction,
                normal,
                random_state,
                path_throughput,
                continuation_direction
            )) {
                break;
            }
            // Commit the new path state only after every sample and finite-state
            // check succeeds. Failed samples terminate without partial updates.
            origin = hit_position + continuation_direction * RAY_EPSILON;
            path_vertex = hit_position;
            direction = continuation_direction;
            traveled_distance = hit_path_distance;
        } else if (hit_payload.hit_type == HIT_DIFFRACTION_EDGE) {
            if (diffraction_events == 0u) {
                // Preserve the zero-edge path once, before changing its vertex,
                // RNG, direction or pressure. Later edges belong exclusively to
                // the approximate diffracted branch and do not fork again.
                HitPayload proxy_hit = hit_payload;
                traceSurfaceSuffix(bounce, origin, path_vertex, direction,
                    path_throughput, traveled_distance, random_state,
                    monte_carlo_normalization, initial_direction);
                hit_payload = proxy_hit;
            }
            ++diffraction_events;
            // The proxy discovers an interaction; the ray bends at the physical
            // edge. Re-trace the adjusted incoming leg so relocation cannot
            // transport pressure through a blocker missed by the proxy ray.
            DiffractionEdge edge = diffraction_edges.edges[hit_payload.data_id];
            float distance_to_edge = hit_payload.barycentrics.x;
            float proxy_leg_length = length(hit_position - path_vertex);
            hit_position = edge.start + hit_payload.barycentrics.y * edge.edge_dir;
            vec3 edge_leg = hit_position - path_vertex;
            float edge_leg_length = length(edge_leg);
            if (edge_leg_length <= RAY_EPSILON) {
                break;
            }
            direction = edge_leg / edge_leg_length;
            float edge_path_distance = traveled_distance + edge_leg_length;
            if (edge_path_distance > pc.maximum_path_length
                || !traceVisibility(path_vertex, direction, edge_leg_length)) {
                break;
            }
            // Projection changes the uniform listener-ray sampling density. On this
            // plane dOmega/dA = |n dot (P-L)| / |P-L|^3; its numerator is
            // unchanged by the in-plane relocation from proxy P to edge E.
            // Apply only to the initial uniform launch: later surface/edge
            // directional PDFs would also need their own relocation correction.
            if (bounce == 0u) {
                float density_ratio = proxy_leg_length / edge_leg_length;
                path_throughput *= density_ratio * density_ratio * density_ratio;
            }
            path_throughput *= sqrt(clamp(edge.base_strength, 0.0, 1.0));
            if (bounce == 0u) {
                initial_direction = direction;
            }

            connectDiffractionToSource(
                edge,
                hit_position,
                direction,
                distance_to_edge,
                edge_path_distance,
                path_throughput,
                monte_carlo_normalization,
                initial_direction
            );
            // As for surfaces, this vertex's source connection remains valid at
            // the final permitted bounce; only continuation is suppressed.
            if (bounce + 1u >= pc.max_bounces) {
                break;
            }

            vec3 continuation_origin;
            vec3 continuation_direction;
            float continuation_path_distance;
            if (!sampleDiffractionContinuation(
                edge,
                hit_position,
                direction,
                traveled_distance,
                edge_leg_length,
                random_state,
                path_throughput,
                continuation_origin,
                continuation_direction,
                continuation_path_distance
            )) {
                break;
            }
            // Continue from the ray-traced interaction in the sampled shadow
            // diffraction direction.
            origin = continuation_origin;
            path_vertex = hit_position;
            direction = continuation_direction;
            traveled_distance = continuation_path_distance;
        } else {
            // Defensive termination for payload values unknown to this shader.
            break;
        }
    }
}
