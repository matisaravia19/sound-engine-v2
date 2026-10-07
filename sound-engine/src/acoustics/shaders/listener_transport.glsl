// Transport helpers issue only depth-one traversal calls. The iterative path
// loop and branch policy are defined in listener_path.glsl and ray generation.
// Miss and closest-hit stages return compact data; they never continue a path
// or write the IR. These helpers evaluate local responses and sampling weights;
// the caller owns accumulated path state and termination.
//
// Direction convention:
// - `direction` points away from the listener along the backward-traced path.
// - `incoming_direction` is that same listener-to-vertex direction at a hit.
// - `source_direction` points from the current vertex toward the source.
// - `initial_direction` is retained for listener-space panning even after bends.
//
// Each edge carries its dimensionless incident capture measure K*a/cos(i).
// Subsequent spreading starts at that edge: an edge-source connection uses 1/b,
// while a surface connection uses the unfolded suffix length. Acoustic delay
// always uses the full path. Paths without diffraction keep their original
// 1/total_length spreading. Initial spherical normalization is applied once.
// Mixed surface/edge paths retain the finite-lobe surface approximation.

#include "listener_interface.glsl"
#include "listener_sampling.glsl"

// Returns true when the open segment from `origin` to the source is clear.
// The caller is responsible for offsetting origin onto the intended side of the
// current surface or edge.
bool traceVisibility(vec3 origin, vec3 direction, float distance_to_source) {
    if (distance_to_source <= 2.0 * RAY_EPSILON) {
        return true;
    }
    visibility_payload.occluded = 0u;
    // SBT offset 1 selects the surface-visibility hit record. Stride zero is
    // intentional: diffraction instances are excluded by the surface-only mask.
    // Miss index 1 selects visibility.rmiss.glsl.
    // tmin skips the offset origin's immediate neighborhood. Reducing tmax by
    // the same tolerance prevents geometry at the source endpoint from
    // incorrectly classifying the connection as blocked.
    traceRayEXT(
        tlas,
        gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT,
        SURFACE_GEOMETRY_MASK,
        1,
        0,
        1,
        origin,
        RAY_EPSILON,
        direction,
        max(distance_to_source - RAY_EPSILON, RAY_EPSILON),
        1
    );
    return visibility_payload.occluded == 0u;
}

#include "listener_output.glsl"
#include "listener_surface.glsl"
#include "listener_diffraction.glsl"
