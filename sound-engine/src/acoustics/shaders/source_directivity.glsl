// Immutable source calibration, resampled once to the tracer's band nodes.
// This multiplies pressure on a traced path; it never creates a propagation path.
layout(std430, set=0, binding=5) readonly buffer SourceDirectivityBuffer {
    uint phi_count;
    uint theta_count;
    uint bands;
    uint padding;
    vec2 pressure[];
} source_directivity;

void sourceLookupCoordinates(vec3 departure, out uvec4 indices, out vec4 weights) {
    indices = uvec4(0u);
    weights = vec4(1.0, 0.0, 0.0, 0.0);
    if (pc.source_profile_enabled == 0u) return;
    vec4 q = pc.source_inverse_rotation;
    vec3 local = departure + 2.0*cross(q.xyz, cross(q.xyz, departure) + q.w*departure);
    local = normalizeOrZero(local);
    // At the poles all azimuth rows agree, so atan(0,0) need not be evaluated.
    float phi = dot(local.yz, local.yz) < 1e-12 ? 0.0 : mod(atan(local.z, local.y)+TWO_PI, TWO_PI);
    float p = phi/TWO_PI*float(source_directivity.phi_count);
    float t = acos(clamp(local.x, -1.0, 1.0))/PI*float(source_directivity.theta_count-1u);
    uint p0 = uint(floor(p)) % source_directivity.phi_count;
    uint p1 = (p0+1u) % source_directivity.phi_count;
    uint t0 = min(uint(floor(t)), source_directivity.theta_count-1u);
    uint t1 = min(t0+1u, source_directivity.theta_count-1u);
    indices = uvec4(p0*source_directivity.theta_count+t0, p0*source_directivity.theta_count+t1,
                    p1*source_directivity.theta_count+t0, p1*source_directivity.theta_count+t1)*source_directivity.bands;
    vec2 fraction = fract(vec2(p, t));
    weights = vec4((1.0-fraction.x)*(1.0-fraction.y), (1.0-fraction.x)*fraction.y,
                   fraction.x*(1.0-fraction.y), fraction.x*fraction.y);
}

vec2 sourceBandPressure(uint band, uvec4 indices, vec4 weights) {
    if (pc.source_profile_enabled == 0u) return vec2(1.0, 0.0);
    return weights.x*source_directivity.pressure[indices.x+band]
         + weights.y*source_directivity.pressure[indices.y+band]
         + weights.z*source_directivity.pressure[indices.z+band]
         + weights.w*source_directivity.pressure[indices.w+band];
}
