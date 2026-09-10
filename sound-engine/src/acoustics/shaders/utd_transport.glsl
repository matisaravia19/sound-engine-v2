#define SPECTRAL_TRACE
#include "listener_transport.glsl"
#include "utd_transition.glsl"
#ifdef UTD_BANDS
#include "source_directivity.glsl"
void depositDirectionalPressure(float path_length, float amplitude, vec3 arrival, vec3 departure);
#endif

vec2 complexMultiply(vec2 a, vec2 b) {
    return vec2(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x);
}

vec2 transitionF(float x) {
    if (x < 1e-6) return sqrt(PI * max(x, 0.0)) * vec2(0.7071067812);
    if (x >= 1e4) return vec2(1.0, 0.5/x);
    float t = (log(x)/log(10.0) + 6.0) * 51.2;
    int index = min(int(floor(t)), 511);
    return mix(UTD_TRANSITION[index], UTD_TRANSITION[index+1], t-float(index));
}

// Uniform cotangent/Fresnel product, including its one-sided boundary limit.
// At an exact boundary use the symmetric average; nearby GO and UTD terms have
// complementary jumps. This is not an empirical maximum-coefficient clamp.
vec3 prepareUtdTerm(float beta, float sign_branch, float n) {
    float u = (PI + sign_branch*beta)/(2.0*n);
    float nearest = floor(u/PI + 0.5);
    float epsilon = 2.0*n*(u-nearest*PI);
    float N = floor((beta + sign_branch*PI)/(TWO_PI*n) + 0.5);
    float a = 2.0 * pow(cos(0.5*(TWO_PI*n*N-beta)), 2.0);
    return vec3(abs(epsilon) < 1e-4 ? 0.0 : cos(u)/sin(u), a, epsilon);
}

vec2 utdTerm(vec3 prepared, float n, float kL) {
    float epsilon = prepared.z;
    if (abs(epsilon) < 1e-4) {
        float first = sqrt(TWO_PI*kL)*sign(epsilon);
        return n * (vec2(0.7071067812)*first - vec2(0.0, 2.0*kL*epsilon));
    }
    return transitionF(kL*prepared.y) * prepared.x;
}

// Rigid-wedge UTD coefficient (Tsingos Appendix A, equations 4–8).
// The coefficient acts at this ray's edge vertex. No edge-integral IR is read
// or evaluated; the ray's traced lengths supply its spherical wavefront radius.
vec2 utdCoefficient(float k, float L, float sin_incident, float n, vec3 terms[4]) {
    float kL = k*L;
    vec2 sum = utdTerm(terms[0], n, kL) + utdTerm(terms[1], n, kL)
        + utdTerm(terms[2], n, kL) + utdTerm(terms[3], n, kL);
    return complexMultiply(vec2(-0.7071067812, 0.7071067812), sum)
        / (2.0*n*sqrt(TWO_PI*k)*sin_incident);
}

void writeFrequency(uint bin, vec2 value, uint channel) {
    if (finiteVector(vec3(value, 0.0))) {
        uint offset = bin + channel*pc.sample_count;
        atomicAdd(ir.samples[offset].left, value.x);
        atomicAdd(ir.samples[offset].right, value.y);
    }
}

vec2 phaseStep(float path_length) {
    float phase = -TWO_PI*float(pc.sample_rate)/float(2u*(pc.sample_count-1u))*path_length/pc.speed_of_sound;
    return vec2(cos(phase), sin(phase));
}

// Spatialize the ray before accumulation so reflections and edge arrivals keep
// their own direction. The conventions match the time-domain listener tracer.
void spatialResponse(vec3 arrival, out vec2 gains, out float half_delay_length) {
    gains = vec2(1.0, 0.0);
    half_delay_length = 0.0;
    if (pc.output_channels != OUTPUT_CHANNELS_MONO) {
        float lateral = clamp(dot(normalizeOrZero(arrival), normalizeOrZero(pc.listener_right)), -1.0, 1.0);
        gains = sqrt(0.5*vec2(1.0-lateral, 1.0+lateral));
        if (pc.output_channels == OUTPUT_CHANNELS_BINAURAL) {
            float ratio = pow(10.0, -pc.max_ild_db*abs(lateral)/20.0);
            float near_gain = inversesqrt(1.0+ratio*ratio);
            gains = near_gain * (lateral < 0.0 ? vec2(1.0, ratio) : vec2(ratio, 1.0));
            half_delay_length = 0.5*pc.head_radius_meters*(asin(abs(lateral))+abs(lateral))*sign(lateral);
        }
    }
}

void spatialPath(float length, vec3 arrival, out vec2 gains, out vec2 left_step, out vec2 right_step) {
    float half_delay_length;
    spatialResponse(arrival, gains, half_delay_length);
    left_step = phaseStep(length+half_delay_length);
    right_step = phaseStep(length-half_delay_length);
}

#ifdef UTD_TIME_SURFACES
void depositTemporalPressure(float path_length, float amplitude, vec3 arrival, bool direct) {
    if (path_length > pc.maximum_path_length || !finiteFloat(amplitude)) return;
    vec2 gains;
    float half_delay;
    spatialResponse(arrival, gains, half_delay);
    vec2 positions = (path_length+vec2(half_delay, -half_delay))/pc.speed_of_sound*float(pc.sample_rate);
    if (direct) {
        depositDirectChannel(positions.x, amplitude*gains.x, 0u);
        if (pc.output_channels != OUTPUT_CHANNELS_MONO) depositDirectChannel(positions.y, amplitude*gains.y, 1u);
    } else {
        depositChannel(positions.x, amplitude*gains.x, 0u);
        if (pc.output_channels != OUTPUT_CHANNELS_MONO) depositChannel(positions.y, amplitude*gains.y, 1u);
    }
}
#endif

void depositPressure(float path_length, float amplitude, vec3 arrival, vec3 source_departure) {
#ifdef UTD_BANDS
    if (pc.source_profile_enabled != 0u) {
        depositDirectionalPressure(path_length, amplitude, arrival, source_departure);
        return;
    }
#endif
#ifdef UTD_TIME_SURFACES
    depositTemporalPressure(path_length, amplitude, arrival, false);
#else
    if (path_length > pc.maximum_path_length || !finiteFloat(amplitude)) return;
    uint stop = min(pc.sample_count-1u, uint(pc.maximum_frequency_hz*float(2u*(pc.sample_count-1u))/float(pc.sample_rate)));
    vec2 gains, step, right_step;
    spatialPath(path_length, arrival, gains, step, right_step);
    vec2 phase = step;
    vec2 right_phase = right_step;
    for (uint bin=1u; bin<=stop; ++bin) {
        writeFrequency(bin, amplitude*gains.x*phase, 0u);
        if (pc.output_channels != OUTPUT_CHANNELS_MONO) {
            writeFrequency(bin, amplitude*gains.y*right_phase, 1u);
            right_phase = complexMultiply(right_phase, right_step);
        }
        phase = complexMultiply(phase, step);
    }
#endif
}

void depositDirectPressure(float path_length, float amplitude, vec3 arrival, vec3 source_departure) {
#ifdef UTD_BANDS
    if (pc.source_profile_enabled != 0u) {
        depositDirectionalPressure(path_length, amplitude, arrival, source_departure);
        return;
    }
#endif
#ifdef UTD_TIME_SURFACES
    depositTemporalPressure(path_length, amplitude, arrival, true);
#else
    depositPressure(path_length, amplitude, arrival, source_departure);
#endif
}

float exteriorAngle(DiffractionEdge edge, vec3 radial) {
    vec3 perpendicular = normalizeOrZero(radial-dot(radial, edge.edge_dir)*edge.edge_dir);
    return mod(atan(dot(edge.edge_dir, cross(edge.positive_shadow_dir, perpendicular)),
                    dot(edge.positive_shadow_dir, perpendicular)) + TWO_PI, TWO_PI);
}

#ifdef UTD_BANDS
// A fractional ray delay is shared by every carried frequency. Interpolation
// occurs on complex pressure, so both diffraction sign and phase are retained.
void fractionalWeights(float position, out float weights[16]) {
    float fraction = fract(position);
    float sum = 0.0;
    for (int j=0; j<16; ++j) {
        float d = float(j-7)-fraction;
        weights[j] = abs(d) < 1e-6 ? 1.0 : sin(PI*d)/(PI*d)*sin(PI*d/8.0)/(PI*d/8.0);
        sum += weights[j];
    }
    for (int j=0; j<16; ++j) weights[j] /= sum;
}

void writeRayBand(uint band, uint channel, float position, float weights[16], vec2 pressure) {
    uint count = timeDomainSampleCount();
    uint channels = pc.output_channels == OUTPUT_CHANNELS_MONO ? 1u : 2u;
    uint plane = (band*channels+channel)*count;
    int base = int(floor(position));
    if (!finiteVector(vec3(pressure, 0.0))) return;
    for (int j=0; j<16; ++j) {
        int sample_index = base+j-7;
        if (sample_index < 0 || uint(sample_index) >= count) continue;
        atomicAdd(ir.samples[plane+uint(sample_index)].left, pressure.x*weights[j]);
        atomicAdd(ir.samples[plane+uint(sample_index)].right, pressure.y*weights[j]);
    }
}
#endif

#ifdef UTD_BANDS
// Direct and reflected ray pressure needs the same angular source spectrum as
// diffracted pressure. Omni sources keep their original fast temporal deposits.
void depositDirectionalPressure(float path_length, float amplitude, vec3 arrival, vec3 departure) {
    if (path_length > pc.maximum_path_length || !finiteFloat(amplitude)) return;
    vec2 gains;
    float half_delay;
    spatialResponse(arrival, gains, half_delay);
    vec2 positions = (path_length+vec2(half_delay,-half_delay))/pc.speed_of_sound*float(pc.sample_rate);
    float left_weights[16], right_weights[16];
    fractionalWeights(positions.x, left_weights);
    fractionalWeights(positions.y, right_weights);
    uvec4 indices;
    vec4 weights;
    sourceLookupCoordinates(departure, indices, weights);
    for (uint band=0u; band<pc.band_count; ++band) {
        vec2 pressure = amplitude*sourceBandPressure(band, indices, weights);
        writeRayBand(band, 0u, positions.x, left_weights, gains.x*pressure);
        if (pc.output_channels != OUTPUT_CHANNELS_MONO) writeRayBand(band, 1u, positions.y, right_weights, gains.y*pressure);
    }
}
#endif

// The source-launched pass contains only paths with at least one surface
// before their edge. Together with the listener-launched pass these are
// disjoint path classes; direct and zero-surface diffraction are counted once.
void traceZeroAndOneEdge(bool source_launched) {
    uint index = pc.ray_offset + gl_LaunchIDEXT.x;
    if (!source_launched && index == 0u) evaluateDirectPath();
    if (pc.max_bounces == 0u || pc.source_amplitude == 0.0) return;
    if (source_launched && pc.max_bounces < 2u) return;
    uint rng = hashState(index ^ pc.sampling_seed ^ 0xa511e9b3u);
    vec3 direction = fibonacciSphereDirection(index, pc.listener_ray_count, hashState(pc.sampling_seed));
    vec3 initial = direction;
    vec3 origin = source_launched ? pc.source_position : pc.listener_position;
    vec3 vertex = origin;
    vec3 last_surface_normal = vec3(0.0);
    float distance = 0.0;
    float pressure = 1.0;
    float normalization = 4.0*PI/float(pc.listener_ray_count);
    for (uint bounce=0u; bounce<pc.max_bounces; ++bounce) {
        if (!finiteFloat(pressure) || abs(pressure) < pc.minimum_throughput
            || !finiteVector(direction) || distance >= pc.maximum_path_length) break;
        hit_payload.hit_type = HIT_NONE;
        traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, CONTINUATION_GEOMETRY_MASK,
                    0, 0, 0, origin, RAY_EPSILON, direction,
                    max(pc.maximum_path_length-distance, RAY_EPSILON), 0);
        if (hit_payload.hit_type == HIT_NONE) break;
        vec3 hit = origin + direction*hit_payload.t;
        float leg = length(hit-vertex);
        if (distance+leg >= pc.maximum_path_length) break;
        if (hit_payload.hit_type == HIT_SURFACE) {
            vec3 normal = normalizeOrZero(hit_payload.world_normal);
            last_surface_normal = normal;
            SceneMaterial material = scene_materials.materials[hit_payload.data_id];
            if (!source_launched) {
                connectSurfaceToSource(hit, direction, normal, material, distance+leg,
                                       pressure, normalization, initial);
            }
            if (!sampleSurfaceContinuation(material, direction, normal, rng, pressure, direction)) break;
            distance += leg;
            vertex = hit;
            origin = hit+RAY_EPSILON*direction;
            continue;
        }
        if (source_launched && bounce == 0u) break;
        DiffractionEdge edge = diffraction_edges.edges[hit_payload.data_id];
        // Projecting an exterior proxy back onto its own reflecting face can
        // fabricate a tangent reflected leg. A distinct specular reflection
        // from an off-plane endpoint cannot reach that coplanar physical edge.
        if (bounce > 0u && abs(dot(edge.start-vertex, last_surface_normal)) < RAY_EPSILON
            && abs(dot(edge.end-vertex, last_surface_normal)) < RAY_EPSILON) break;
        float radial = hit_payload.barycentrics.x;
        float proxy_cosine = abs(dot(direction, edge.positive_plane_normal));
        float proxy_distance = distance+leg;
        hit = edge.start+hit_payload.barycentrics.y*edge.edge_dir;
        vec3 incoming = normalizeOrZero(hit-vertex);
        leg = length(hit-vertex);
        float r = distance+leg;
        vec3 target = source_launched ? pc.listener_position : pc.source_position;
        vec3 to_source = target-hit;
        float rho = length(to_source);
        if (rho <= RAY_EPSILON || r <= RAY_EPSILON || r+rho > pc.maximum_path_length) break;
        vec3 outgoing = to_source/rho;
        float ai = dot(incoming, edge.edge_dir);
        float ao = dot(outgoing, edge.edge_dir);
        // The Keller condition conserves the signed axial direction component.
        // A normalized Epanechnikov delta approximates that one-dimensional
        // constraint. Its derivative converts edge length to axial mismatch.
        float mismatch = (ai-ao)/pc.cone_width;
        if (abs(mismatch) >= 1.0 || proxy_cosine <= DIRECTION_EPSILON) break;
        float derivative = (1.0-ai*ai)/r + (1.0-ao*ao)/rho;
        float delta = 0.75*(1.0-mismatch*mismatch)/pc.cone_width;
        float incident = exteriorAngle(edge, -incoming);
        float outgoing_angle = exteriorAngle(edge, outgoing);
        if (incident > edge.edge_angle_radians || outgoing_angle > edge.edge_angle_radians) break;
        if (!traceVisibility(vertex, incoming, leg) || !traceVisibility(hit, outgoing, rho)) break;
        // Inverse solid-angle/strip-area Jacobian, then normalized radial
        // capture kernel. No arbitrary edge gain or measured-peak fit appears.
        float radial_kernel = 2.0*max(1.0-radial/edge.diffraction_radius, 0.0)/edge.diffraction_radius;
        pressure *= edge.pressure_gain*normalization
            * proxy_distance*proxy_distance/proxy_cosine*radial_kernel*delta*derivative;
        float spreading = inversesqrt(r*rho*(r+rho));
        float n = edge.edge_angle_radians/PI;
        float sin_incident = sqrt(max(1.0-ai*ai, 1e-6));
        float L = r*rho/(r+rho)*sin_incident*sin_incident;
        // Geometry is invariant across this ray's frequencies. Prepare its
        // four angular terms once, without approximating spectral resolution.
        vec3 terms[4] = vec3[4](prepareUtdTerm(outgoing_angle-incident, 1.0, n),
            prepareUtdTerm(outgoing_angle-incident, -1.0, n),
            prepareUtdTerm(outgoing_angle+incident, 1.0, n),
            prepareUtdTerm(outgoing_angle+incident, -1.0, n));
#ifdef UTD_BANDS
        vec2 gains;
        float half_delay;
        vec3 arrival = source_launched ? -outgoing : (bounce == 0u ? incoming : initial);
        spatialResponse(arrival, gains, half_delay);
        vec2 positions = (r+rho+vec2(half_delay,-half_delay))/pc.speed_of_sound*float(pc.sample_rate);
        float left_weights[16], right_weights[16];
        fractionalWeights(positions.x, left_weights);
        fractionalWeights(positions.y, right_weights);
        // Include the first resolvable FFT frequency. Starting at an arbitrary
        // audio band would incorrectly suppress the low-frequency wedge limit.
        uvec4 source_indices;
        vec4 source_weights;
        sourceLookupCoordinates(source_launched ? initial : -outgoing, source_indices, source_weights);
        float minimum_frequency = float(pc.sample_rate)/exp2(ceil(log2(float(timeDomainSampleCount()))));
        for (uint band=0u; band<pc.band_count; ++band) {
            float f = minimum_frequency*pow(pc.maximum_frequency_hz/minimum_frequency, float(band)/float(pc.band_count-1u));
            vec2 ray_pressure = utdCoefficient(TWO_PI*f/pc.speed_of_sound, L, sin_incident, n, terms)
                * pressure*spreading*pc.source_amplitude;
            ray_pressure = complexMultiply(ray_pressure, sourceBandPressure(band, source_indices, source_weights));
            writeRayBand(band, 0u, positions.x, left_weights, gains.x*ray_pressure);
            if (pc.output_channels != OUTPUT_CHANNELS_MONO) writeRayBand(band, 1u, positions.y, right_weights, gains.y*ray_pressure);
        }
#else
        vec2 gains, step, right_step;
        spatialPath(r+rho, source_launched ? -outgoing : (bounce == 0u ? incoming : initial), gains, step, right_step);
        vec2 phase = step;
        vec2 right_phase = right_step;
        uint stop = min(pc.sample_count-1u, uint(pc.maximum_frequency_hz*float(2u*(pc.sample_count-1u))/float(pc.sample_rate)));
        for (uint bin=1u; bin<=stop; ++bin) {
            float f = float(bin)*float(pc.sample_rate)/float(2u*(pc.sample_count-1u));
            vec2 ray_pressure = utdCoefficient(TWO_PI*f/pc.speed_of_sound, L, sin_incident, n, terms)
                * pressure*spreading*pc.source_amplitude;
            writeFrequency(bin, gains.x*complexMultiply(ray_pressure, phase), 0u);
            if (pc.output_channels != OUTPUT_CHANNELS_MONO) {
                writeFrequency(bin, gains.y*complexMultiply(ray_pressure, right_phase), 1u);
                right_phase = complexMultiply(right_phase, right_step);
            }
            phase = complexMultiply(phase, step);
        }
#endif
        // This first-order research estimator terminates after diffraction.
        // Unsupported higher orders must not silently fall back to cosine gain.
        break;
    }
}
