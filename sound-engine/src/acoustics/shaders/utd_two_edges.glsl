// Consecutive two-edge research estimator. Geometry is discovered by traversal;
// the completed ray carries the two local complex coefficients and its delay.
// Derivation and limits: references/two-edge-ray-estimator.md.
float kellerDelta(float mismatch) {
    float x = mismatch/pc.cone_width;
    return abs(x) < 1.0 ? 0.75*(1.0-x*x)/pc.cone_width : 0.0;
}

bool traceEdgeVertex(vec3 vertex, vec3 direction, float remaining, out uint id,
                     out vec3 edge_point) {
    hit_payload.hit_type = HIT_NONE;
    traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, CONTINUATION_GEOMETRY_MASK,
        0, 0, 0, vertex+RAY_EPSILON*direction, RAY_EPSILON, direction, remaining, 0);
    if (hit_payload.hit_type != HIT_DIFFRACTION_EDGE) return false;
    id = hit_payload.data_id;
    DiffractionEdge edge = diffraction_edges.edges[id];
    float cosine = abs(dot(direction, edge.positive_plane_normal));
    if (cosine <= DIRECTION_EPSILON) return false;
    edge_point = edge.start+hit_payload.barycentrics.y*edge.edge_dir;
    float length = distance(edge_point, vertex);
    if (length <= RAY_EPSILON || !traceVisibility(vertex, (edge_point-vertex)/length, length)) return false;
    return true;
}

bool prepareEdgeTerms(DiffractionEdge edge, vec3 incoming, vec3 outgoing, out vec3 terms[4]) {
    float ti = exteriorAngle(edge, -incoming);
    float to = exteriorAngle(edge, outgoing);
    if (ti > edge.edge_angle_radians || to > edge.edge_angle_radians) return false;
    float n = edge.edge_angle_radians/PI;
    terms[0] = prepareUtdTerm(to-ti, 1.0, n);
    terms[1] = prepareUtdTerm(to-ti, -1.0, n);
    terms[2] = prepareUtdTerm(to+ti, 1.0, n);
    terms[3] = prepareUtdTerm(to+ti, -1.0, n);
    return true;
}

// Importance sampling changes only ray launch directions. The selected proxy
// must still be the first traversed edge, and its relocated physical leg must
// pass visibility. A uniform component retains support over the entire edge.
bool sampleProxyRay(vec3 vertex, vec3 endpoint, uint target, float remaining, inout uint rng,
                    out vec3 point, out float weight) {
    DiffractionEdge edge = diffraction_edges.edges[target];
    if (edge.edge_length <= RAY_EPSILON || edge.diffraction_radius <= RAY_EPSILON) return false;
    vec3 v = vertex-edge.start;
    vec3 s = endpoint-edge.start;
    float zv = dot(v, edge.edge_dir), zs = dot(s, edge.edge_dir);
    float rv = length(v-zv*edge.edge_dir), rs = length(s-zs*edge.edge_dir);
    float center = clamp((zv*rs+zs*rv)/max(rv+rs,RAY_EPSILON), 0.0, edge.edge_length);
    float half_width = 2.0*pc.cone_width/max(1.0/max(rv,0.01)+1.0/max(rs,0.01), 1e-5);
    half_width = max(half_width, 0.01);
    float lo = max(0.0, center-half_width), hi = min(edge.edge_length, center+half_width);
    bool guided = randomFloat(rng) < 0.9;
    float z = mix(guided ? lo : 0.0, guided ? hi : edge.edge_length, randomFloat(rng));
    float pdf_z = 0.1/edge.edge_length + (z >= lo && z <= hi ? 0.9/(hi-lo) : 0.0);
    float radial = edge.diffraction_radius*(1.0-sqrt(1.0-randomFloat(rng)));
    vec3 proxy = edge.start+z*edge.edge_dir+radial*edge.bisector_dir;
    vec3 direction = normalizeOrZero(proxy-vertex);
    uint actual;
    if (!traceEdgeVertex(vertex, direction, remaining, actual, point) || actual != target) return false;
    // The triangular radial proposal exactly cancels the radial capture kernel;
    // the edge-coordinate PDF and discrete edge selection remain explicit.
    weight = 1.0/pdf_z;
    return true;
}

// Complete an actually traversed pair. A planar reflected prefix retains a
// spherical wavefront with radius r0; its pressure/proposal factors arrive here
// as ray state, never as an analytically supplied reflection path.
void depositTwoEdgePath(uint id1, uint id2, vec3 e1, vec3 e2,
                        float r0, vec3 incoming, vec3 initial, bool source_launched,
                        float proposal, float prefix_pressure) {
    DiffractionEdge edge1 = diffraction_edges.edges[id1];
    DiffractionEdge edge2 = diffraction_edges.edges[id2];
    vec3 between = e2-e1;
    float r1 = length(between);
    between /= r1;
    vec3 outgoing = (source_launched ? pc.listener_position : pc.source_position)-e2;
    float r2 = length(outgoing);
    float total = r0+r1+r2;
    if (r2 <= RAY_EPSILON || total > pc.maximum_path_length) return;
    outgoing /= r2;
    if (!traceVisibility(e2, outgoing, r2)) return;
    float a = dot(incoming, edge1.edge_dir);
    float b = dot(between, edge1.edge_dir);
    float c = dot(between, edge2.edge_dir);
    float d = dot(outgoing, edge2.edge_dir);
    float delta = kellerDelta(a-b)*kellerDelta(c-d);
    if (delta == 0.0) return;
    // The two constraints are coupled by the middle leg. A product of diagonal
    // derivatives overweights short gaps; the off-diagonal term is essential.
    float h11 = (1.0-a*a)/r0 + (1.0-b*b)/r1;
    float h22 = (1.0-c*c)/r1 + (1.0-d*d)/r2;
    float h12 = -(dot(edge1.edge_dir, edge2.edge_dir)-b*c)/r1;
    float determinant = max(0.0, h11*h22-h12*h12);
    float weight = proposal*delta*determinant;
    weight *= (edge1.pressure_gain * edge2.pressure_gain);
    vec3 terms1[4], terms2[4];
    if (!prepareEdgeTerms(edge1, incoming, between, terms1) || !prepareEdgeTerms(edge2, between, outgoing, terms2)) return;
    float sin1 = sqrt(max(1.0-a*a, 1e-6));
    float sin2 = sqrt(max(1.0-c*c, 1e-6));
    float L1 = r0*r1/(r0+r1)*sin1*sin1;
    // The wave incident at edge 2 is astigmatic, not spherical: its principal
    // radii are r1 and r0+r1. Project the edge planes onto the transverse plane
    // and use normal curvature to obtain the second edge's caustic distance.
    vec3 transverse1 = normalizeOrZero(edge1.edge_dir-b*between);
    vec3 transverse2 = normalizeOrZero(edge2.edge_dir-c*between);
    float aligned = clamp(dot(transverse1, transverse2), -1.0, 1.0);
    float rho2 = 1.0/(aligned*aligned/(r0+r1)+(1.0-aligned*aligned)/r1);
    // General UTD distance parameter: Kodra (2025), Eq. 3.41. For parallel
    // edges this reduces to r1*r2/(r1+r2), independent of tracing direction.
    float L2 = r2*(rho2+r2)*r1*(r0+r1)/(rho2*(r1+r2)*total)*sin2*sin2;
    float spreading = sqrt(rho2/(r0*r1*(r0+r1)*r2*(rho2+r2)));
    vec2 gains;
    float half_delay;
    spatialResponse(source_launched ? -outgoing : initial, gains, half_delay);
    vec2 positions = (total+vec2(half_delay,-half_delay))/pc.speed_of_sound*float(pc.sample_rate);
    float left_weights[16], right_weights[16];
    fractionalWeights(positions.x, left_weights);
    fractionalWeights(positions.y, right_weights);
    uvec4 source_indices;
    vec4 source_weights;
    sourceLookupCoordinates(source_launched ? initial : -outgoing, source_indices, source_weights);
    float minimum_frequency = float(pc.sample_rate)/exp2(ceil(log2(float(timeDomainSampleCount()))));
    for (uint band=0u; band<pc.band_count; ++band) {
        float f = minimum_frequency*pow(pc.maximum_frequency_hz/minimum_frequency, float(band)/float(pc.band_count-1u));
        float k = TWO_PI*f/pc.speed_of_sound;
        vec2 ray_pressure = utdCoefficient(k, L1, sin1, edge1.edge_angle_radians/PI, terms1);
        ray_pressure = complexMultiply(ray_pressure, utdCoefficient(k, L2, sin2, edge2.edge_angle_radians/PI, terms2));
        ray_pressure *= pc.source_amplitude*prefix_pressure*weight*spreading;
        ray_pressure = complexMultiply(ray_pressure, sourceBandPressure(band, source_indices, source_weights));
        writeRayBand(band, 0u, positions.x, left_weights, gains.x*ray_pressure);
        if (pc.output_channels != OUTPUT_CHANNELS_MONO) writeRayBand(band, 1u, positions.y, right_weights, gains.y*ray_pressure);
    }
}

void traceTwoEdges() {
    if (pc.max_bounces < 2u || pc.source_amplitude == 0.0) return;
    uint index = pc.ray_offset + gl_LaunchIDEXT.x;
    // Hash the seed before combining it with the index: index XOR a small seed
    // merely permutes a power-of-two launch and repeats the entire sample set.
    uint rng = hashState(index + hashState(pc.sampling_seed ^ 0x82745321u));
    uint count = uint(diffraction_edges.edges.length());
    if (count < 2u) return;
    uint id1 = min(uint(randomFloat(rng)*float(count)), count-1u);
    uint id2 = min(uint(randomFloat(rng)*float(count-1u)), count-2u);
    if (id2 >= id1) ++id2;
    vec3 e1, e2;
    float weight1, weight2;
    if (!sampleProxyRay(pc.listener_position, pc.source_position, id1, pc.maximum_path_length, rng, e1, weight1)) return;
    vec3 incoming = e1-pc.listener_position;
    float r0 = length(incoming);
    incoming /= r0;
    if (!sampleProxyRay(e1, pc.source_position, id2, max(RAY_EPSILON, pc.maximum_path_length-r0), rng, e2, weight2)) return;
    float proposal = float(count)*float(count-1u)/float(pc.listener_ray_count)*weight1*weight2;
    depositTwoEdgePath(id1, id2, e1, e2, r0, incoming, incoming, false, proposal, 1.0);
}

// Edge-axis guidance changes launch density only. A full-sphere component
// preserves support when a surface rotates the axial direction. The marginal
// density sums every axis proposal, so coincident/parallel edges are not counted
// as different pressure paths. No reflection vertex is solved by this proposal.
vec3 sampleSurfaceLaunch(vec3 origin, vec3 endpoint, uint index, inout uint rng) {
    uint count = uint(diffraction_edges.edges.length());
    float total = 0.0;
    for (uint i=0u; i<count; ++i) total += diffraction_edges.edges[i].edge_length;
    float choice = randomFloat(rng)*total;
    vec3 axis = vec3(0.0, 0.0, 1.0);
    for (uint i=0u; i<count; ++i) {
        choice -= diffraction_edges.edges[i].edge_length;
        if (choice <= 0.0) { axis = diffraction_edges.edges[i].edge_dir; break; }
    }
    float center = dot(normalizeOrZero(endpoint-origin), axis);
    // Two tolerated Keller mismatches can move the launch axial cosine by up
    // to twice the kernel width for parallel edges. A narrower guide puts much
    // of that contribution into rare, high-weight uniform-component samples.
    float width = max(pc.cone_width*2.0, 0.02);
    float lo = max(-1.0, center-width), hi = min(1.0, center+width);
    bool guided = randomFloat(rng) < 0.9;
    // Shifted equal-area latitude and golden-angle azimuth reduce launch holes.
    // Integer phase accumulation avoids large-index float angle quantization.
    float shift = float(hashState(pc.sampling_seed ^ 0x43872195u) >> 8)*(1.0/16777216.0);
    float unit_axial = fract((float(index)+0.5)/float(pc.listener_ray_count)+shift);
    uint phase = index*0x61c88647u+hashState(pc.sampling_seed ^ 0x19376285u);
    float axial = mix(guided ? lo : -1.0, guided ? hi : 1.0, unit_axial);
    float phi = TWO_PI*float(phase >> 8)*(1.0/16777216.0);
    vec3 tangent, bitangent;
    orthonormalBasis(axis, tangent, bitangent);
    return axial*axis+sqrt(max(0.0, 1.0-axial*axial))*(cos(phi)*tangent+sin(phi)*bitangent);
}

float surfaceLaunchPdf(vec3 direction, vec3 origin, vec3 endpoint) {
    uint count = uint(diffraction_edges.edges.length());
    vec3 straight = normalizeOrZero(endpoint-origin);
    float width = max(pc.cone_width*2.0, 0.02);
    float total = 0.0, density = 0.0;
    for (uint i=0u; i<count; ++i) {
        DiffractionEdge edge = diffraction_edges.edges[i];
        total += edge.edge_length;
        float center = dot(straight, edge.edge_dir);
        float lo = max(-1.0, center-width), hi = min(1.0, center+width);
        float axial = dot(direction, edge.edge_dir);
        if (axial >= lo && axial <= hi) density += edge.edge_length/(TWO_PI*(hi-lo));
    }
    return 0.1/(4.0*PI)+0.9*density/total;
}

// Exactly one surface followed by two consecutive edges. Source/listener
// launches represent R-D-D and D-D-R respectively, disjoint from pure D-D.
// No reflected endpoint or path graph is constructed during runtime traversal.
void traceSurfaceTwoEdges(bool source_launched) {
    if (pc.max_bounces < 3u || pc.source_amplitude == 0.0) return;
    uint count = uint(diffraction_edges.edges.length());
    if (count < 2u) return;
    uint index = pc.ray_offset + gl_LaunchIDEXT.x;
    // Use the same stream for a physical endpoint in either tracing direction.
    // Reversing an omni-source query then pairs each sampled R-D-D ray with its
    // D-D-R counterpart. Correlation does not change either proposal's density;
    // independent seeds, not reversal differences, measure sampling variance.
    uint salt = 0x24673185u;
    uint rng = hashState(index + hashState(pc.sampling_seed ^ salt));
    vec3 origin = source_launched ? pc.source_position : pc.listener_position;
    vec3 endpoint = source_launched ? pc.listener_position : pc.source_position;
    vec3 direction = sampleSurfaceLaunch(origin, endpoint, index, rng);
    vec3 initial = direction;
    hit_payload.hit_type = HIT_NONE;
    traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, CONTINUATION_GEOMETRY_MASK,
                0, 0, 0, origin, RAY_EPSILON, direction, pc.maximum_path_length, 0);
    if (hit_payload.hit_type != HIT_SURFACE) return;
    vec3 surface = origin+direction*hit_payload.t;
    float prefix_length = length(surface-origin);
    vec3 normal = normalizeOrZero(hit_payload.world_normal);
    SceneMaterial material = scene_materials.materials[hit_payload.data_id];
    float pressure = 1.0;
    if (!sampleSurfaceContinuation(material, direction, normal, rng, pressure, direction)) return;
    if (!finiteFloat(pressure) || abs(pressure) < pc.minimum_throughput) return;
    float remaining = pc.maximum_path_length-prefix_length;
    if (remaining <= RAY_EPSILON) return;
    hit_payload.hit_type = HIT_NONE;
    traceRayEXT(tlas, gl_RayFlagsOpaqueEXT, CONTINUATION_GEOMETRY_MASK,
                0, 0, 0, surface+RAY_EPSILON*direction, RAY_EPSILON, direction, remaining, 0);
    if (hit_payload.hit_type != HIT_DIFFRACTION_EDGE) return;
    uint id1 = hit_payload.data_id;
    DiffractionEdge edge1 = diffraction_edges.edges[id1];
    // A distinct specular reflection cannot reach a coplanar edge along its
    // own face: its stationary vertex collapses onto the edge. Proxy projection
    // can otherwise manufacture a grazing leg already covered by the wedge.
    if (abs(dot(edge1.start-surface, normal)) < RAY_EPSILON
        && abs(dot(edge1.end-surface, normal)) < RAY_EPSILON) return;
    float cosine = abs(dot(direction, edge1.positive_plane_normal));
    if (cosine <= DIRECTION_EPSILON) return;
    float radial = hit_payload.barycentrics.x;
    float proxy_distance = prefix_length+hit_payload.t+RAY_EPSILON;
    vec3 e1 = edge1.start+hit_payload.barycentrics.y*edge1.edge_dir;
    vec3 incoming = e1-surface;
    float leg = length(incoming);
    if (leg <= RAY_EPSILON) return;
    incoming /= leg;
    float r0 = prefix_length+leg;
    if (r0 >= pc.maximum_path_length || !traceVisibility(surface, incoming, leg)) return;
    uint id2 = min(uint(randomFloat(rng)*float(count-1u)), count-2u);
    if (id2 >= id1) ++id2;
    vec3 e2;
    float weight2;
    if (!sampleProxyRay(e1, endpoint, id2, pc.maximum_path_length-r0, rng, e2, weight2)) return;
    float radial_kernel = 2.0*max(1.0-radial/edge1.diffraction_radius, 0.0)/edge1.diffraction_radius;
    // Only the second edge is selected discretely. The first hit's launch PDF
    // includes all axis proposals as well as their full-sphere mixture component.
    float proposal = 1.0/(float(pc.listener_ray_count)*surfaceLaunchPdf(initial, origin, endpoint))*proxy_distance*proxy_distance/cosine
                   *radial_kernel*float(count-1u)*weight2;
    depositTwoEdgePath(id1, id2, e1, e2, r0, incoming, initial, source_launched, proposal, pressure);
}
