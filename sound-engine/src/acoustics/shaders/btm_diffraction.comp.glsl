#version 460
#extension GL_EXT_shader_atomic_float : require

// Each work item integrates one midpoint segment of one authored finite edge.
// The formulation is the time-domain Biot-Tolstoy-Medwin wedge response used by
// the repository's independent CPU reference benchmark.

const float PI = 3.14159265358979323846;
const float TWO_PI = 6.28318530717958647692;
const float EPSILON = 0.000001;
const uint OUTPUT_CHANNELS_MONO = 0;
const uint OUTPUT_CHANNELS_STEREO = 1;
const uint OUTPUT_CHANNELS_BINAURAL = 2;

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct IrSample {
    float left;
    float right;
};

layout(std430, set = 0, binding = 0) buffer IrBuffer {
    IrSample samples[];
} ir;

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

layout(std430, set = 0, binding = 1) readonly buffer DiffractionEdgeBuffer {
    DiffractionEdge edges[];
} diffraction_edges;

layout(push_constant) uniform PushConstants {
    layout(offset = 0) vec3 source_position;
    layout(offset = 12) float source_amplitude;
    layout(offset = 16) vec3 listener_position;
    layout(offset = 28) float maximum_path_length;
    layout(offset = 32) vec3 listener_right;
    layout(offset = 44) float speed_of_sound;
    layout(offset = 48) uint sample_rate;
    layout(offset = 52) uint sample_count;
    layout(offset = 56) uint output_channels;
    layout(offset = 60) uint edge_count;
    layout(offset = 64) uint segments_per_edge;
    layout(offset = 68) float head_radius_meters;
    layout(offset = 72) float max_ild_db;
    layout(offset = 76) uint _pad0;
} pc;

bool finiteFloat(float value) {
    return !(isnan(value) || isinf(value));
}

vec3 normalizeOrZero(vec3 value) {
    float squared_length = dot(value, value);
    return squared_length > EPSILON * EPSILON ? value * inversesqrt(squared_length) : vec3(0.0);
}

void depositChannel(float sample_position, float amplitude, uint channel) {
    if (!finiteFloat(sample_position)
        || !finiteFloat(amplitude)
        || sample_position <= -1.0
        || sample_position >= float(pc.sample_count)) {
        return;
    }

    int base = int(floor(sample_position));
    int next = base + 1;
    float fraction = sample_position - float(base);
    if (base >= 0 && uint(base) < pc.sample_count) {
        if (channel == 0u) {
            atomicAdd(ir.samples[uint(base)].left, amplitude * (1.0 - fraction));
        } else {
            atomicAdd(ir.samples[uint(base)].right, amplitude * (1.0 - fraction));
        }
    }
    if (fraction > 0.0 && next >= 0 && uint(next) < pc.sample_count) {
        if (channel == 0u) {
            atomicAdd(ir.samples[uint(next)].left, amplitude * fraction);
        } else {
            atomicAdd(ir.samples[uint(next)].right, amplitude * fraction);
        }
    }
}

void depositPressure(float total_path_length, float amplitude, vec3 listener_arrival_direction) {
    if (!finiteFloat(total_path_length)
        || !finiteFloat(amplitude)
        || total_path_length < 0.0
        || total_path_length > pc.maximum_path_length) {
        return;
    }

    float center_sample_position = total_path_length
        / max(pc.speed_of_sound, EPSILON)
        * float(pc.sample_rate);
    if (pc.output_channels == OUTPUT_CHANNELS_MONO) {
        depositChannel(center_sample_position, amplitude, 0u);
        return;
    }

    float lateral = clamp(
        dot(normalizeOrZero(listener_arrival_direction), normalizeOrZero(pc.listener_right)),
        -1.0,
        1.0
    );
    if (pc.output_channels == OUTPUT_CHANNELS_STEREO) {
        depositChannel(center_sample_position, amplitude * sqrt(0.5 * (1.0 - lateral)), 0u);
        depositChannel(center_sample_position, amplitude * sqrt(0.5 * (1.0 + lateral)), 1u);
        return;
    }

    float absolute_lateral = abs(lateral);
    float itd_seconds = pc.head_radius_meters
        / max(pc.speed_of_sound, EPSILON)
        * (asin(absolute_lateral) + absolute_lateral)
        * sign(lateral);
    float half_itd_samples = 0.5 * itd_seconds * float(pc.sample_rate);
    float far_to_near = pow(10.0, -pc.max_ild_db * absolute_lateral / 20.0);
    float near_factor = inversesqrt(1.0 + far_to_near * far_to_near);
    float far_factor = far_to_near * near_factor;
    float left_factor = lateral < 0.0 ? near_factor : far_factor;
    float right_factor = lateral > 0.0 ? near_factor : far_factor;
    if (lateral == 0.0) {
        left_factor = near_factor;
        right_factor = near_factor;
    }
    depositChannel(center_sample_position + half_itd_samples, amplitude * left_factor, 0u);
    depositChannel(center_sample_position - half_itd_samples, amplitude * right_factor, 1u);
}

float wedgeAngle(vec3 origin, vec3 axis, vec3 direction) {
    vec3 normalized_origin = normalizeOrZero(origin);
    vec3 normalized_direction = normalizeOrZero(direction);
    float sine = dot(axis, cross(normalized_origin, normalized_direction));
    float cosine = dot(normalized_origin, normalized_direction);
    float angle = atan(sine, cosine);
    return angle < 0.0 ? angle + TWO_PI : angle;
}

void main() {
    uint segment_index = gl_GlobalInvocationID.x;
    uint edge_index = gl_GlobalInvocationID.y;
    if (edge_index >= pc.edge_count || segment_index >= pc.segments_per_edge) {
        return;
    }

    DiffractionEdge edge = diffraction_edges.edges[edge_index];
    float strength = sqrt(clamp(edge.base_strength, 0.0, 1.0));
    if (strength <= 0.0 || edge.edge_length <= EPSILON) {
        return;
    }

    vec3 edge_axis = normalizeOrZero(edge.edge_dir);
    vec3 source_from_start = pc.source_position - edge.start;
    vec3 listener_from_start = pc.listener_position - edge.start;
    vec3 source_radial = source_from_start - dot(source_from_start, edge_axis) * edge_axis;
    vec3 listener_radial = listener_from_start - dot(listener_from_start, edge_axis) * edge_axis;
    float source_radius = length(source_radial);
    float listener_radius = length(listener_radial);
    if (source_radius <= EPSILON || listener_radius <= EPSILON) {
        return;
    }

    float exterior_angle = clamp(edge.edge_angle_radians, PI + EPSILON, TWO_PI - EPSILON);
    float wedge_index = PI / exterior_angle;
    float source_angle = wedgeAngle(edge.positive_shadow_dir, edge_axis, source_radial);
    float listener_angle = wedgeAngle(edge.positive_shadow_dir, edge_axis, listener_radial);

    float segment_length = edge.edge_length / float(pc.segments_per_edge);
    float edge_coordinate = (float(segment_index) + 0.5) * segment_length;
    vec3 edge_point = edge.start + edge_axis * edge_coordinate;
    float source_axial = dot(edge_point - pc.source_position, edge_axis);
    float listener_axial = dot(edge_point - pc.listener_position, edge_axis);
    float source_distance = sqrt(source_radius * source_radius + source_axial * source_axial);
    float listener_distance = sqrt(listener_radius * listener_radius + listener_axial * listener_axial);
    float total_path_length = source_distance + listener_distance;
    if (total_path_length > pc.maximum_path_length) {
        return;
    }

    float eta_argument = (source_distance * listener_distance + source_axial * listener_axial)
        / (source_radius * listener_radius);
    float eta = acosh(max(eta_argument, 1.0));
    float beta_sum = 0.0;
    for (int source_sign = -1; source_sign <= 1; source_sign += 2) {
        for (int listener_sign = -1; listener_sign <= 1; listener_sign += 2) {
            float phi = PI + float(source_sign) * source_angle + float(listener_sign) * listener_angle;
            float denominator = cosh(wedge_index * eta) - cos(wedge_index * phi);
            if (denominator > EPSILON) {
                beta_sum += sin(wedge_index * phi) / denominator;
            }
        }
    }

    // The engine's direct field is 1/r rather than 1/(4*pi*r), so the common
    // 4*pi calibration reduces the published BTM coefficient to -nu here.
    float pressure = pc.source_amplitude
        * strength
        * (-wedge_index * beta_sum)
        * segment_length
        / (source_distance * listener_distance);
    depositPressure(total_path_length, pressure, edge_point - pc.listener_position);
}
