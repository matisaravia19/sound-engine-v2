#version 460
#extension GL_EXT_ray_tracing : require

hitAttributeEXT vec2 listener_hit;

layout(push_constant) uniform PushConstants {
    layout(offset = 0) vec3 source_position;
    layout(offset = 12) float source_energy;
    layout(offset = 16) vec3 listener_position;
    layout(offset = 28) float speed_of_sound;
    layout(offset = 32) vec3 listener_half_extent;
    layout(offset = 44) uint ray_count;
    layout(offset = 48) uint max_bounces;
    layout(offset = 64) vec3 listener_right;
    layout(offset = 76) uint sample_rate;
    layout(offset = 80) uint sample_count;
    layout(offset = 84) uint output_channels;
} pc;

void main() {
    vec3 box_min = -pc.listener_half_extent;
    vec3 box_max = pc.listener_half_extent;
    vec3 origin = gl_ObjectRayOriginEXT;
    vec3 direction = gl_ObjectRayDirectionEXT;
    float t_near = gl_RayTminEXT;
    float t_far = gl_RayTmaxEXT;

    for (int axis = 0; axis < 3; axis++) {
        if (abs(direction[axis]) < 0.000001) {
            if (origin[axis] < box_min[axis] || origin[axis] > box_max[axis]) {
                return;
            }
            continue;
        }

        float inv_dir = 1.0 / direction[axis];
        float t0 = (box_min[axis] - origin[axis]) * inv_dir;
        float t1 = (box_max[axis] - origin[axis]) * inv_dir;
        if (t0 > t1) {
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }

        t_near = max(t_near, t0);
        t_far = min(t_far, t1);
        if (t_near > t_far) {
            return;
        }
    }

    listener_hit = vec2(max(t_far - t_near, 0.0), 0.0);
    reportIntersectionEXT(t_near, 0);
}
