#version 460
#extension GL_EXT_ray_tracing : require

layout(push_constant) uniform PushConstants {
    vec4 source;
    vec4 listener;
    vec4 listener_half_extent;
} pc;

void main() {
    vec3 box_min = -pc.listener_half_extent.xyz;
    vec3 box_max = pc.listener_half_extent.xyz;
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

    reportIntersectionEXT(t_near, 0);
}
