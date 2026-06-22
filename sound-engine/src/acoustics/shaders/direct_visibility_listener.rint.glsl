#version 460
#extension GL_EXT_ray_tracing : require

hitAttributeEXT vec2 listener_hit;

layout(push_constant) uniform PushConstants {
    layout(offset = 0) vec3 source_position;
    layout(offset = 12) float source_energy;
    layout(offset = 16) vec3 listener_position;
    layout(offset = 28) float speed_of_sound;
    layout(offset = 32) float listener_radius;
    layout(offset = 36) float listener_volume;
    layout(offset = 44) uint ray_count;
    layout(offset = 48) uint max_bounces;
    layout(offset = 64) vec3 listener_right;
    layout(offset = 76) uint sample_rate;
    layout(offset = 80) uint sample_count;
    layout(offset = 84) uint output_channels;
} pc;

void main() {
    vec3 origin = gl_ObjectRayOriginEXT;
    vec3 direction = gl_ObjectRayDirectionEXT;
    float radius = pc.listener_radius;

    float b = dot(origin, direction);
    float c = dot(origin, origin) - radius * radius;
    float discriminant = b * b - c;
    if (discriminant < 0.0) {
        return;
    }

    float root = sqrt(discriminant);
    float t_near = -b - root;
    float t_far = -b + root;
    if (t_far < gl_RayTminEXT || t_near > gl_RayTmaxEXT) {
        return;
    }

    float hit_t = max(t_near, gl_RayTminEXT);
    float exit_t = min(t_far, gl_RayTmaxEXT);
    listener_hit = vec2(max(exit_t - hit_t, 0.0), 0.0);
    reportIntersectionEXT(hit_t, 0);
}
