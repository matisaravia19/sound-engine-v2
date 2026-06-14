#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;

struct AcousticPayload {
    float ray_energy;
    float path_distance;
    uint reflection_order;
};

struct SceneObject {
    uint material_index;
    uint indexed;
    uint _pad0;
    uint _pad1;
    uint64_t vertex_address;
    uint64_t index_address;
    mat4 object_to_world;
};

struct SceneMaterial {
    float absorption;
    float scattering;
    float transmission;
    float _pad0;
};

layout(std430, set = 0, binding = 2) readonly buffer SceneObjectBuffer {
    SceneObject objects[];
} scene_objects;

layout(std430, set = 0, binding = 3) readonly buffer SceneMaterialBuffer {
    SceneMaterial materials[];
} scene_materials;

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer VertexBuffer {
    float values[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer IndexBuffer {
    uint values[];
};

layout(location = 0) rayPayloadInEXT AcousticPayload payload;

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

vec3 loadVertex(VertexBuffer vertices, uint vertex_index) {
    uint base = vertex_index * 3;
    return vec3(vertices.values[base], vertices.values[base + 1], vertices.values[base + 2]);
}

vec3 transformPoint(mat4 object_to_world, vec3 point) {
    return (object_to_world * vec4(point, 1.0)).xyz;
}

void main() {
    float hit_distance = gl_HitTEXT;
    payload.path_distance += hit_distance;

    if (payload.reflection_order >= pc.max_bounces) {
        return;
    }

    SceneObject object = scene_objects.objects[gl_InstanceCustomIndexEXT];
    if (object.vertex_address == 0) {
        return;
    }

    VertexBuffer vertices = VertexBuffer(object.vertex_address);
    uint i0 = gl_PrimitiveID * 3;
    uint i1 = i0 + 1;
    uint i2 = i0 + 2;
    if (object.indexed != 0) {
        IndexBuffer indices = IndexBuffer(object.index_address);
        i0 = indices.values[i0];
        i1 = indices.values[i1];
        i2 = indices.values[i2];
    }

    vec3 p0 = transformPoint(object.object_to_world, loadVertex(vertices, i0));
    vec3 p1 = transformPoint(object.object_to_world, loadVertex(vertices, i1));
    vec3 p2 = transformPoint(object.object_to_world, loadVertex(vertices, i2));
    vec3 edge0 = p1 - p0;
    vec3 edge1 = p2 - p0;
    vec3 normal = normalize(cross(edge0, edge1));
    if (dot(normal, normal) <= 0.0) {
        normal = -normalize(gl_WorldRayDirectionEXT);
    }
    if (dot(normal, gl_WorldRayDirectionEXT) > 0.0) {
        normal = -normal;
    }

    // Only the absorption coefficient is used for now; scattering is reserved
    // in the material record for later diffuse reflection paths.
    float absorption = clamp(scene_materials.materials[object.material_index].absorption, 0.0, 1.0);
    payload.ray_energy *= (1.0 - absorption);
    if (payload.ray_energy <= 0.000001) {
        return;
    }

    vec3 reflected_direction = normalize(reflect(gl_WorldRayDirectionEXT, normal));
    vec3 hit_position = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * hit_distance;
    payload.reflection_order += 1;

    // Offset the secondary ray to avoid immediately re-hitting the same triangle.
    traceRayEXT(
        tlas,
        gl_RayFlagsOpaqueEXT,
        0xff,
        0,
        0,
        0,
        hit_position + reflected_direction * 0.001,
        0.001,
        reflected_direction,
        10000.0,
        0
    );
}
