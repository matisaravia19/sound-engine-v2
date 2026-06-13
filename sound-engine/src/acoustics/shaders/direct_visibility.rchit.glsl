#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;

struct AcousticPayload {
    vec4 throughput_distance;
    uvec4 control;
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
    vec4 absorption_scattering;
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
    vec4 source;
    vec4 listener;
    vec4 listener_half_extent;
    uvec4 ray_config;
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
    payload.throughput_distance.y += hit_distance;

    if (payload.control.x >= pc.ray_config.y) {
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

    float absorption = clamp(scene_materials.materials[object.material_index].absorption_scattering.x, 0.0, 1.0);
    payload.throughput_distance.x *= (1.0 - absorption);
    if (payload.throughput_distance.x <= 0.000001) {
        return;
    }

    vec3 reflected_direction = normalize(reflect(gl_WorldRayDirectionEXT, normal));
    vec3 hit_position = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * hit_distance;
    payload.control.x += 1;

    traceRayEXT(
        tlas,
        gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT,
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
