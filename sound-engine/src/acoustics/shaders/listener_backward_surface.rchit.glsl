#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

// Surface closest-hit is intentionally a traversal adapter. It reconstructs
// the world-space geometric normal and material index needed by ray generation,
// but performs no acoustic response evaluation and launches no child rays.

struct HitPayload {
    uint hit_type;
    float t;
    uint instance_id;
    uint primitive_id;
    vec2 barycentrics;
    vec3 world_normal;
    uint data_id;
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

layout(std430, set = 0, binding = 2) readonly buffer SceneObjectBuffer {
    SceneObject objects[];
} scene_objects;

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer VertexBuffer {
    float values[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer IndexBuffer {
    uint values[];
};

layout(location = 0) rayPayloadInEXT HitPayload payload;
hitAttributeEXT vec2 triangle_barycentrics;

vec3 loadVertex(VertexBuffer vertices, uint vertex_index) {
    uint base = vertex_index * 3u;
    return vec3(vertices.values[base], vertices.values[base + 1u], vertices.values[base + 2u]);
}

vec3 transformPoint(mat4 object_to_world, vec3 point) {
    return (object_to_world * vec4(point, 1.0)).xyz;
}

void main() {
    // Scene instance custom indices address the compact active-object table
    // uploaded alongside the TLAS. Device addresses keep vertex/index arrays
    // out of the descriptor set and must remain alive with the scene resources.
    SceneObject object = scene_objects.objects[gl_InstanceCustomIndexEXT];
    if (object.vertex_address == 0ul) {
        payload.hit_type = 0u;
        return;
    }

    VertexBuffer vertices = VertexBuffer(object.vertex_address);
    uint i0 = gl_PrimitiveID * 3u;
    uint i1 = i0 + 1u;
    uint i2 = i0 + 2u;
    if (object.indexed != 0u) {
        IndexBuffer indices = IndexBuffer(object.index_address);
        i0 = indices.values[i0];
        i1 = indices.values[i1];
        i2 = indices.values[i2];
    }

    vec3 p0 = transformPoint(object.object_to_world, loadVertex(vertices, i0));
    vec3 p1 = transformPoint(object.object_to_world, loadVertex(vertices, i1));
    vec3 p2 = transformPoint(object.object_to_world, loadVertex(vertices, i2));
    // Recompute a geometric normal after object-to-world transformation so
    // non-uniform object transforms do not require a separate normal matrix.
    vec3 normal = normalize(cross(p1 - p0, p2 - p0));
    if (any(isnan(normal)) || any(isinf(normal)) || dot(normal, normal) <= 0.0) {
        normal = -normalize(gl_WorldRayDirectionEXT);
    }
    // Orient the normal against the incident path. Ray generation can then use
    // consistent reflection hemisphere tests.
    if (dot(normal, gl_WorldRayDirectionEXT) > 0.0) {
        normal = -normal;
    }

    payload.hit_type = 1u;
    payload.t = gl_HitTEXT;
    payload.instance_id = gl_InstanceCustomIndexEXT;
    payload.primitive_id = gl_PrimitiveID;
    payload.barycentrics = triangle_barycentrics;
    payload.world_normal = normal;
    payload.data_id = object.material_index;
}
