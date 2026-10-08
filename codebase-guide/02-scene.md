# Scene: editable descriptions and a lazy GPU mirror

[Guide index](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/codebase-guide/README.md)

The scene module translates application-owned descriptions into validated CPU state and then into GPU resources. Its central design is that editing a scene and rebuilding acceleration structures are separate operations.

Moving the listener through the doorway does not edit the scene. Moving a wall, changing its material, or adding a diffraction edge does.

## The vocabulary: assets, placements, and edges

[types.rs](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/scene/types.rs) defines the input data.

A `MeshAsset` is reusable triangle geometry in local coordinates. A `SceneObject` places a mesh in world space, selects its material, and decides whether it participates in traversal. Several objects can instance one mesh. An inactive object remains in CPU state.

A `Material` carries scalar absorption, scattering, and roughness. Absorption controls the reflected remainder; scattering divides reflection between two lobes; roughness controls lobe width. None of these fields provides surface transmission.

A `DiffractionEdge` is an explicitly supplied finite segment in world coordinates, with an exterior wedge angle, bisector, and capture radius. It is independent of object transforms. Moving a mesh does not automatically move its authored edges. The caller must keep them consistent.

The wedge angle is the **exterior air angle**: a solid right-angle corner uses `3π/2`, while a thin screen uses `2π`. The capture radius controls a virtual discovery strip, not the location of the physical diffraction vertex.

## Three owners, three responsibilities

| Type | Responsibility |
| --- | --- |
| `SceneDescription` | Complete replacement payload supplied by a caller. |
| `SceneStore` | Canonical validated records, identifiers, and revisions. |
| `SceneManager` | Owns the store and its most recently built `GpuSceneResources`. |

[SceneStore](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/scene/store.rs) applies changes transactionally. A complete load is assembled and validated before replacing the old state. A nonempty update batch is applied to a clone, in order, and committed only after every edit succeeds. Immutable mesh arrays are shared through `Arc` during that process.

If you remove a mesh while an object still references it, the batch fails. If you first remove that object and then remove the mesh in the same batch, the ordering can make the operation valid. Failure leaves the published records and revisions unchanged.

An empty batch does not increment the version. An accepted nonempty batch does, even if an assigned value happens to equal its previous value.

## Why there is more than one revision

The overall scene version invalidates acoustic results. More specific revisions allow independent GPU groups to survive unrelated edits:

| Edit | Main GPU consequence |
| --- | --- |
| Material value or object material assignment | Rebuild shading tables; retain geometry and TLAS. |
| Object transform or active membership | Rebuild surface instances and TLAS; retain unchanged mesh BLASes. |
| Mesh replacement | Upload that mesh and rebuild its BLAS; update dependent instance/shading data. |
| Diffraction-edge edit | Rebuild edge records/procedural geometry and the combined TLAS. |

The revisions are conservative invalidation signals, not a fine-grained geometric diff. A full scene load drops the old mirror. Incremental updates enable reuse.

[SceneManager::sync_gpu_if_needed](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/scene/manager.rs) performs lazy synchronization when a query actually needs traversal data. If GPU construction fails, the accepted CPU scene remains, and a later synchronization can retry. This is distinct from the CPU transaction's all-or-nothing validation.

## What gets uploaded

[gpu.rs](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/scene/gpu.rs) owns uploaded meshes, shader tables, edge resources, and one complete acoustic TLAS. A BLAS accelerates one mesh or the procedural edge geometry. The TLAS places those structures into the scene.

Surface and edge instances have different visibility masks. Ordinary visibility tests see surfaces; continuation rays can discover surfaces and edge proxies. Even an empty scene has a valid empty TLAS, so the direct-sound calculation can use the same query machinery.

The material and object tables are compact GPU arrays. Authored IDs are not array indices. Active objects are sorted before assigning the indices used by hit shaders.

[records.rs](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/scene/gpu/records.rs) isolates the Rust/GLSL record layout and CPU conversions. Object records include raw vertex/index device addresses. Their scene owner must retain those buffers for as long as hit shaders can dereference them. Edge AABB input buffers, by contrast, are temporary once their synchronous acceleration-structure build finishes.

## Deeper detail: energy inputs become pressure coefficients

The scene accepts an absorption fraction `α` and scattering fraction `s`. The uploaded lobe pressure coefficients use:

~~~text
a = sqrt(1 - s)
b = sqrt(s)
specular_pressure = sqrt(1 - α) * a / (a + b)
rough_pressure    = sqrt(1 - α) * b / (a + b)
~~~

Their sum is `sqrt(1 - α)`. This normalization matters because transport combines scalar pressure contributions. Simply summing two independent square-root energy gains could increase the pressure weight at every bounce.

This is the implemented broadband convention. It does not model independent diffuse phases or establish exact directional energy conservation. Keep that qualification with the equations in thesis notes.

## Read and verify

Start with [types](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/scene/types.rs), follow one update through [the store](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/scene/store.rs), then inspect [GPU reuse](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/scene/gpu.rs).

[Store tests](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/tests/unit/scene/store.rs) cover rejection and rollback. [GPU synchronization tests](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/tests/unit/scene/gpu_sync.rs) check resource reuse and rebuilding on actual hardware. [Record tests](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/tests/unit/scene/gpu/records.rs) check conversion and layout assumptions.

For the doorway example, identify which edits change geometry and which only change the query. That tells you whether the next IR needs a new TLAS, a new trace, or merely a reusable cache entry.
