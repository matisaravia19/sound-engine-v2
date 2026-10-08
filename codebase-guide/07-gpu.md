# GPU: Vulkan resources, commands, and lifetime boundaries

[Guide index](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/codebase-guide/README.md)

The GPU module provides the machinery used by both acoustic tracing and audio convolution. It translates buffers, pipeline descriptions, and recorded work into Vulkan objects and submissions. It does not define the acoustic model.

For our doorway example, it retains the scene's acceleration structures, runs the acoustic shader, transforms the new IR, and processes audio blocks. Most high-level operations wait before publishing their result, even though Vulkan itself supports asynchronous execution.

## Read the module as a set of owners

| Submodule | Main responsibility |
| --- | --- |
| [backend](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/gpu/backend.rs) | Assemble subsystems around one shared device. |
| [backend/device](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/gpu/backend/device.rs) | Select/create the device and retain instance/loader lifetime. |
| [memory](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/gpu/memory.rs) | Allocate buffers and perform uploads, clears, copies, and downloads. |
| [compute](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/gpu/compute/mod.rs) | Own the queue/command pool, compute pipelines, and submission tracking. |
| [rt](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/gpu/rt/mod.rs) | Own acceleration structures, ray-tracing pipelines, and dispatch support. |
| [shader](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/gpu/shader.rs) | Compile and retain shader modules. |
| [descriptor](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/gpu/descriptor.rs) | Descriptor writes and pool/set ownership shared by compute and RT. |
| [resource](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/gpu/resource.rs) | Common Vulkan lifetime guards and layout validation. |

These are meaningful ownership boundaries. A raw Vulkan handle is merely a reference to an object; retaining an integer handle does not keep its allocation, device, or dependencies alive.

## Device setup is shared by all engine workloads

`VkBackend::new` creates a Vulkan 1.3 instance and selects a suitable physical device with a compute-capable queue family. The requirements include acceleration structures, ray-tracing pipelines, deferred host operations, buffer device addresses, 32-bit floating-point buffer atomic addition, and shader integer support for device-address access.

The same backend serves tracing and convolution. Even an experiment using only the current convolver creates this RT-capable backend; there is no independent CPU or compute-only fallback selected by that API.

Subsystems retain `Arc` owners of the device and one another where necessary. The Vulkan loader stays resident across backend lifetimes, and each device context retains the handles needed for final destruction.

## Buffer ownership and data transfer

`GpuAllocator` uses Vulkan Memory Allocator through `vk-mem`. A [BufferHandle](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/gpu/memory/buffer.rs) owns the buffer/allocation relationship and retains the context needed to release it.

Device-local processing buffers and host-visible staging buffers serve different access patterns. Upload/download helpers validate bounds and ownership, arrange copies and synchronization, and handle mapped-memory coherency where required. A buffer from another backend must not be passed as though its handle belonged to this allocator.

Typed transfers use explicit byte layouts. Host structs shared with GLSL use stable representation and padding. `bytemuck` checks that a type can be viewed as bytes, but it does not independently prove agreement with a shader's offsets.

A **device address** lets a shader dereference buffer storage without a storage-buffer descriptor for that particular array. The surface object table uses this for vertices and indices. The containing scene must retain those arrays throughout traversal.

## Commands have two different safety questions

First: *have the commands finished?* A fence makes completion observable to the CPU.

Second: *can the next operation see the preceding writes in the required way?* Memory barriers specify the stage/access dependency.

These questions are related but not interchangeable. The code uses both completion waits and explicit barriers. For example:

~~~text
clear time-domain IR
  → transfer-write to ray-tracing read/write barrier
  → atomic pressure accumulation
  → ray-tracing-write to compute/transfer-read barrier
  → completion wait
~~~

Convolution has its own sequence: forward FFT writes become visible to spectral multiplication, whose writes become visible to the inverse FFT.

## Submission and descriptor lifetime

[submit.rs](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/gpu/compute/submit.rs) records work under serialized queue/command-pool access. It tracks submitted command buffers and fences, then recycles them on completion. A guard cleans up work that never reached submission, including recording-error paths.

`submit_compute_and_wait` is the convenient synchronous path. The lower-level unsafe `submit_compute` returns a `Submission`. That handle borrows the context; it does **not** own every referenced buffer and does **not** wait on drop. Callers must retain resources and explicitly wait.

Descriptor sets describe the resources a recorded dispatch uses. A later dispatch cannot overwrite bindings still needed by an earlier in-flight command. Allocated sets are associated with command buffers and retained until completion, while a shared guard serializes descriptor-pool allocation/free operations.

The high-level engine's synchronous ownership makes these contracts manageable. It does not make arbitrary raw Vulkan calls through exposed handles automatically safe.

## Acceleration structures

[acceleration.rs](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/gpu/rt/acceleration.rs) builds triangle or procedural-AABB BLASes and instance TLASes. Builds use temporary scratch storage whose address is aligned to the device's requirement. They complete before their IDs are returned.

A TLAS retains strong references to the distinct BLASes it uses. Removing a BLAS's registry entry therefore does not invalidate an existing TLAS's backing storage. Scoped owners such as `OwnedBlas` and `OwnedTlas` release scene-owned registry entries when replaced.

The scene module decides *which* geometry to include and *when* a group is stale. The GPU module decides *how* to build and own the Vulkan structures. Keeping those policies separate is the reason not every scene change requires new mesh storage.

## Pipelines and the shader binding table

A compute pipeline runs the complex multiply shader. A ray-tracing pipeline connects shader stages into groups.

The [shader binding table](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/gpu/rt/sbt.rs), or SBT, is a buffer of pipeline-provided group handles arranged into ray-generation, miss, and hit regions. Record stride and base address must satisfy device alignment limits; the allocation is padded so the usable address can be aligned inside it.

The acoustic pipeline uses distinct hit records for surface continuation, surface visibility, and procedural edges. Instance offsets, trace arguments, and SBT group ordering must agree. A mismatch can invoke valid code for the wrong interaction, so this is an algorithmic interface as well as a memory-layout concern.

The [shader compiler](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/gpu/shader/compiler.rs) invokes `glslangValidator` for Vulkan 1.3. Bundled source bytes and includes are compiled in an owned temporary workspace. Native VkFFT-generated kernels follow a separate compilation path described in the next chapter.

## Evidence and limits

[GPU audits](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/tests/unit/gpu/audit.rs) exercise lifetimes and teardown. [Acceleration tests](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/tests/unit/gpu/rt/acceleration.rs), [SBT tests](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/tests/unit/gpu/rt/sbt.rs), and [submission tests](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/tests/unit/gpu/compute/submit.rs) cover their respective contracts.

Hardware checks with Vulkan synchronization validation can reveal invalid ordering and lifetime use. Passing them is evidence for executed paths and that device/driver combination. It is not evidence that an unexecuted unsafe call is valid or that all drivers behave identically.

When reading a GPU operation, follow the owning values from allocation through the completion wait to destruction. That is often more informative than following raw handles alone.
