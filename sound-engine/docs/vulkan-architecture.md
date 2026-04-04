# Vulkan Architecture

## 1. Backend Goals

- Provide deterministic, low-overhead GPU execution for simulation and FFT workloads.
- Isolate raw Vulkan from most engine modules.
- Support optional fallbacks when ray tracing extensions are unavailable.

## 2. Core Abstractions

```rust
pub struct VkBackend;
pub struct VkDeviceContext;
pub struct QueueSet;
pub struct CommandArena;
pub struct GpuAllocator;
pub struct ResourceRegistry;
```

Recommended submodules:

- `gpu::backend`: instance/device/feature negotiation.
- `gpu::memory`: allocator, buffer/image pools, staging.
- `gpu::rt`: BLAS/TLAS builders + SBT utilities.
- `gpu::sync`: timeline semaphores, fences, frame tokens.
- `gpu::shader`: shader library, reflection metadata, pipeline cache.

## 3. Feature Negotiation

At startup query and store:

- Vulkan API level (target `1.3+`).
- `VK_KHR_acceleration_structure`
- `VK_KHR_ray_tracing_pipeline`
- `VK_KHR_deferred_host_operations`
- `VK_KHR_buffer_device_address`
- `VK_KHR_synchronization2`

If missing required RT features:

- switch to `ReducedGpu` mode (compute fallback or reduced acoustic model).

## 4. Queue Strategy

Preferred:

- Dedicated compute queue for simulation.
- Optional transfer queue for uploads.
- Optional async compute queue for convolution precompute.

Fallback:

- Single queue with explicit submission ordering.

## 5. Memory Strategy

Memory classes:

- Device-local large pools for scene/static buffers and AS storage.
- Host-visible staging ring for updates and readback.
- Persistent mapped upload buffers for small frequent updates.

Guidelines:

- Avoid per-frame allocations.
- Use sub-allocation and free-lists for transient resources.
- Track budget by category (scene, tracing scratch, IR, audio DSP).

## 6. Command Recording Model

Use frame contexts for simulation ticks:

1. Acquire command context.
2. Record update passes and pipeline dispatches.
3. Submit with timeline semaphore value.
4. Recycle resources after completion value is reached.

This removes ad-hoc fence-heavy immediate submits from hot paths.

## 7. Ray Tracing Pipeline Design

Pipeline groups:

- Ray generation shader (acoustic ray launch).
- Miss shaders (environment/default).
- Hit groups for triangle geometry (reflection/transmission decision points).
- Optional callable shaders for material models.

SBT layout:

- Distinct regions for raygen, miss, hit, callable.
- Stable indexing by material/acoustic effect class.

## 8. Descriptor and Resource Binding

Use descriptor indexing where available:

- Global scene descriptor set (geometry/material/AS handles).
- Per-dispatch descriptor set (source/listener config, output buffers).
- Optional bindless arrays for materials and edge data.

Push constants:

- Small per-dispatch settings (`ray_count`, `max_bounces`, `time_seed`, flags).

## 9. Synchronization Rules

- Prefer timeline semaphores for inter-queue dependencies.
- Use barriers per pass boundary (AS build -> ray trace -> accumulation -> readback/FFT).
- No blocking waits on audio thread.

CPU waits are limited to:

- controlled simulation sync points,
- teardown and explicit debugging modes.

## 10. Shader Build and Cache

Recommendations:

- Offline SPIR-V compilation for production.
- Optional runtime compile in development mode.
- Persistent pipeline cache on disk keyed by GPU + driver + shader hash.

## 11. Error Handling and Recovery

On recoverable failures:

- invalidate affected resources,
- rebuild pipelines/resources lazily,
- degrade features if possible.

On unrecoverable backend failure:

- transition to `Bypass` mode with explicit diagnostic event.
