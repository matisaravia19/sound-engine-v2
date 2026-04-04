# System Design Overview

## 1. Purpose

Design a reusable real-time sound engine that simulates acoustic propagation in dynamic 3D scenes and produces low-latency spatial audio output.

Core requirements:

- Geometric acoustics simulation on GPU (ray/path tracing).
- Real-time auralization with partitioned convolution.
- Dynamic playback events and listener motion.
- Extensible architecture for diffraction (main extension) and transmission (optional).

## 2. Scope

In-scope:

- Specular reflections and absorption.
- Early reflections + late energy accumulation into IR.
- Mono source input with stereo or binaural output.
- Vulkan backend for compute and ray tracing.
- Internal IR cache for nearby source/listener positions.

Out-of-scope (for initial thesis delivery):

- Full wave-based simulation (FDTD/BEM/FEM).
- Complex moving deformable geometry at very high rates.
- Full psychoacoustic rendering pipeline (HRTF personalization, room coloration fitting, etc.).

## 3. High-Level Architecture

The engine is split into four layers:

1. `api`: public Rust API and lifecycle.
2. `simulation`: scene preprocessing + GPU acoustic path simulation.
3. `acoustics`: IR construction, temporal filtering, and acoustic model composition.
4. `auralization`: partitioned convolution + spatial renderer in audio callback-safe form.

Cross-cutting layer:

- `gpu`: Vulkan abstractions used by simulation and FFT/convolution stages.

Thesis simplifications:

- Minimal public API surface.
- No telemetry subsystem in v1.
- Scene model contains meshes and materials only.

## 4. Runtime Dataflow

Per update tick (simulation rate, e.g. 30-120 Hz):

1. Consume pending scene updates and sound-play events.
2. Rebuild/patch GPU scene state (BLAS/TLAS/material/edge structures).
3. Resolve IR cache lookup for current source/listener position cells.
4. Launch acoustic simulation only for cache misses or invalid entries.
5. Aggregate arrivals into one or more IRs.
6. Publish IR snapshot to audio thread through lock-free handoff.

Per audio callback (audio rate, e.g. 48 kHz):

1. Read latest stable IR snapshot.
2. Run partitioned convolution on source blocks.
3. Apply spatial output rendering (stereo/binaural).
4. Return audio frame with hard real-time safety.

## 5. Timing Domains

The design uses independent timing domains:

- Control thread: user API calls, scene graph mutation.
- Simulation thread: GPU dispatch + IR generation.
- Audio thread: deterministic low-jitter DSP.

Rules:

- Audio thread never blocks on GPU or allocator locks.
- Simulation can lag by one or more ticks; audio always uses last valid IR.
- Listener/source movement is tolerated up to cache reuse threshold before recompute.

## 6. Quality/Performance Targets

Initial numeric targets (configurable per platform):

- Sample rate: `48_000 Hz`.
- Audio block size: `64-256` samples.
- End-to-end auralization latency budget: `< 20 ms` (excluding external I/O).
- Simulation update period: `8-33 ms` depending scene complexity.
- Ray budget: tunable (`10^4` to `10^6` rays per update).

## 7. Extensibility Strategy

Acoustic effects are composed as independent contributors:

- Reflection contributor (baseline).
- Diffraction contributor (UTD-style edge scattering approximation).
- Transmission contributor (material-filtered through-path energy).

Each contributor outputs arrivals into a common accumulation format:

- `time_of_flight`
- `amplitude` (possibly multi-band)
- `direction_at_listener`
- `source_id`
- optional metadata (`path_order`, `effect_tag`, confidence)

## 8. Failure Model

The engine provides explicit operating modes:

- `FullGpu`: all configured GPU features available.
- `ReducedGpu`: fallback when some extensions are missing.
- `Bypass`: safe dry signal output if simulation fails.

The API exposes diagnostics and mode changes so host applications can react.
