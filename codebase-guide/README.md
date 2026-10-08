# A reading guide to the codebase

This project turns a geometric scene into spatial audio. It first estimates how sound travels from a point source to a listener, then applies that response to a recording or synthesized signal. Vulkan performs ray tracing and FFT work; Rust coordinates scene state, resource lifetimes, voices, and playback.

The guide follows the implementation at commit `1959058f5`, reviewed on 7 October 2026. It is a companion for returning to the code while writing the thesis. Model descriptions below are statements about this implementation; they do not establish that every approximation is physically exact. Links lead to the current source and the tests that support particular claims.

## Start with one sound in one room

Imagine a short dry recording playing beside a doorway. The listener stands in the neighboring room and then moves toward the opening.

The scene describes the walls and doorway. A point source gives the sound's position and reference pressure amplitude. A listener pose gives the receiver's position and orientation. The acoustic calculation produces an **impulse response**, or IR: a sampled description of arrivals, delays, and signed amplitudes for that configuration.

A **voice** combines the dry recording with a prepared version of that IR. Convolution turns the dry signal into the signal heard through the scene. The output becomes interleaved audio blocks, which can be written to a file or queued for an audio device.

When the listener moves, the dry recording keeps its playback position. The engine requests an updated IR and transitions the existing voice toward it. Moving the listener is therefore a change in the acoustic filter, not a restart of the recording.

## The architecture

The arrows below show the principal flow of information and work, rather than every Rust dependency.

~~~mermaid
flowchart TD
    App["Application or sandbox"] --> Engine["engine: orchestration"]
    Engine --> Scene["scene: validated geometry and updates"]
    Scene --> Acoustics["acoustics: time-domain pressure IR"]
    Engine --> Acoustics
    Dry["Dry mono asset"] --> Auralization["auralization: voices and convolution"]
    Acoustics --> Auralization
    Engine --> Auralization
    Auralization --> Blocks["Interleaved audio blocks"]
    Blocks --> File["Application or file output"]
    Blocks --> Playback["playback: bounded queue and audio device"]
    Scene -.-> GPU["gpu: Vulkan resources and submission"]
    Acoustics -.-> GPU
    Auralization -.-> GPU
    Auralization -.-> FFT["vkfft-rs: native FFT plans"]
~~~

The central library is `sound-engine`. Its public facade is `SoundEngine`, but the lower-level scene, acoustic, and auralization APIs are also available. This is useful for experiments that need an IR without playback, or convolution without retracing a scene.

The workspace has three other crates:

| Crate | Role |
| --- | --- |
| `sandbox` | Interactive application built with raylib and the playback-enabled engine. |
| `acoustic-fixtures` | Shared scene builders and input readers for experiments and the sandbox. |
| `vkfft-rs` | Rust/native integration for the FFT operations used by convolution. |

`acoustic-fixtures` is a development dependency of the engine. Normal library use does not load the experimental scene catalog. The library's `playback` feature enables CPAL integration; rendering blocks directly does not require an audio device. It still uses the Vulkan backend.

## A small vocabulary that prevents confusion

| Term | Meaning here |
| --- | --- |
| Sound / `SoundId` | An immutable mono asset in a sound bank. |
| Source / `SoundSourceId` | A spatial emitter; a tracked source can move while voices follow it. |
| Voice / `VoiceId` | One playback instance, with its own cursor, gain, and convolution history. |
| `GpuIrSnapshot` | A completed time-domain IR with shared GPU storage and query provenance. |
| `IrSnapshot` | A downloaded CPU copy, suitable for inspection and export. |
| `ImpulseResponse` | A prepared frequency-domain filter used by the convolver. |
| Block | A fixed number of audio frames processed together. A stereo frame has two samples. |
| BLAS / TLAS | Acceleration structures for geometry and its scene instances, respectively. |

Several voices can share one sound or one prepared filter. They cannot share mutable overlap history: previous input contributes to each voice's future output independently.

## What the model currently does

The acoustic pipeline traces paths backward from the listener and explicitly tests connections to the source. Direct sound is handled once, separately from sampled indirect paths. Surfaces absorb or reflect; authored finite edges provide a scalar diffraction approximation. Stereo and analytic binaural output use the arrival direction at the listener.

The stored IR is signed **pressure**, not a histogram of independent path energies. This matters when arrivals add or cancel. The diagnostic field named `energy` sums squared stored samples; it is not, by itself, a proof of conservation of physical acoustic energy.

Important limits to keep in view while writing:

- Materials and diffraction use broadband scalar responses.
- The engine does not transmit sound through surfaces.
- Binaural output uses an analytic delay/level model, without a measured HRTF database.
- Rays, interaction limits, path-length limits, and the finite IR window restrict the calculation.
- The implementation uses full-response FFT convolution, rather than partitioning a long IR.

The chapters explain where each limitation enters the code.

## Choose a reading route

For a first pass, read the overview, **Engine**, **Scene**, **Acoustics**, and **Auralization**. That follows one complete sound. Return to **Core** for units and configuration, then read **Playback**, **GPU**, and **VkFFT** to understand execution and ownership.

| Chapter | Question it answers |
| --- | --- |
| [Core](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/codebase-guide/01-core.md) | What settings, units, and invariants do the modules share? |
| [Scene](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/codebase-guide/02-scene.md) | How does editable geometry become reusable GPU traversal data? |
| [Acoustics](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/codebase-guide/03-acoustics.md) | How are paths sampled and turned into an IR? |
| [Auralization](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/codebase-guide/04-auralization.md) | How does that IR turn a dry asset into audio? |
| [Engine](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/codebase-guide/05-engine.md) | Who coordinates requests, voices, spatial updates, and threads? |
| [Playback](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/codebase-guide/06-playback.md) | How do rendered blocks reach the device safely? |
| [GPU](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/codebase-guide/07-gpu.md) | How are Vulkan memory, commands, and ray-tracing objects managed? |
| [VkFFT](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/codebase-guide/08-vkfft.md) | What crosses the native boundary, and who owns it? |
| [Sandbox](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/codebase-guide/09-sandbox.md) | How does an interactive application drive the engine? |
| [Fixtures](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/codebase-guide/10-fixtures.md) | Where do experimental scenes and measured inputs come from? |
| [Tests and experiments](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/codebase-guide/11-tests-and-experiments.md) | Which checks support which claims, and how are results reproduced? |

Read the narrative first; use the source links on a second pass. The deeper sections in Acoustics and GPU are reference material when a formula or synchronization decision needs closer attention.

Most chapters take roughly four to six minutes to read without following the code links; Acoustics takes a little longer. Each chapter covers one major module or supporting crate, with its smaller submodules explained together.

## Your first source tour

Open [the library entry point](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/lib.rs), then [the direct-mode example](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/examples/sound_engine_direct_mode.rs). Follow its calls into [the facade](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/engine/mod.rs). At each step, ask which value is geometry, which is an acoustic response, and which is actual audio.

For experiments, preserve the configuration and the code revision alongside the result. A figure from an older transport implementation cannot be explained reliably by reading today's shaders alone.
