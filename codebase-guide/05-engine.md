# Engine: coordinating a sound from request to playback

[Guide index](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/codebase-guide/README.md)

The engine module is the orchestration layer. It connects scene editing, acoustic queries, asset playback, spatial refreshes, and optional device playback. Start here when you want to understand what an application call actually does.

The public entry point is [SoundEngine](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/engine/mod.rs). The private [EngineCore](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/engine/direct.rs) owns the acoustic engine, auralizer, source registry, spatial voice metadata, listener pose, and shared backend.

## Follow the first play request

The direct-mode example follows the essential sequence: construct configuration, create the engine, load a scene, set the listener, insert a dry sound, start a voice, and repeatedly render blocks.

A spatial play request has two forms:

| Request | Source lifetime |
| --- | --- |
| `play_sound` | The voice retains a fixed `PointSource` from the request. |
| `play_sound_from_source` | The voice follows a registered `SoundSourceId`. |

The [spatial implementation](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/engine/direct/spatial.rs) checks the request, resolves the source, and verifies that the dry asset is playable before preparing GPU IR resources. It obtains a cached or newly traced time-domain response, prepares its spectrum through the auralizer, starts the voice, and records the spatial metadata.

The auralizer tracks playback cursors and convolution history. The engine separately tracks where each voice gets its source, which raw acoustic IR it last used, and whether a refresh is pending. These registries describe different aspects of the same voice.

A tracked source cannot be removed while a voice still references it. Stopping or naturally completing a voice releases that reference; it does not implicitly remove the source itself.

## Movement makes voices dirty

Changing the listener marks all spatial voices for refresh. Changing a tracked source marks the voices bound to it. Successful scene changes also invalidate affected acoustic results and request voice refreshes. Repeating an identical listener/source value avoids unnecessary dirtying.

A dirty voice keeps playing with its existing filter until refreshed. Refresh resolves the latest source and listener, asks acoustics for a response, and prepares a replacement only if the raw response buffer differs from the one already assigned.

Buffer identity matters here. Moving away and back before the next render can return the same cached response. The engine consumes that pending refresh without starting an unnecessary crossfade.

## Direct mode: the caller chooses the pace

In direct mode, `SoundEngine` owns the core. `render_block` validates output shape, refreshes every pending spatial voice, renders the mix, and removes finished spatial metadata.

This mode is convenient for file generation and controlled tests. It blocks until the requested work finishes. “Direct” means the application drives block production; it does not mean the acoustics are limited to direct sound.

The retained [direct-mode example](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/examples/sound_engine_direct_mode.rs) synthesizes its own input and writes a WAV. It is the smallest practical source tour for the whole stack.

## Runtime mode: ownership moves to a worker

With the `playback` feature, `start_sound_player` moves the core to a background worker. The facade keeps a runtime handle. Ordinary operations are sent as owned closures, executed by that worker between blocks, and acknowledged with their result.

There is still one core owner. The application and worker do not render the same mutable engine concurrently. Calling the facade's direct `render_block` while runtime mode is active returns an error.

There are two listener update semantics:

- `set_listener_pose` waits for an acknowledged update and clears an older pending coalesced pose.
- `update_listener_pose` replaces a capacity-one mailbox without waiting for the worker's next block.

The mailbox suits frequent camera movement: rendering every obsolete camera pose would waste acoustic work. Submission to that mailbox is not proof that a new IR is already audible.

## The runtime's acoustic work budget

[Runtime rendering](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/engine/runtime.rs) normally refreshes at most one dirty voice per block. A rotating identifier cursor prevents a continually moving voice from always winning over other dirty voices.

The loop also tracks render debt. If prior work exceeded the duration represented by an audio block, later blocks can defer spatial refreshes while continuing convolution with existing filters. This trades acoustic freshness for audio continuity. It does not stop the dry cursor or eliminate convolution costs.

The main flow is:

~~~mermaid
sequenceDiagram
    participant App as Application
    participant Worker as Engine worker
    participant GPU as GPU
    participant Queue as Playback queue
    App->>Worker: Coalesced listener pose
    Worker->>Worker: Apply latest pose; mark voices dirty
    Worker->>GPU: Refresh an eligible voice if budget permits
    GPU-->>Worker: Completed response and prepared filter
    Worker->>GPU: Convolve active voices
    GPU-->>Worker: Wet blocks
    Worker->>Queue: Push mixed block
~~~

A voice count reports rendering state. Queued device audio can remain after the last voice has finished generating blocks.

## Startup and shutdown are ownership operations

[worker.rs](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/engine/worker.rs) initializes thread-local playback state before handing over the core. If startup fails, the caller keeps a usable core.

Stopping first wakes a producer that may be waiting on a full playback queue, then requests command-loop shutdown and joins the worker. Ordinary shutdown can recover the core for direct rendering. A worker panic can destroy that owner instead; the facade retains the failure rather than pretending a new core exists.

Sample rate, block size, and output-channel getters remain available independently of that ownership outcome. Worker status publishes errors before waiting callers lose their reply channel, preserving the original cause where possible.

[diagnostics.rs](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/engine/diagnostics.rs) measures tracing, IR preparation, convolution, refresh age, and starvation-related runtime behavior. Its measurements should be read with the buffering model: request age excludes the time for a crossfade to finish and for queued samples to reach the device.

## Read and verify

Follow a facade method into either `EngineCore` or `EngineRuntime`, then inspect [spatial refresh](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/engine/direct/spatial.rs). [Spatial tests](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/tests/unit/engine/direct/spatial.rs) cover source lifetimes, cached refreshes, and coherent pressure updates. [Runtime tests](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/tests/unit/engine/runtime.rs) cover handoff, recovery, and panic reporting.

For our moving listener, keep four moments separate: the UI accepts a pose, the worker applies it, a voice receives a new filter, and the device plays that filter's output. They are intentionally not simultaneous.
