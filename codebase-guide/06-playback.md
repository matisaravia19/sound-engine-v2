# Playback: the boundary between rendered blocks and a device

[Guide index](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/codebase-guide/README.md)

Playback receives interleaved audio that has already been rendered. It owns a CPAL stream and a bounded queue. It knows nothing about triangles, ray paths, or convolution filters.

The distinction matters for our doorway example: moving the listener can make the producer busy tracing, but the device callback still needs to deliver its next buffer on time.

## Four small responsibilities

| File | Responsibility |
| --- | --- |
| [player.rs](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/playback/player.rs) | Public `SoundPlayer`, configuration, push/finish/stop operations. |
| [state.rs](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/playback/state.rs) | Queue ownership, prebuffer gate, lifecycle flags, retained failure, and starvation counters. |
| [queue.rs](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/playback/queue.rs) | Bounded FIFO and producer/consumer coordination. |
| [device.rs](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/playback/device.rs) | Default device selection, typed CPAL callbacks, and channel/sample conversion. |

The module is compiled only with the `playback` feature. The sandbox enables it; applications that only render blocks can omit it.

## The producer may wait; the callback avoids waiting for data

The engine worker pushes complete blocks into a preallocated `VecDeque<f32>`. When there is insufficient capacity for a whole block, the producer waits on a condition variable.

The callback uses `try_lock` when draining the queue. If the producer currently holds the mutex, the callback leaves the queued samples intact and outputs silence for that invocation. It does not wait for the producer to finish. This is a bounded mutex-based design, not a lock-free queue.

Completion and cancellation changes notify waiting producers while respecting the queue's predicate lock. That pairing prevents a producer from checking an old flag and then sleeping after its wakeup has already occurred.

## Prebuffering buys time

`PlaybackConfig` specifies a preferred number of prefilled blocks and a queue-duration target. The allocation is rounded to whole blocks and is at least large enough for the prefill requirement and one block.

Until the prefill threshold is met, callbacks produce silence. The runtime explicitly releases the gate when a short voice finishes producing: otherwise a sound shorter than the prefill target could remain queued forever.

The default player policy is eight prefill blocks and a 0.1-second queue target. The sandbox deliberately supplies its own policy. Values here are configuration choices, not universal engine latency guarantees.

A larger queue can absorb a longer expensive IR update, but also delays the effect of movement. A larger prebuffer changes startup behavior. Neither makes slow sustained rendering faster.

## Mapping engine frames to device frames

The callback starts with silence, then copies available engine channels into the first channels of each device frame. Additional device channels remain silent. A device exposing fewer channels than the engine requires is rejected.

Supported callback sample types are `f32`, `i16`, and `u16`. Values are clamped to [-1, 1] before conversion; non-finite values become silence. Consequently, device output can conceal clipping that remains visible in an offline floating-point render.

The requested engine sample rate is passed to the stream configuration. This layer does not resample output blocks when a device rejects that rate. Asset-rate conversion happens earlier in auralization.

## Finish, stop, failure, and silence mean different things

`finish` closes production and allows queued samples to drain. `wait_until_finished` waits for the callback to observe completion. Later pushes fail.

`stop` requests cancellation and wakes blocked producers. Silence after an explicit stop should not be counted as active starvation.

A device error is stored as the first failure and triggers cancellation. Subsequent checks return that failure rather than treating the interrupted stream as a successful user stop.

Temporary starvation is different again. The callback may have fewer frames than it needs because rendering is late or the queue mutex is busy. That becomes final completion only after the producer has explicitly finished and the callback can actually observe the queue's exhausted tail.

## Interpreting underrun measurements

The state records callbacks with missing frames after audio has begun, while an active producer is expected to supply more. Initial prebuffer silence, idle periods, completed production, and explicit-stop silence are excluded.

Counts describe engine frames, independently of extra device channels. Timing snapshots can lag a concurrent callback; completed-run reports are easier to interpret than individual live reads.

The callback's ordinary sample path does not render the engine, allocate scratch audio buffers, or write diagnostic files. Error handling has separate responsibilities, so avoid describing the entire system as having a formal hard-real-time guarantee.

## Read and verify

Trace `push_block` through player, state, and queue, then follow `write_output` in the other direction. [Queue tests](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/tests/unit/playback/queue.rs) exercise blocking and cancellation, [device tests](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/tests/unit/playback/device.rs) cover mapping and conversion, and [state tests](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/tests/unit/playback/state.rs) cover lifecycle behavior.

[TestPlaybackControl](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/tests/support/playback.rs) drives the same queue/callback logic without opening a physical audio device. Those tests verify software coordination; they do not measure a real device's latency or driver behavior.
