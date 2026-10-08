# Core: shared settings, units, and failure contracts

[Guide index](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/codebase-guide/README.md)

The core module supplies the small agreements that let the rest of the engine cooperate. It contains configuration, error types, identifier allocation, and geometry checks. It owns no scene, voice, audio stream, or Vulkan device.

For our doorway example, core answers questions such as: what does a coordinate mean, how many frames are in an audio block, how long is the response, and what happens when a caller supplies an invalid pose or mesh?

## Read the configuration as separate budgets

[EngineConfig](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/core/config.rs) groups five settings:

| Setting | Controls |
| --- | --- |
| `SoundConfig` | Sample rate, generated IR length, and mono/stereo/binaural output. |
| `AcousticsConfig` | Listener ray count, interaction budget, path-length cutoff, throughput cutoff, and seed. |
| `IrCacheConfig` | Permitted spatial/orientation reuse and cache entry/byte limits. |
| `BinauralConfig` | Head radius and maximum broadband interaural level difference. |
| `AuralizationConfig` | Input frames per convolution block. |

These budgets affect different things. Increasing rays changes indirect-path sampling. Increasing IR length extends the storage window and can increase FFT cost. Increasing block size changes audio granularity and the required FFT length. A long IR does not make paths longer if the tracing cutoff remains short.

`EngineConfig::validate` checks settings before device initialization. Limits specific to the selected GPU are checked later, when that device and its properties are known.

## Units worth writing beside your notes

Positions and path lengths are in **meters**. Sample rate is frames per second. IR length and block size are frame counts. The acoustic pipeline uses a fixed sound speed of **343 m/s**. A path of length `L` reaches nominal sample position:

~~~text
sample_position = (L / 343) * sample_rate
~~~

The result is usually fractional; the shader interpolates the arrival into the sample grid.

For a block of `B` frames and `C` output channels, the caller provides `B * C` floating-point samples. At 48 kHz, 1,024 frames represent about 21.3 ms. That is one block's duration, not the complete motion-to-ear latency: scheduling, queued audio, filter transitions, and the device add other delays.

The convolver selects:

~~~text
FFT length = next_power_of_two(block_size + ir_num_samples - 1)
~~~

The padding makes room for linear convolution. Arithmetic is checked before allocation so overflow cannot turn an enormous requested buffer into a small valid-looking one.

## The output enum encodes two different ideas

`OutputChannels::Mono` has one output channel. `Stereo` and `Binaural` both have two, but the acoustic shader interprets them differently. Stereo uses equal-power panning; binaural adds an analytic head-related delay and level difference.

By the time samples reach the convolver, both two-channel modes have the same packed layout. The convolver does not know whether their left/right differences came from panning, the binaural model, or a caller-supplied IR.

## Shared geometry checks

[geometry.rs](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/core/geometry.rs) validates complete triangle triples, finite vertices, legal indices, and representable counts. Instance transforms must be finite, affine, and invertible.

These checks are shared because both scene editing and lower-level GPU uploads need consistent input rules. They do not prove that a mesh is watertight, correctly oriented, or physically appropriate for an acoustic experiment. Authored wedge geometry has additional checks in the scene module.

Endpoint validation lives in [acoustics/spatial.rs](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/acoustics/spatial.rs). A listener's approximate forward/up vectors are converted to an orthonormal basis. Non-finite, zero, or collinear directions are rejected. The default listener faces negative Z with positive Y up.

## Errors survive the operation that created them

[SoundError](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/core/error.rs) contains an `ErrorCode`, a message, and an optional shared underlying cause. `SoundResult<T>` is the library-wide result type. Sharing the cause makes errors cloneable without discarding the original dependency error—useful when a worker failure must remain observable by later callers.

Distinguish invalid arguments from invalid state. A wrongly sized output block is an argument error; trying to render directly while a worker owns the engine is a state error. Device unavailability, allocation failure, shader compilation, I/O, and poisoned locks have their own categories.

[id.rs](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/core/id.rs) allocates monotonically increasing engine-local identifiers with checked exhaustion. An old handle should not silently acquire a new meaning after an entry is removed. These identifiers are not globally unique across independent engine instances.

## Follow this in the source

Read configuration validation, then the error representation, then the short geometry checks. [Configuration tests](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/tests/unit/core/config.rs), [geometry tests](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/tests/unit/core/geometry.rs), and [identifier tests](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/tests/unit/core/id.rs) show the rejected boundary cases.

As a small exercise, change only the hypothetical sample rate in a configuration you are reading. Identify which quantities retain their physical duration and which retain only their sample count. That distinction recurs throughout the experiments.
