# Auralization: turning a response into sound

[Guide index](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/codebase-guide/README.md)

Acoustics produces a filter. Auralization applies it to a dry mono signal and mixes the resulting voices into audio blocks. This module has no audio-device callback and does not decide where a source or listener is located.

For the doorway example, it keeps the recording progressing while the engine replaces the filter as the listener moves.

For a fixed IR, the underlying operation is `y[n] = Σ h[k] x[n-k]`: each dry sample contributes delayed, scaled copies according to the response. Here `x` is the dry signal, `h` is one output channel's IR, and `y` is the wet signal. The FFT is an efficient way to calculate this sum.

## Separate the asset, voice, filter, and history

[assets.rs](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/auralization/assets.rs) stores decoded mono samples in a shared immutable allocation. `SoundBank` assigns a `SoundId`. Loading a WAV decodes it, averages interleaved channels to mono, and uses linear resampling when the source rate differs from the target rate. This preprocessing happens before block rendering.

[AuralizationEngine](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/auralization/engine.rs) owns active voices. Each voice retains:

- The dry asset, so removing its sound-bank entry does not interrupt playback.
- A cursor pointing to the next dry frame.
- A finite linear gain applied before convolution.
- A `ConvolutionStream` with its filter and independent overlap history.

A prepared `ImpulseResponse` owns a frequency-domain GPU buffer and its sample-rate contract. Several voices can retain the same prepared filter through `Arc`. Their mutable stream histories remain separate.

This distinction explains why playing a sound twice creates two voices but need not duplicate the dry sample array.

## One block through the engine

`AuralizationEngine::render_block` first validates the output length and clears the mix buffer. For each voice with dry samples remaining, it copies the next mono block, applies gain, pads the final partial block with zeros, and asks the convolver to process it.

When a voice has no dry samples left, the engine drains its already-computed overlap tail instead. A finished recording can therefore remain an active voice until its response has decayed through the retained tail. Completed voices are removed after iteration.

Each wet block is added into the mix. The auralization engine does not normalize the sum to a target loudness. Device output later clamps samples; offline callers can retain the unclamped floating-point mix.

## The FFT calculation

[FftConvolver](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/auralization/convolver.rs) prepares filters, validates compatibility, and provides the public stream operations. [GpuConvolution](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/auralization/convolver/gpu.rs) owns reusable FFT workspaces and the complex-multiply pipeline.

For an input block of `B` frames and an IR of `M` frames, the configured FFT size is the next power of two at least as large as `B + M - 1`.

The calculation is:

~~~text
padded dry block → forward FFT
                 → multiply by prepared IR spectrum
                 → normalized inverse FFT
                 → add preceding overlap
                 → emit B frames and retain the remaining tail
~~~

The implementation stores both ears in a single complex signal. An IR sample becomes `left + i * right`; the dry signal occupies only the real component. By linearity, convolution with that complex IR carries the left result in the real component and the right result in the imaginary component. This avoids separate transform sequences for the two ears.

That packing is a computational representation. The imaginary component is not an additional acoustic quadrature or a second independent physical phase model.

For each processed filter, the padded input is uploaded and the inverse result is downloaded. Forward FFT, spectral multiplication, and inverse FFT are recorded in one GPU submission with barriers between them. Overlap handling and interleaving happen on the CPU. Calling this GPU convolution does not imply that the whole audio path stays GPU-resident.

## Preparing an IR does different work from processing a block

A CPU IR is uploaded into reusable IR scratch storage. A `GpuIrSnapshot` can be copied directly into that storage without an intermediate CPU readback. Both paths zero the unused padding, perform a forward transform, and copy the spectrum into a separately owned filter buffer.

The signal workspace uses inverse normalization. The IR workspace is reused for forward transforms. Recreating native FFT plans for every listener movement would put initialization costs into the update path, so the workspaces persist.

Compatibility checks reject filters from another allocator/backend, a different FFT length, or a different sample rate. Streams also have to match the expected history dimensions and channel count.

## Why overlap belongs to the stream

[stream.rs](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/auralization/convolver/stream.rs) owns current filter history, transition history, pending replacement, and output layout. Each filter's overlap tail represents contributions from dry samples already processed.

After a new inverse transform, those old contributions are added before the next tail is extracted. The tail occupies `FFT length - block size` frames, including numerical padding. Flushing copies this retained data in blocks and zeroes unfilled output frames; it performs no new FFT.

Beginning a flush closes the stream to further input, even if the tail is empty. The cursor uses an explicit “not flushing / flushing at this position” state. A caller that wants another input sequence creates a new stream.

## Moving the listener: preserve the past, replace the future

A filter update starts a four-block, sample-ramped crossfade. The incoming state carries forward the existing overlap tail. Both outgoing and incoming filters then process new dry blocks, and their outputs are blended.

Preserving that tail is crucial. Earlier dry samples have already produced delayed contributions. Deleting those contributions at an update would break continuity even if the old and new filters were numerically identical.

While a transition is active, later updates are coalesced into one pending replacement. Once the current transition finishes, the latest pending IR starts the next one. Reaffirming the incoming filter cancels an unnecessary queued replacement. If tail flushing begins, a pending filter is discarded because no future dry samples remain to excite it.

The transition lasts `4 * block_size / sample_rate` seconds of rendered audio. It is not the same quantity as the time spent tracing the replacement or the delay until queued output reaches the device.

## What to inspect and what the tests establish

Read [the mixing loop](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/auralization/engine.rs), [the convolver facade](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/auralization/convolver.rs), then [stream ownership](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/auralization/convolver/stream.rs). Visit the GPU workspace only when you need the command ordering.

[Convolution regressions](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/tests/unit/auralization/convolver.rs) compare with CPU convolution and exercise padding, ownership, capacity, and draining. [Sine validation](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/tests/sine_convolution_validation.rs) compares output with an analytic delayed-input response. [Moving-IR validation](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/tests/moving_ir_transition_validation.rs) checks continuity across overlapping updates.

The present algorithm transforms a block large enough for the full response. It does not partition a long IR into smaller convolution stages. Long responses and many voices therefore have a different cost profile from a partitioned audio engine, and transitions can temporarily process two filters per voice.
