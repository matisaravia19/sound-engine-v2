# Acoustics: from geometry to a pressure impulse response

[Guide index](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/codebase-guide/README.md)

The acoustics module answers one question: **given this scene, source, and listener, what sampled response should filter the dry sound?** It does not advance a recording, mix voices, or talk to an audio device.

In the doorway example, moving the listener changes both propagation paths and their apparent arrival directions. This chapter follows the calculation that produces the replacement IR.

## Begin at the Rust boundary

[AcousticsEngine](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/acoustics/engine.rs) owns a scene manager, an `AcousticPipeline`, and an IR cache. An `AcousticQuery` contains a source, a listener pose, and an application query identifier. The engine's convenience API uses identifier zero.

The query is validated and its listener basis canonicalized before cache lookup. The same canonical values are preserved in the result. On a miss, the pipeline synchronizes scene resources, allocates and clears a time-domain IR buffer, traces, waits for completion, and publishes a `GpuIrSnapshot`. Storage always has left/right lanes; mono output uses only the left lane.

The GPU snapshot shares its allocation and records the actual query and scene version that produced it. Downloading it creates an `IrSnapshot`; [export.rs](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/acoustics/export.rs) can write a CSV with sample indices, times, channels, and metadata.

## The path runs backward

The entry point is [listener_backward.rgen.glsl](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/acoustics/shaders/listener_backward.rgen.glsl). Each invocation launches from the listener. Its state stays in the ray-generation shader while traversal calls report the next interaction.

The initial directions use a seeded, rotated Fibonacci-sphere distribution. Indirect contributions receive a spherical normalization of `4π/N`, where `N` is the listener ray count. Random choices then select continuations.

Invocation zero separately evaluates the direct listener-to-source segment. Its contribution does not receive `4π/N`: it is evaluated once, rather than estimated from sphere samples. With zero allowed interactions, the direct calculation still runs.

A useful conceptual trace is:

~~~text
listener
  → discover the next surface or edge proxy
  → prepare the physical interaction
  → test a connection from that interaction to the source
  → choose one continuation
  → repeat while the path budget permits
~~~

The explicit source connection is next-event estimation. It avoids requiring a random continuation ray to hit a zero-size point source. A blocked connection does not necessarily end the path; a later interaction might still connect.

The GPU pipeline uses recursion depth one. More acoustic interactions mean more iterations of the ray-generation loop, not deeper recursive calls from hit shaders.

## Read the shaders by responsibility

| Source | Responsibility |
| --- | --- |
| [listener_path.glsl](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/acoustics/shaders/listener_path.glsl) | Accumulated state, interaction preparation, path termination, and continuation orchestration. |
| [listener_transport.glsl](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/acoustics/shaders/listener_transport.glsl) | Shared traversal/visibility helpers and inclusion of local transport operations. |
| [listener_surface.glsl](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/acoustics/shaders/listener_surface.glsl) | Surface source connections and reflection sampling. |
| [listener_diffraction.glsl](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/acoustics/shaders/listener_diffraction.glsl) | Edge response, capture measure, and continuation proposals. |
| [listener_sampling.glsl](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/acoustics/shaders/listener_sampling.glsl) | Random draws, direction generation, and lobe densities. |
| [listener_output.glsl](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/acoustics/shaders/listener_output.glsl) | Arrival timing, output spatialization, and atomic IR deposition. |
| [listener_interface.glsl](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/acoustics/shaders/listener_interface.glsl) | Shader-side descriptors, push constants, and shared declarations. |

The seven acoustic shader entry points are the ray-generation stage, two miss stages, two surface hit stages, and the diffraction intersection/closest-hit pair. Include files factor their implementation; they are not additional Vulkan stages.

Continuation hit shaders return compact geometry information. Visibility uses a separate payload and surface-only traversal. Neither kind of hit shader owns the acoustic path or writes the IR.

## Surfaces: one sampled continuation, evaluated source connections

A surface has specular and rough reflection lobes. The coefficients uploaded by the scene module are pressure gains. Source connections evaluate the valid local lobe responses; continuation selects one branch and divides by its selection probability and directional sampling density.

For a perfectly smooth specular branch, continuation follows the ideal mirror direction. The source-connection calculation still uses a finite lobe to make point-source estimation practical. Rougher branches use finite power-cosine lobes. Samples on the wrong side are folded into the physical hemisphere, with a matching folded density.

Surface source connections also contain a geometric angular correction and distance spreading. For paths without diffraction, spreading uses the total path length. The comments explicitly limit the interpretation of that correction: rough paths and mixed surface/edge paths retain approximations. The smooth planar reasoning should not be generalized into a proof for arbitrary scenes.

## Edges: discovery geometry and acoustic geometry differ

A mathematical edge is difficult for a finite collection of random rays to discover. The implementation gives each authored edge a procedural strip inside an AABB. [diffraction.rint.glsl](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/acoustics/shaders/diffraction.rint.glsl) performs the actual intersection test; the AABB is only its traversal bound.

At a proxy hit, the path chooses between a straight pass and a diffraction event. Both consume an interaction from the shared budget. A straight pass preserves the physical path; it does not deposit a contribution at the artificial strip.

For diffraction, the interaction is relocated from the strip to the physical edge. Lengths, directions, visibility, and relevant sampling corrections are evaluated for that physical path. A special correction handles relocation after an initial ideal mirror reflection.

The radial discovery kernel is triangular. Its effective radius `R` is the smaller of the authored radius and the incident distance. With proxy distance `r`, it has the form:

~~~text
K(r) = 2 * max(1 - r/R, 0) / R
~~~

Its integral over the accessible half-strip is one. The capture factor also includes incident distance and projected incidence. Consequently, radius is principally a discovery/variance parameter; it is not a knob that should simply multiply acoustic strength. Finite sampling and numerical approximations can still cause radius-sensitive results, which is why radius sweeps are useful.

The directional diffraction response is a regularized, broadband scalar approximation. It uses four angular terms and an axial-mismatch term, with a 0.02-radian angular floor. **It is not an exact BTM solution or a frequency-domain UTD coefficient.**

Source connections evaluate the implemented bilateral response outside the solid wedge and may contribute signed corrections on the lit side. Continued edge events use a shadow-only response, evaluated after the next physical endpoint is known. The continuation proposal mixes source guidance with broader support and divides by the complete mixture density. Proposal probabilities affect sampling, not an independent acoustic gain.

The model does not guarantee complete field continuity across a shadow boundary.

## Distance, pressure, and sample placement

A path of total length `L` arrives at `L * sample_rate / 343`. Even after diffraction changes how spreading is accounted for, delay uses the complete physical path.

Direct sound uses inverse-distance pressure spreading, with an epsilon floor near coincident endpoints. Edge capture accounts for the incident measure; an edge-to-source connection supplies the remaining `1/b` spreading for its final leg. Replacing these distinctions with a blanket `1/L` factor would change the estimator.

Contributions are atomically added as signed pressure. Different arrival types use different interpolation kernels:

- Direct arrivals use a normalized 16-tap Lanczos kernel.
- Edge source connections use a four-tap cubic kernel.
- Surface source connections use linear fractional-delay deposition.

Contributions outside the IR window are clipped. Interpolation affects sampled amplitude and phase, so the stored sum of squares depends on more than geometric path strength. Float atomics can also make accumulation order vary even with a fixed sampling seed.

## Stereo, binaural output, and orientation

The initial listener-side arrival direction determines lateral position relative to the listener's right axis. Stereo uses equal-power left/right weights. Binaural mode adds a spherical-head interaural time delay and a bounded broadband level difference.

Both outputs still use two floats per IR sample. Binaural processing here has no measured HRTF, pinna filtering, or frequency-dependent head-shadow filter. Front/back and elevation ambiguities remain in the analytic model.

## What reuse means

[IrCache](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/acoustics/cache.rs) is a small linear-search LRU. A hit requires the same scene version, query ID, and source amplitude; source/listener displacement and both orientation axes must fall within configured tolerances.

A hit returns the previously traced snapshot with its original provenance. It does not retrace at the requested position or interpolate between several responses. Entry and GPU-byte limits bound cache ownership; eviction does not destroy a buffer still retained by a voice.

In our moving-listener example, small motion may reuse an IR. Larger motion can trigger a fresh trace. Cache tolerance is therefore part of the experiment's spatial accuracy policy.

## Evidence to read alongside the algorithm

[CPU acoustic reference tests](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/tests/acoustic_model_reference.rs) check selected mathematical properties. [Listener transport validation](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/tests/listener_backward_validation.rs) exercises the actual GPU pipeline. [Diffraction regressions](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/tests/unit/acoustics/diffraction.rs) cover specific edge cases, while [cache tests](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/tests/unit/acoustics/cache.rs) cover reuse rules.

Treat passing tests as evidence for their scenarios. Neither a plausible waveform nor a green regression suite validates every mixed path or every room. For thesis claims, record the tested geometry, budgets, seed, cache policy, reference model, and measured error.
