# Pipeline Specification

## 1. End-to-End Pipeline

```text
Scene Update
  -> Play Sound Events
  -> GPU Scene Sync
  -> IR Cache Query
  -> Acoustic Path Simulation
  -> Path/Event Accumulation
  -> IR Generation
  -> IR Publication
  -> Partitioned Convolution
  -> Spatial Rendering
  -> Output Audio
```

## 2. Scene Preprocessing

Inputs:

- Triangle meshes.
- Material acoustic parameters.
- Active sound events and listener state.

Outputs:

- GPU geometry buffers.
- Material lookup tables.
- Edge/corner datasets for diffraction.
- Acceleration structures (BLAS/TLAS).

Scope note:

- No mesh-instance layer in thesis v1.

## 3. Acoustic Path Simulation Stage

Per active sound event and listener pair (or batched):

1. Launch `N` rays from source distribution.
2. Trace up to `max_bounces`.
3. For each hit, evaluate material reflection/absorption model.
4. Spawn continuation rays based on stochastic policy.
5. Test listener interception criteria.

Recorded event fields:

- `arrival_time_seconds`
- `linear_gain` or `band_gain[B]`
- `incoming_direction`
- `path_order`
- `effect_type` (`reflection`, `diffraction`, `transmission`)

## 4. Reflection Model (Baseline)

Per bounce:

- Geometric spreading attenuation (`1/r` or `1/r^2` according to chosen convention).
- Angle/frequency-dependent reflection from material coefficients.
- Energy cutoff and max path length constraints.

## 5. Diffraction Model (Main Extension)

Suggested implementation path:

1. Precompute wedge/edge candidates from scene topology.
2. During tracing, detect shadow-zone transitions or near-edge interactions.
3. Emit secondary diffraction events using UTD-inspired gain estimate.
4. Inject events into the same accumulator as reflection arrivals.

Design constraints:

- Runtime toggle (`enabled/disabled`).
- Budget-controlled candidate evaluation per tick.
- Clear separation from reflection code path.

## 6. Transmission Model (Optional Extension)

Per crossing event:

- Evaluate material transmission coefficient (possibly multi-band).
- Add extra propagation through medium/thickness factor.
- Emit transmitted event with filtered gain and direction metadata.

## 7. Event Accumulation Stage

Accumulator responsibilities:

- Bin events by arrival time at IR sample resolution.
- Sum contributions with optional band separation.
- Keep deterministic floating-point accumulation policy when reproducibility is required.

Optional enhancements:

- Temporal smoothing across simulation ticks.
- Outlier rejection for unstable stochastic spikes.

## 8. IR Generation Stage

IR builder outputs:

- Mono IR per source-listener pair, or
- Stereo/binaural-ready directional IR representation.

Post-processing:

- Normalization/limiting.
- Early/late split handling.
- Optional noise floor shaping for late tail stability.

## 9. IR Cache Stage

- Key cache entries by quantized source/listener positions.
- Reuse IR if both positions remain inside reuse tolerance.
- Invalidate entries when scene version changes.
- Crossfade when swapping between reused and recomputed IR.

## 10. IR Publication Contract

Published IR object must be:

- Immutable after publish.
- Timestamped and versioned.
- Safe for lock-free consumption by audio thread.

If no fresh IR is available:

- audio thread reuses last valid IR.

## 11. Auralization Stage

Use partitioned convolution:

- Small head partitions for low latency.
- Larger tail partitions for efficiency.
- Overlap-save or overlap-add implementation, chosen once and documented.

Dataflow:

1. Fetch source audio block.
2. Convolve using active IR partitions.
3. Apply direct path (if modeled separately).
4. Apply spatialization/HRTF.
5. Write output block.

## 12. Spatial Rendering

Modes:

- Stereo panning (lightweight).
- Binaural rendering (HRTF convolution stage).

Direction data source:

- Use arrival direction metadata from simulation when available.
- Fallback to source-listener geometric direction.

## 13. Validation Metrics

Track per build/run:

- IR energy decay profile and EDT/RT proxies.
- Delay accuracy for canonical scenes.
- Audio callback overrun count.
- Simulation tick duration and dropped tick count.
- Diffraction contribution statistics (event count, mean gain, CPU/GPU cost).
