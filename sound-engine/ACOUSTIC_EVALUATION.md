# Acoustic transport correctness and real-time evaluation

Date: 2026-07-28

## Decision

Do not ship either baseline transport implementation unchanged.

The listener-originated design is the better foundation: it makes direct sound
deterministic, constructs the final fractional pressure IR directly, and is
substantially faster than source-forward tracing at useful ray counts. However,
the committed implementation has two surface-estimator defects and its
diffraction response has the correct broad spectral shape at approximately the
wrong absolute level.

The recommended production direction is a hybrid:

1. deterministic direct sound;
2. listener tracing with the integer golden-angle sequence and a formally
   derived path-space Jacobian/MIS estimator for early reflection;
3. deterministic BTM/BTMS or a validated filter approximation for a culled set
   of important diffraction edges;
4. a stochastic or statistical late-reverberation stage, rather than attempting
   to resolve a dense rigid-room wave response with millions of coherent rays;
5. reusable VkFFT plans, asynchronous IR construction, double-buffered spectra,
   and click-free IR crossfades.

The experimental branch proves several parts of that direction, but the
first-vertex Jacobian prototype must not be treated as the final general
multi-bounce derivation.

## Frozen comparison

| Name | Git commit | Local branch |
|---|---|---|
| Source-forward baseline | `48d9a11b96a52ad3ca082e11dfac89836fd2aca9` | `bench/source-forward` |
| Listener-backward baseline | `a5c4e408483cd91e71301f6c6b88e8d9c807f2c9` | `bench/listener-backward` |
| Corrected/optimized experiments | based on listener baseline | `bench/listener-experiments` |

The benchmark adapters use identical geometry, frequencies, sample rate, IR
length, analytical references, metrics, and warm-up policy. The only differences
are the APIs needed by each frozen implementation. Source-forward energy is set
to `4*pi` so its expected 1 m free-field level is calibrated to the listener
method's unit pressure amplitude.

No branch was pushed. The primary `acoustics-v2` checkout was not modified and
its existing untracked `sound-engine/LISTENER_BACKWARD.md` was preserved.

## Machine and budgets

- GPU: NVIDIA GeForce RTX 3060 12 GiB
- driver: 591.86
- CPU: AMD Ryzen 5 5600X, 6 cores / 12 threads
- OS: Windows 10 build 26200
- acoustic IR: 48 kHz, 16,384 samples (341.33 ms), 343 m/s
- audio block: 1,024 frames (21.33 ms hard deadline)

The acceptance targets used here are:

- audio callback p99 below 5.33 ms (25% of the hard block deadline);
- asynchronous acoustic update p99 below 4 ms, hard maximum below 8 ms;
- no FFT-plan creation, ray tracing, GPU readback, or blocking wait on the audio
  callback or game-frame thread;
- stable GPU resource use under repeated movement.

These are engineering budgets, not universal perceptual thresholds. The GPU
measurements were made without a simultaneous graphics workload. A real game
must repeat the tail-latency tests while rendering representative worst-case
frames before release.

## Mathematical references

The shared benchmark uses:

- free-field monopole pressure `exp(-ikr)/r`;
- the exact rigid-plane image-source solution;
- exact finite-order image paths for parallel planes and the damped six-wall
  geometric reference;
- 14,030 rigid shoebox images arriving inside the 341 ms IR. For rigid Neumann
  rectangular walls this is the time-domain wave-equation solution in the
  tested window; the maximum required order is 30;
- a numerically converged 131,072-segment Biot-Tolstoy-Medwin secondary-edge
  integral for the exterior rigid wedge;
- direct time-domain sine convolution and analytical two-path interference.

The rigid-room image/normal-mode equivalence and the limits of frequency-
independent wall reflection follow Allen and Berkley,
[Image method for efficiently simulating small-room acoustics](https://jontalle.web.engr.illinois.edu/Public/AllenBerkley79.pdf).
The diffraction reference follows Svensson, Fred, and Vanderkooy,
[An analytic secondary source model of edge diffraction impulse responses](https://user.faktiskt.io/svante/JASA106_2331_Svensson.pdf),
and the interactive finite-edge treatment described by Calamia and Svensson,
[Fast Time-Domain Edge-Diffraction Calculations for Interactive Acoustic Simulations](https://doi.org/10.1155/2007/63560).

Every path is deposited into adjacent samples with the same linear fractional
delay used by the production listener shader. Results are reported both against
the continuous analytical transfer and against that sampled reference, so
transport error is separated from the fractional-delay filter.

## Correctness results

### Time-domain error

Selected results are relative L1 error of the left pressure IR. Lower is better.
Unless noted, ray count is 1,048,576.

| Case | Source-forward | Listener baseline | Corrected experiment |
|---|---:|---:|---:|
| Free field, 1 m | 0.279 | 0.000034 | 0.000034 |
| Free field, 2 m | 0.348 | 0.000021 | 0.000021 |
| Free field, 5 m | 0.777 | 0.000205 | 0.000205 |
| Rigid single plane | 0.584 | 0.333 | 0.0165 |
| Absorbing single plane, pressure reflection 0.8 | 0.625 | 0.286 | 0.0146 at 262,144 rays |
| Parallel planes, six orders | 0.635 | 0.572 | 0.336 |
| Damped shoebox, six orders | 0.971 | 0.929 | 0.424 |
| BTM wedge shadow | 0.990 | 0.997 | 0.997 |
| Rigid exact shoebox, 65,536 rays | 0.992 | 0.993 | 1.503 |
| Rigid exact shoebox, 4,194,304 rays | not practical | not tested | 0.728 |

The source-forward direct estimator falls approximately as `1/r^2`, while the
pressure reference falls as `1/r`. At one million rays its measured/expected
pressure ratios are 0.934 at about 1 m, 0.652 at 2 m, and 0.223 at 5 m. It also
rounds arrivals to one time bin.

The listener baseline's direct path is deterministic and independent of ray
count. Its sampled reference is effectively exact at 1, 2, and 5 m.

### Surface estimator findings

The baseline listener shader evaluates a normalized finite lobe in
source-connection solid angle but integrates uniform listener solid angle. For
a planar first reflection, the local angular mismatch changes by
`total_path_length / source_distance` in each angular dimension. The missing
determinant is therefore:

```text
J = (total_path_length / source_distance)^2
```

Without this factor, the narrow-lobe limit for the rigid single-plane case is
0.02095 reflected pressure instead of the exact 0.24992. The experimental
factor produces 0.24994 and makes the complete direct-plus-reflection IR agree
to 1.65% relative L1 at one million rays.

This derivation is exact for the tested planar first reflection. Later vertices,
curved geometry, transmission, and rough branches need the accumulated
path-space Jacobian and a proper MIS derivation. Applying the first-vertex form
at every source connection substantially improves the rooms, but the remaining
multi-bounce error shows that it is not a complete proof.

### Fibonacci precision defect

The baseline computes:

```glsl
float phi = float(index) * golden_angle + rotation;
```

The unreduced product loses float mantissa precision at large indices. In the
single-plane test, reflected output disappears exactly at 2,097,152 rays and
remains zero through 16,777,216 rays.

The experiment performs the golden rotation modulo `2^32`:

```glsl
uint phase = ray_index * 0x61c88647u + azimuth_rotation;
float phi = TWO_PI * float(phase >> 8) / 16777216.0;
```

With this change, measured pressure is stable to a few parts in `10^6` from one
million through 16 million rays and is also effectively independent of seed.

### Dense late reflections

The corrected six-bounce shoebox improves from relative L1 0.424 at one million
rays to 0.323 at 4.2 million and 0.306 at 16.8 million. Cost rises from 2.63 ms
to 9.30 ms and 37.98 ms respectively. The rigid 30-bounce exact room is still
0.728 L1 at 4.2 million rays and costs 37.10 ms.

This is not a favorable accuracy/cost curve for real-time coherent late
reverberation. Early deterministic paths plus a statistical or modal late field
are more appropriate.

### Diffraction

At one million listener rays, the wedge response tracks the BTM frequency trend
surprisingly well but is almost a constant 52 dB too low:

| Hz | measured magnitude | BTM magnitude | error dB | phase error |
|---:|---:|---:|---:|---:|
| 125 | 0.001980 | 0.788270 | -52.00 | -3.04 deg |
| 250 | 0.001405 | 0.572413 | -52.20 | -2.76 deg |
| 500 | 0.000994 | 0.408151 | -52.27 | -1.18 deg |
| 1,000 | 0.000715 | 0.289063 | -52.13 | 0.87 deg |
| 2,000 | 0.000503 | 0.203601 | -52.14 | -1.28 deg |
| 4,000 | 0.000320 | 0.141374 | -52.91 | -8.12 deg |
| 8,000 | 0.000213 | 0.092304 | -52.74 | -4.03 deg |

The current listener proxy-radius normalization works: radii 0.20, 0.45, and
0.90 m differ by only 0.26 dB at 125 Hz and 2.8% in total pressure. The absolute
physical normalization is wrong, not the radius compensation.

Source-forward tracing produces only one nonzero diffraction bin at one million
rays, remains 32.5 dB low at 125 Hz, and changes by roughly 13.4 dB across the
same proxy radii.

The deterministic CPU BTM experiment gives:

| Edge segments | Max spectral error through 8 kHz | p99 per edge |
|---:|---:|---:|
| 512 | 101% | 0.025 ms |
| 2,048 | 2.91% | 0.113 ms |
| 8,192 | 0.499% | 0.389 ms |
| 131,072 | reference | 6.095 ms |

This makes a culled deterministic BTM/BTMS path, or a fitted BTM filter, more
promising than spending more rays on the current proxy response. Filter-based
finite-edge alternatives are also discussed in
[Filter-based first- and higher-order diffraction modeling for geometrical acoustics](https://doi.org/10.1051/aacus/2024059).

## Broadband and sine behavior

The production VkFFT convolver is numerically correct. Two unit-amplitude
arrivals separated by 24 samples have analytical gain
`2*abs(cos(pi*f*24/48000))`. GPU output from 125 Hz through 8 kHz has relative
L2 error below `4e-6`. The 1 kHz half-period case cancels to residual gain
`3.27e-5`.

The unchanged broadband transport captures only frequency behavior produced by
arrival timing and interference:

- direct/reflected/diffracted delays create phase and comb filtering;
- the diffraction delay distribution creates a plausible decreasing
  high-frequency trend;
- linear two-bin deposition itself is a fractional-delay low-pass filter. In
  the 1 m direct test it contributes -0.67 dB and 1.06 degrees at 8 kHz versus
  the continuous solution, although the sampled-reference error is negligible.

It does not model:

- frequency-dependent wall absorption or transmission: GPU upload uses only
  `absorption_bands.first()` and `transmission.first()`;
- frequency-dependent reflection phase;
- atmospheric absorption;
- wavelength-dependent rough scattering;
- low-frequency room modes beyond what a complete coherent path set happens to
  reconstruct.

Consequently, multiple sine frequencies produce different output because the
same scalar paths interfere differently, not because materials or propagation
change with frequency. This is useful broadband behavior, but it should not be
described as a band-aware acoustic material model.

In the corrected six-bounce room at one million rays, sampled magnitude errors
range from -2.89 dB at 125 Hz to +14.0 dB at 4 kHz. The 4 kHz reference is near
an interference notch and has a 99-degree phase error. Even 16.8 million rays
leaves a 1.22 complex-relative error there, illustrating the precision demanded
by coherent high-frequency cancellation.

## Performance results

### Acoustic query tail latency

One hundred and one warm queries, isolated process per case:

| Method/case | Rays | median | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|
| Listener baseline, rigid plane | 65,536 | 0.268 ms | 0.362 | 0.815 | 0.841 |
| Listener baseline, rigid plane | 1,048,576 | 0.531 ms | 0.627 | 1.418 | 1.779 |
| Listener baseline, six-wall room | 65,536 | 0.941 ms | 1.670 | 1.938 | 2.676 |
| Listener baseline, six-wall room | 1,048,576 | 5.753 ms | 6.302 | 6.397 | 6.442 |
| Source-forward, rigid plane | 1,048,576 | 3.928 ms | 4.163 | 4.791 | 5.067 |
| Source-forward, six-wall room | 1,048,576 | 11.622 ms | 12.242 | 15.630 | 17.672 |
| Corrected listener, rigid plane | 262,144 | 0.313 ms | 0.343 | 0.374 | 0.603 |
| Corrected listener, parallel planes | 1,048,576 | 2.015 ms | 2.239 | 2.274 | 2.309 |
| Corrected listener, six-wall room | 1,048,576 | 2.631 ms | 2.876 | 2.895 | 2.932 |

The listener approach is substantially more efficient at equal ray count.
The corrected sequence also avoids the degenerate duplicated directions that
make the baseline room trace slower.

Pipeline/engine construction remains a load-time operation:

- Vulkan backend: about 0.30 s;
- one acoustic pipeline/setup: about 1.0-1.4 s;
- complete `SoundEngine`: 1.68-1.86 s before reusable IR-plan optimization and
  about 2.02 s after moving the extra plan build to startup.

These operations must happen during loading or on a background initialization
path, never during an interactive frame.

### Convolution

The unoptimized production convolution path, 341 ms IR:

| Voices | p50 block | p99 block | p99 / 21.33 ms deadline |
|---:|---:|---:|---:|
| 1 | 0.39 ms | 0.75 ms | 3.5% |
| 8 | 2.66 ms | 5.07 ms | 23.8% |
| 32 | 12.98 ms | 15.68 ms | 73.5% |
| 64 | 27.98 ms | 30.97 ms | 145% |

Eight voices only just meet the chosen 25% p99 budget. Thirty-two meet the hard
deadline in isolation but leave too little headroom; sixty-four underrun.

The implementation is not actually uniformly partitioned: every 1,024-sample
block performs a 32,768-point FFT per voice. A true uniform/non-uniform
partitioned convolver, batching, and reduced CPU/GPU transfers should be the
next throughput optimization.

### Moving spatial voices

Originally, every IR registration creates a VkFFT plan synchronously:

| Operation, one active voice | Baseline latency |
|---|---:|
| IR registration | 162-171 ms |
| render after listener movement, cache hit | 166.8 ms |
| render after listener movement, cache miss | 165.5 ms |
| render after source movement | 168.0 ms |

The cache hit does not help because a fresh frequency-domain IR and plan are
still created in `render_block`.

The reusable-plan experiment binds one scratch buffer/plan at startup, transforms
each new IR there, and copies the completed spectrum into voice-owned storage:

| Operation, one active voice | Optimized latency |
|---|---:|
| IR registration | 0.56-1.11 ms |
| render after listener movement, cache hit | 0.62 ms |
| render after listener movement, cache miss | 0.88 ms |
| render after source movement | 0.86 ms |

The analytical sine validation remains unchanged. This optimization is suitable
for production after normal code review.

## Resource and lifecycle defects

1. `VkBackend.entry` is currently declared before child Vulkan owners. Rust
   therefore unloads the Vulkan library before `VkDeviceContext::drop` calls
   `destroy_device`, which hangs indefinitely. Moving `entry` to the final field
   fixes teardown; the reproduction completes in 1.9 s instead of timing out.
2. Every acoustic query inserts a new TLAS into `RtContext.tlas` and never removes
   it. The synchronous trace already guarantees completion, so the experiment
   removes and destroys the per-query TLAS immediately afterward.
3. BLAS and RT/compute pipeline registries also lack explicit removal. This is
   mainly a scene-reload/engine-reconstruction leak rather than a per-block
   issue, but ownership APIs are required for long-running editors and games.
4. IR and descriptor allocation are synchronous. Reuse/pooling is preferable
   even after the large VkFFT-plan stall is removed.

## Recommended implementation order

1. Merge the Vulkan loader-lifetime fix, integer golden-angle sequence,
   per-query TLAS destruction, and reusable IR FFT plan.
2. Move acoustic updates and IR transforms entirely off the audio callback and
   frame thread. Publish completed spectra atomically and crossfade old/new IRs.
3. Replace the first-vertex Jacobian experiment with a documented general
   path-space derivation and MIS tests. Keep the rigid and absorbing single-wall
   cases as hard acceptance gates.
4. Implement deterministic BTM/BTMS (or a benchmarked fitted filter) for a
   visibility-culled edge set. Keep the exact BTM case, radius sweep, and shadow
   continuity tests.
5. Split early coherent paths from late reverberation. Validate late decay,
   energy, and spectrum statistically instead of requiring every dense coherent
   rigid-room image to be sampled in real time.
6. Make material/air filtering genuinely band dependent and use a higher-order
   fractional-delay filter when 8 kHz phase/magnitude precision matters.
7. Replace the full-length overlap-add implementation with a true partitioned
   convolver and repeat the 1/8/32/64 voice p99 tests.
8. Run the same acoustic and audio tail benchmarks under a representative
   graphics frame workload. Reject any configuration that violates the 4 ms
   acoustic budget, 5.33 ms audio budget, or produces main-thread waits.

## Reproduction

The important examples on the local benchmark branches are:

```text
sound-engine/examples/acoustic_method_benchmark.rs
sound-engine/examples/sine_convolution_validation.rs
sound-engine/examples/auralization_realtime_benchmark.rs
sound-engine/examples/spatial_update_realtime_benchmark.rs
sound-engine/examples/btm_realtime_benchmark.rs   # experiments branch
```

Typical commands:

```text
cargo test -p sound-engine --example acoustic_method_benchmark
cargo run -p sound-engine --release --example acoustic_method_benchmark -- \
  --case rigid_single_plane --rays 1048576 --seed 7 --warm-repeats 101
cargo run -p sound-engine --release --example sine_convolution_validation
cargo run -p sound-engine --release --example auralization_realtime_benchmark
cargo run -p sound-engine --release --example spatial_update_realtime_benchmark
cargo run -p sound-engine --release --example btm_realtime_benchmark
```

Acceptance checks completed on all three branches:

```text
cargo test -p sound-engine --lib
cargo check -p sound-engine --lib --tests
```

Results:

- listener baseline: 39 library tests passed;
- source-forward baseline: 37 library tests passed;
- corrected/optimized experiment: 39 library tests passed;
- all three `cargo check` commands passed;
- workspace `--all-targets` was intentionally not used because of the known
  unrelated ungated `cpal` example failure.
