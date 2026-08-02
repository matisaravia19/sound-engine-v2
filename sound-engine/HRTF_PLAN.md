# Deferred measured-HRTF plan

This document records possible follow-up work. The analytic binaural estimator
does not add HRTF assets, HRTF dependencies, feature flags, or HRTF runtime
paths.

## Viability conclusion

A measured HRTF is viable if acoustic paths are first accumulated into a small
set of directional impulse responses and decoded to a final stereo IR after ray
tracing. Applying every tap of a measured HRIR directly for every completed ray
would multiply atomic writes by the filter length and is not suitable for the
current 65,536-to-1,000,000-ray workloads.

The preferred experiment is therefore:

1. Deposit each completed path into one or several listener-local directional
   IR bins.
2. Transform all bins with a batched GPU FFT.
3. Multiply each directional spectrum by its left and right HRTF spectra and
   sum the results.
4. Inverse-transform once into the existing packed stereo `GpuIrSnapshot`.

The final snapshot remains compatible with the existing cache, IR registration,
and audio callback. All additional work stays on the IR construction path.

## Generic data and preprocessing

Use the SADIE II D2 KEMAR measurement as the first bundled generic profile. The
database provides full-sphere HRIR data at 48 kHz and is licensed under Apache
2.0. Any derived asset must retain the University of York copyright, license,
required attribution, source version, and a reproducible checksum.

- Database: <https://www.york.ac.uk/sadie-project/database.html>
- Archive: <https://zenodo.org/records/10886409>
- SOFA convention: <https://www.sofacoustics.org/mediawiki/index.php/SimpleFreeFieldHRIR>

An offline tool should validate SOFA metadata, honor `Data.Delay`, remove only
dataset-wide common measurement latency, preserve interaural and
direction-relative delays, normalize diffuse-field average power, and emit a
small engine-native binary. Runtime SOFA parsing is deferred until the bundled
path is proven; custom SOFA files can later use the same validated intermediate
representation.

Precompute cubemap direction grids with face resolutions two, four, and eight,
giving 24, 96, and 384 filters. At 256 taps and two `f32` ears, the time-domain
assets require approximately 48 KiB, 192 KiB, and 768 KiB respectively.

## GPU decode architecture

Ray tracing writes a mono `direction_count * ir_num_samples` buffer. Listener
orientation converts every initial arrival direction into local space before a
constant-time cubemap lookup. Two deposition policies should be compared:

- Nearest direction: one bin and no interpolation overhead, with possible
  direction-switch discontinuities.
- Four-bin interpolation: bilinear cubemap weights that sum to one, producing
  smoother motion at up to four times the deposition atomics.

Extend the local VkFFT bridge and builder with a checked `numberBatches` option.
At engine initialization, transform and retain the HRIR kernels for the active
FFT size. For each IR update, batch-transform the directional IRs, run one
frequency-domain reduction shader that forms left and right spectra, then run a
single inverse FFT. Reuse all scratch allocations and plans.

For a 16,384-sample acoustic IR and 256-tap HRIR, use an FFT size of 32,768. A
96-direction complex signal buffer is about 24 MiB and its two-ear kernel is
about 48 MiB. A 384-direction configuration raises those values to roughly
96 MiB and 192 MiB, so memory is likely to limit the densest mode.

Convolution produces `ir_num_samples + hrir_taps - 1` samples. Preserve the
public IR-length contract by returning the first `ir_num_samples` values and
reporting the cropped tail in diagnostics. A future API may instead reserve the
HRIR tail explicitly.

## Correctness and performance study

Build a feature-gated benchmark only after the analytic estimator is stable.
Compare Stereo with 24, 96, and 384 directions, using nearest and four-bin
deposition at 48 kHz with 16,384 output samples. Measure 65,536 and 1,048,576
rays across direct, reflection, diffraction, and six-bounce room scenes.

Correctness gates:

- Batched transforms agree with a CPU `realfft` reference.
- A direct impulse at an exact grid direction reproduces its shifted HRIR within
  `1e-4` relative error.
- Reflected and diffracted paths use the initial listener-side direction.
- Directional weights sum to one and IR-tail cropping is explicit.
- Report ITD, ILD, and log-spectral error against held-out dense KEMAR
  directions, plus render rotating broadband and room-response WAVs.

Performance reporting uses 101 warm queries and includes p50, p95, p99, setup
time, phase-level GPU timing, and peak temporary memory. The measured-HRTF path
is a production candidate only if the 96-direction nearest mode keeps the
complete IR update at or below 4 ms p95 without adding work or synchronization
to the audio callback. If it fails, retain the analytic estimator and record
whether ray deposition, batched transforms, spectral reduction, or memory
traffic is the limiting phase.
