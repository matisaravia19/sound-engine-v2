# Tests and experiments: knowing what a result establishes

[Guide index](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/codebase-guide/README.md)

The repository contains several kinds of evidence: CPU contract tests, GPU integration tests, native failure tests, timing benchmarks, and comparisons with acoustic references. They answer different questions.

For the doorway example, a scene test can establish that the opening and edge are represented correctly. A transport test can examine the resulting arrivals. A moving-IR test can examine transition continuity. A playback run can expose missed deadlines. None of those observations alone answers all four questions.

## Start with the two usage examples

The retained examples of ordinary engine use are small enough to read first:

~~~text
cargo run -p sound-engine --example sound_engine_direct_mode
cargo run -p sound-engine --features playback --example sound_engine_runtime_mode
~~~

[Direct mode](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/examples/sound_engine_direct_mode.rs) synthesizes its own dry tone, renders blocks, and writes `target/sound_engine_direct_mode.wav`. [Runtime mode](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/examples/sound_engine_runtime_mode.rs) plays synthesized spatial tones through the background worker and then stops it.

Both share scene/signal setup in [support/demo.rs](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/examples/support/demo.rs). They do not require an external recording.

Run commands in this chapter from the repository root. Engine execution requires a suitable Vulkan device; device playback also requires a working audio output. Building requires the Vulkan SDK and C++ tools, with `glslangValidator` available for acoustic shader compilation. Building the sandbox also involves raylib's native build dependencies, including CMake. Even CPU-only test execution must first build the linked crates.

## Unit tests are separate files but retain module access

Private unit tests live under each crate's `tests/unit` tree and are attached to their source modules under `#[cfg(test)]` with a path attribute. This keeps test bodies out of the production source tree while preserving access to private contracts.

Cargo does not run each of those files as a separate integration-test binary. Public integration tests, such as [acoustic_model_reference.rs](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/tests/acoustic_model_reference.rs), live directly under `sound-engine/tests`.

A practical starting set is:

~~~text
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets -- --test-threads=1
~~~

The test command leaves ignored hardware tests unexecuted. The ordinary feature set also excludes the optional VkFFT native fault-injection harness. `--all-features` includes that harness; it is more than a switch for playback.

Formatting and Clippy establish useful consistency and static checks. They do not establish acoustic correctness or real-time performance.

## Select GPU checks deliberately

Hardware tests are ignored by default. Run them serially, including across separate processes:

~~~text
cargo test -p sound-engine --test listener_backward_validation short_direct_smoke -- --ignored --test-threads=1
cargo test -p sound-engine --test sine_convolution_validation -- --ignored --test-threads=1
cargo test -p sound-engine --test moving_ir_transition_validation -- --ignored --test-threads=1
cargo test -p sound-engine --lib --features playback audit_runtime_handoff_status_failure_recovery_and_drop -- --ignored --test-threads=1
~~~

These cover a short direct-transport case, numerical convolution, overlapping filter transitions, and runtime ownership/failure recovery. Longer transport cases in [listener_backward_validation.rs](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/tests/listener_backward_validation.rs) should be selected for the behavior being investigated.

Vulkan validation is a separate configuration choice. Enable the SDK's validation layer and synchronization validation when checking resource ordering, and confirm in the run output that the intended configuration was active. Merely passing a test without validation does not check the same set of contracts.

The [GPU unit-test tree](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/tests/unit/gpu) contains lower-level lifetime, acceleration-structure, descriptor, and submission audits. A numerical result can look plausible despite an invalid Vulkan operation, so these audits complement numerical comparisons.

## Native and Python checks

Run the VkFFT fault-injection harness explicitly:

~~~text
cargo test -p vkfft-rs --features native-tests
~~~

It exercises native cleanup and error paths, including failures that would be difficult to induce reliably on a real GPU. It does not replace actual GPU FFT/convolution comparisons.

The Python tools have pinned project tooling:

~~~text
uv run --project sound-engine/scripts ruff check sound-engine/scripts
uv run --project sound-engine/scripts ruff format --check sound-engine/scripts
uv run --project sound-engine/scripts python -m unittest discover -s sound-engine/scripts
~~~

Those tests cover runner behavior and provenance checks. They do not re-run every measured acoustic comparison.

## The retained benchmark programs

Run an individual benchmark with `cargo run --release -p sound-engine --example NAME`. Inspect its arguments and required fixture files before starting a long run.

| Example | Main question |
| --- | --- |
| [acoustic_method_benchmark](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/examples/acoustic_method_benchmark.rs) | How do acoustic outputs and costs compare with independent mathematical references? |
| [gpu_ir_diffraction_edge](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/examples/gpu_ir_diffraction_edge.rs) | How do edge, doorway, and capture-radius cases behave? |
| [bell_labs_box](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/examples/bell_labs_box.rs) | How does the reconstructed room/baffle case behave? |
| [bras_diffraction](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/examples/bras_diffraction.rs) | How do the RS5/RS6 fixtures compare with prepared reference data? |
| [bras_extended](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/examples/bras_extended.rs) | How do file-driven BRAS scene/position cases behave? |
| [convolution_method_benchmark](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/examples/convolution_method_benchmark.rs) | How do direct FIR and GPU FFT costs differ? |
| [auralization_realtime_benchmark](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/examples/auralization_realtime_benchmark.rs) | What is the distribution of audio-block rendering times? |
| [spatial_update_realtime_benchmark](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/examples/spatial_update_realtime_benchmark.rs) | What does refreshing moving-listener responses cost? |
| [runtime_costs](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/examples/runtime_costs.rs) | Where are scene, trace, and convolution costs incurred? |
| [btm_realtime_benchmark](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/examples/btm_realtime_benchmark.rs) | What is the cost of an offline BTM mathematical reference? |

The CPU FIR and BTM reference code are experimental comparators. They are not alternate production backends selected by `SoundEngine`.

[acoustic_method/reference.rs](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/examples/acoustic_method/reference.rs) contains independent reference calculations. [support/timing.rs](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/examples/support/timing.rs) provides quantiles, and [support/provenance.rs](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/examples/support/provenance.rs) records revisions. These helpers belong to experiments rather than the runtime library.

## Preserve a run, not just its final plot

[run_current_benchmarks.py](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/scripts/run_current_benchmarks.py) coordinates Windows benchmark stages: `build`, `accuracy`, `runtime`, `playback`, `checks`, and `reports`. Its shared [benchmark_runner.py](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/scripts/benchmark_runner.py) records commands, timeouts, source/build provenance, and process output.

Begin with a fresh result directory. For example:

~~~text
uv run --project sound-engine/scripts python sound-engine/scripts/run_current_benchmarks.py build --output output/guide-benchmark-run
uv run --project sound-engine/scripts python sound-engine/scripts/run_current_benchmarks.py runtime --output output/guide-benchmark-run
~~~

The second stage verifies the recorded build before using it. Other stages have their own data and device requirements; the reports stage expects prior results and the applicable reference/calibration inputs.

Rebuild after GLSL changes. Shader source and includes are tracked, and stale executables reject a changed shader tree. Copying an old executable beside new shaders is not a valid way to reproduce the new model.

Record the code revision, configuration, seed, ray count, device/driver, fixture sources, calibration, and relevant errors with each result. Fixed seeds help comparisons, but parallel floating-point accumulation can still vary in its final bits.

## Keep timing claims precise

Cold backend/plan construction, warm tracing, a cache hit, IR preparation, block convolution, and spatial-refresh age are different measurements. They should not share an unlabeled “latency” axis.

A block's duration is its frame count divided by sample rate. Rendering faster than that duration on average is useful evidence, but occasional long blocks can still cause underruns. Queueing and device buffering also contribute to what the listener experiences.

Similarly, a successful listener update acknowledges a state change. It does not imply that every playing voice has already received the new filter or that already queued audio has reached the device.

For thesis figures, explain the measured interval and what work it includes. For model claims, connect the observation to a defined fixture and a stated approximation. This makes the result interpretable even after the implementation changes.
