# Engine examples and validation

The workspace requires a Vulkan SDK, a ray-tracing-capable GPU, and
`glslangValidator` on `PATH`. Device playback additionally requires an audio
output device and the `playback` feature.

## Engine usage

```text
cargo run -p sound-engine --example sound_engine_direct_mode
cargo run -p sound-engine --features playback --example sound_engine_runtime_mode
cargo run -p sandbox --release -- --help
```

Both engine examples synthesize their own audio. The direct example writes
`target/sound_engine_direct_mode.wav`. The runtime example plays two spatial
tones and stops the background worker. Shared scene and signal setup lives in
`support/demo.rs`.

The sandbox separates room geometry, input handling, receiver/source poses,
rendering, and startup configuration. Use `--sound=PATH` to select a WAV,
`--list-scenes` to inspect the catalog, and `--scene=ID` to select a fixture.
Optional BRAS assets are located under `--benchmark-root=PATH`.

## Regression tests

```text
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features
```

Ordinary builds and tests skip the native C++ fault-injection harness. Run it
explicitly with `cargo test -p vkfft-rs --features native-tests`; the workspace
`--all-features` command above includes it. The FFT wrapper supports only the
single-batch, one-dimensional, power-of-two transforms used by audio convolution.
The old low-level FFT demos are covered by the retained regression tests.

Hardware tests are ignored by default. Run Vulkan tests serially, including
across separate test processes:

```text
cargo test -p sound-engine --test listener_backward_validation short_direct_smoke -- --ignored --test-threads=1
cargo test -p sound-engine --test sine_convolution_validation -- --ignored --test-threads=1
cargo test -p sound-engine --test moving_ir_transition_validation -- --ignored --test-threads=1
cargo test -p sound-engine --lib --features playback audit_runtime_handoff_status_failure_recovery_and_drop -- --ignored --test-threads=1
```

These checks cover direct transport, analytical convolution, overlapping IR
transitions, and runtime ownership recovery. Longer transport tests also live
in `listener_backward_validation`; select them explicitly when needed.

## Benchmarks

Run benchmarks with `cargo run --release -p sound-engine --example NAME`.

| Purpose | Example |
| --- | --- |
| Analytical acoustic references and timing | `acoustic_method_benchmark` |
| CGA02 room/baffle geometry | `bell_labs_box` |
| BRAS RS5/RS6 measured geometry | `bras_diffraction` |
| File-driven BRAS room fixtures | `bras_extended` |
| Edge, doorway, and radius validation | `gpu_ir_diffraction_edge` |
| Direct FIR versus GPU FFT cost | `convolution_method_benchmark` |
| Render-block latency | `auralization_realtime_benchmark` |
| Moving-listener IR refresh cost | `spatial_update_realtime_benchmark` |
| Scene-update and convolution costs | `runtime_costs` |
| Offline BTM mathematical reference | `btm_realtime_benchmark` |

`support/geometry.rs` and `support/bell_box.rs` define benchmark fixtures.
`support/timing.rs` supplies latency quantiles, `support/provenance.rs` records
source revisions, and `acoustic_method/reference.rs` contains independent
mathematical references. None of these helpers enters the runtime library.

The Python tools in `../scripts` prepare measured fixtures and report raw
benchmark output. `run_current_benchmarks.py` drives recorded Windows benchmark
stages; use a fresh output directory to preserve provenance. The engine's current
transport is scalar and omnidirectional. Materials use a scalar `absorption`
fraction, `scattering`, and `roughness`; surfaces absorb or reflect sound and
do not transmit it. Scripts for retired UTD and directional
source experiments are preserved in Git history.

Maintain the Python scripts with their pinned development tools:

```text
uv run --project sound-engine/scripts ruff check sound-engine/scripts
uv run --project sound-engine/scripts ruff format --check sound-engine/scripts
uv run --project sound-engine/scripts python -m unittest discover -s sound-engine/scripts
```

The two benchmark runners share process recording and provenance checks through
`benchmark_runner.py`. Each run owns its output directory and command environment.
Use `run_current_benchmarks.py STAGE --output PATH`; extended BRAS runs and reports
accept `--base PATH` for the directory containing prepared fixtures and results.

Shader includes are recorded by the build script. Rebuild after changing GLSL;
stale executables reject a changed shader tree. Report ray count, seed, GPU,
source revision, and model assumptions alongside measurements. Passing the
regression tests does not establish physical accuracy for every scene.
