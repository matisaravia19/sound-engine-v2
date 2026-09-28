# Memory, Vulkan lifetime and safety audit — 2026-09-28

The audit identified and repaired concrete resource-lifetime, synchronization,
validation and shutdown defects. These defects can cause resource accumulation,
invalid Vulkan use or out-of-bounds GPU accesses. The reported system-wide driver
crash was not reproduced during the bounded validation runs; its exact cause is
not established, and this report is not a guarantee that every defect is gone.

## Scope

Reviewed the first-party Rust runtime, Vulkan allocation/submission/compute/RT
code, scene and acoustic resource ownership, GPU shader indexing contracts,
convolution/VkFFT FFI, playback shutdown, and locked Rust dependencies. All
workspace targets and features, including examples and the sandbox, were checked.
Existing benchmark outputs, scripts and unrelated working-tree edits were
preserved. The obsolete `docs/` directory was not updated.

This was not an exhaustive audit of the vendored VkFFT/glslang/VMA implementations,
the graphics driver, or third-party audio/windowing libraries.

## Findings and repairs

| Area | Defect and resulting change |
| --- | --- |
| Device lifetime | A buffer retained VMA without retaining the Vulkan device and dynamically loaded entry points. Buffers now retain an allocator owner that keeps the device and loader alive through destruction. Device metadata and buffer bounds/handles are exposed read-only so safe code cannot replace an owner's destruction handle or falsify allocation bounds. |
| Scene lifetime | Repeated scene rebuilds retained obsolete BLAS allocations in the RT registry. Scoped BLAS/TLAS owners now remove registry entries on replacement, scene destruction and error paths. TLAS ownership is released before referenced scene BLAS ownership. |
| Geometry replacement | Replacing a mesh under the same application ID reused its old uploaded geometry. Per-mesh revisions now invalidate those uploads, and removed meshes release their resources. This also prevents scene metadata and device-addressed geometry from describing different revisions. |
| Query and pipeline lifetime | Per-query TLAS cleanup occurred only on success. A scoped owner now covers all exits. Acoustic and convolution pipelines are released when their high-level owners disappear, even when the backend remains shared. Identical shader modules are reused instead of accumulating on each engine recreation. |
| Submission failure | Recording, fence creation, command-buffer finalization and submission failures could strand submission resources. A pending-submission guard recycles unsubmitted buffers, descriptors and fences. Tracking is acquired before submitting so successful work cannot become untracked because of a later lock failure. |
| Wait and destruction failure | Cleanup no longer panics on poisoned locks or reported device loss. All tracked submissions are reclaimed before dependent pools are destroyed. A transient fence-wait allocation failure does not let synchronous callers immediately destroy resources still in use: waiting continues until completion/device loss, then the error is returned. This remains an unbounded completion wait, not a watchdog timeout. |
| GPU memory dependencies | CPU fence waits were relied upon without a complete device memory dependency between separate submissions. Entry barriers now order previous writes before subsequent reads/writes; exit barriers make writes available for host readback. Dependencies within a submission remain explicit. |
| Descriptor synchronization | Allocation and freeing from compute/RT descriptor pools now share external synchronization. Pool growth is limited to exhaustion/fragmentation errors; other driver errors propagate. New pools are owned before their first fallible allocation. |
| Partial construction | Pipeline/layout/descriptor-layout construction uses an error-path owner, including partial handles returned by Vulkan. Instance/device/command-pool setup also cleans up failures. Stored queried Vulkan properties no longer retain dangling `pNext` pointers. |
| RT shader binding table | Records are aligned relative to the buffer's actual device address, with padding reserved in the allocation. Pipeline validation checks ray-generation group count, contiguous SBT regions, stride and recursion limits. |
| Dispatch contracts | Compute and RT dispatch validate complete descriptor initialization, binding types, push-constant alignment/ranges/stages, and dispatch limits. Raw-handle recording, asynchronous submission and geometry-build entry points now have explicit unsafe contracts. |
| Input validation | Geometry validation rejects incomplete triangles, out-of-range indices and non-finite vertices. Instance validation rejects invalid transforms and overflowing 24-bit fields. Allocation, transfer, fill, descriptor and FFT sizes are checked before use. Required validation now runs in release builds as well as debug builds. |
| Device ownership and mapping | High-level scene, IR and transfer paths reject buffers from a different allocator/device. Mapped host reads/writes are serialized per allocation. Empty transfers are no-ops; copies and fills validate buffer usage and bounds. |
| Convolution | VkFFT initialization receives an owned non-null fence and exclusive queue/command-pool access. Plan construction/destruction serializes process-global compiler use. Voice CPU state and IR spectrum dimensions must match the convolver, preventing a short or foreign IR buffer from reaching the multiply shader. |
| VkFFT ABI and API | The wrapper checks handles and byte capacities before FFI, uses 64-bit non-dispatchable Vulkan handles, and verifies its maximum-dimension ABI at compile time. C++ application allocation uses `nothrow`. Plan construction/recording explicitly requires unsafe lifetime/synchronization guarantees; in-tree callers were updated. |
| Cache limits | Multiple exact queries in one quantized key bucket could become unreachable from eviction and escape entry/byte limits. Eviction now retires the entire selected bucket and its byte accounting. GPU byte accounting uses actual retained buffer size. |
| Playback shutdown | Stop notification is paired with the queue predicate mutex, closing a missed-wakeup window for a blocked producer. Device errors request stop and wake producers. Queue/block/channel validation and overflow checks remain active in release builds. |

## Validation

Tests were run serially on the GPU, using the installed NVIDIA RTX 3060 and
Khronos validation layer. No deliberate device-loss, out-of-memory or watchdog
fault injection was performed. The loader trace confirms that the validation
layer was inserted; synchronization validation was requested with
`VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT`.

| Check | Result |
| --- | --- |
| `cargo check --workspace --all-targets --all-features --offline` | Passed. |
| `cargo test --workspace --all-features --lib --offline` | 78 passed: 77 engine tests and 1 VkFFT test; 10 hardware tests excluded from this ordinary run. |
| Same library test command with `--release` | 78 passed; 10 hardware tests excluded. |
| Six `audit_` hardware regression tests | Passed in debug and release with validation requested. |
| Existing staging-growth/explicit-release hardware test | Passed with validation. Exercises transfers across staging-buffer growth. |
| Existing engine scalar-update/voice-refresh integration test | Passed with validation. Exercises scene RT, moving queries and convolution together. |
| `cargo clippy --workspace --all-targets --all-features --offline` | Completed with style/API-shape warnings; no compiler errors. This is not a warning-free result or a proof of safety. |
| `git diff --check` | Passed. |

The six audit hardware tests exercise:

- Buffer/snapshot/IR owners outliving the backend, followed by final device release.
- Empty transfers and rejection of invalid allocations.
- Thirty-two intentional recording failures with stable command-buffer accounting.
- Eight same-ID mesh replacements with stable BLAS/TLAS registry counts.
- Repeated acoustic queries and engine-pipeline recreation without retained TLAS/pipelines.
- VkFFT initialization, known impulse spectrum, pipeline release and rejection of malformed voice/IR dimensions.

The runs reported no application Vulkan validation errors. Two long diffraction
physics tests were left out of this safety run; no long benchmark soak or peak
VRAM/RSS trend study was performed. Registry-count tests demonstrate cleanup for
the covered resources, not the absence of every possible driver allocation leak.

Evidence is saved under `output/memory-audit-20260928/`, including debug/release
test logs, Vulkan validation logs, the earlier loader trace, and Clippy JSON.
To rerun the focused hardware tests in PowerShell:

```powershell
$env:VK_INSTANCE_LAYERS = 'VK_LAYER_KHRONOS_validation'
$env:VK_LAYER_ENABLES = 'VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT'
cargo test -p sound-engine --all-features --lib --release --offline audit_ -- --ignored --test-threads=1 --nocapture
cargo test -p sound-engine --all-features --lib --release --offline staging_growth_and_explicit_release_preserve_transfers -- --ignored --test-threads=1 --nocapture
cargo test -p sound-engine --all-features --lib --release --offline scalar_updates_preserve_pressure_and_refresh_all_voices -- --ignored --test-threads=1 --nocapture
```

## Dependency check

`cargo-audit` was unavailable. A snapshot of the complete public
[RustSec advisory database](https://github.com/RustSec/advisory-db) was downloaded
and compared locally against all 125 locked registry package versions. The local
scanner supports the comparison ranges needed by the matching advisories and
fails on unsupported syntax. It is saved alongside its JSON results for review;
this was not a `cargo audit` run.

No applicable vulnerability advisory was found. One applicable informational
notice remains: [`paste` 1.0.15 is unmaintained](https://rustsec.org/advisories/RUSTSEC-2024-0436.html),
through `raylib` in the sandbox. This notice does not establish a memory flaw or
explain the driver crash. No broad dependency upgrade was made as part of the
runtime repair. The scan does not cover advisories outside RustSec or vendored
native libraries.

Database archive SHA-256:
`50711c520fae8cc152439d43d75bfd83e81cb4fee6edf82377b0b1fd103a1435`.
No source code or dependency inventory was uploaded for the completed scan.

## Compatibility and remaining limits

- Callers of raw GPU command recording, asynchronous submission, BLAS builders
  and VkFFT plan construction/recording must now use `unsafe` and uphold the
  documented contracts. Workspace callers and examples compile with these changes.
- Device handles and buffer size/handle metadata can still be read with the same
  field syntax, but cannot be reassigned through the owning wrappers.
- Conservative submission barriers may affect benchmark timings. Rerun performance
  comparisons on a freshly rebuilt binary; existing binaries do not include fixes.
- Application-created raw handles, shader programs and custom GPU buffers still
  require valid lifetime, layout and synchronization proofs at unsafe boundaries.
  No claim is made that arbitrary external Vulkan use is validated completely.
- The loader reported installed Overwolf/OBS implicit layers advertising Vulkan
  1.2 while the application requests 1.3. System layers/settings were left intact;
  there is no evidence here establishing them as the crash cause.
- Large trace workloads can still exceed available memory or execution-time
  limits. This repair did not add workload chunking or change OS watchdog settings.
  The next diagnostic evidence for any remaining system crash is the exact failing
  workload plus its validation/device-loss log, followed by a controlled soak run.

The synchronization changes follow Vulkan's separate requirements for
[memory dependencies](https://docs.vulkan.org/spec/latest/chapters/synchronization.html)
and [queue/device completion](https://docs.vulkan.org/spec/latest/chapters/devsandqueues.html).
