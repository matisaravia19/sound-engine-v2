# Continued review — 2026-09-28

The active objective remains: continue reviewing the code, repair defects,
improve performance where supported by evidence, and keep the implementation
readable and organized. This records verified progress, not completion of the
whole review or a guarantee that all defects have been found.

## Completed in this pass

- **Scene transactions:** mesh data is now immutable shared storage, so atomic
  scene edits do not clone every vertex/index array. Mesh content and its revision
  are stored together instead of in parallel maps. Empty updates return before
  cloning, and revision exhaustion returns an error instead of silently stopping
  cache/GPU invalidation. Tests cover rollback, replacement, snapshot independence,
  geometry sharing and version exhaustion.
- **Convolution submissions:** removed the GPU clear before a full signal upload.
  FFT output copying now shares the FFT submission. GPU IR padding initialization
  and prefix copying also share a submission, with an explicit transfer dependency.
  Buffer checks are centralized in reusable recorded-transfer helpers. CPU-source
  IR preparation now needs three submissions instead of four; GPU-source IR
  preparation needs two instead of four; each audio block needs three instead of four.
- **Convolution correctness:** a GPU IR with a different sample rate is rejected.
  A numerical hardware regression compares CPU- and GPU-sourced IR convolution
  against a direct CPU reference across consecutive audio blocks. It first fills
  the scratch allocation with a longer IR to detect stale padding after replacement.
- **IR cache:** reproduced two failures before fixing them. Re-inserting an exact
  query returned the older snapshot and retained both allocations; insertion now
  replaces it and updates byte accounting. Snapshots with incompatible scene/query
  metadata are rejected. The quantized-bucket eviction regression still exercises
  distinct queries sharing one bucket.
- **Organization:** scene and convolution tests moved to adjacent test modules.
  Acoustic and core cache configuration now share one type while preserving both
  public import paths. Runtime checks use `validate!`; `debug_validate!` remains a
  compatibility alias with the same always-on behavior.

## Additional runtime and asset review

- **Audio assets:** mono and interleaved constructors reject non-finite samples.
  Downmixing uses wider accumulation and compacts the owned input buffer in place;
  interpolation also avoids overflow when the input samples themselves are finite.
  Resampling computes output capacity with checked integer arithmetic and uses
  fallible allocation. Three regressions failed before the corresponding fixes.
- **Identifiers:** sound, source, voice and registered-IR IDs now fail on exhaustion
  instead of saturating and replacing an existing object. `SoundBank::default()`
  now has the same initial ID sequence as `new()`. **API change:**
  `SoundBank::insert` returns `SoundResult<SoundId>`; workspace callers were updated.
- **Voice ownership:** active voices retain the immutable dry asset, so removing
  its sound-bank entry no longer interrupts playback. The playback entry point
  also checks sample rate even when callers inserted assets through `sound_bank_mut`.
- **Runtime shutdown:** a stop handle wakes producers waiting on a full playback
  queue before shutdown joins the worker. Idle workers periodically observe the
  stop flag, including device failure without an accompanying engine command.
  Thread creation failures are returned as errors.
- **Audio callback:** drains directly into the device output, eliminating the
  growable scratch buffer. A contended queue yields silence without waiting or
  consuming samples; contention cannot incorrectly mark playback complete.
  Completion is observed before draining so a concurrent final enqueue cannot
  be mistaken for an already-finished queue.
  Non-finite samples are silenced before device-format conversion. Exceptional
  shutdown/error paths still take locks; this is not a fully lock-free audio system.
- **Cache ownership:** even a cache hit rejects a different backend before
  returning a GPU allocation owned by the original backend.
- **Shader reloads:** verified bundled sources reuse the module without launching
  another compiler. Every bundled file/include is still compared with the build
  manifest on each load; custom source files still compile before SPIR-V deduplication.
  Tests check both source-edit rejection and the actual compiler invocation count.
- **Organization:** lifecycle and shader tests are at the ends of their modules;
  checked ID allocation is a small shared core helper.

The bundled-shader reload benchmark measured a median **94,949.550 µs before**
and **2,716.975 µs after** (about 35 times faster for this setup operation). Each
sample is the median of five windows of four loads after four warm-up loads.
This does not describe ray-trace or whole-engine speed. Logs are in
`output/review-runtime-20260928/shader-before.log` and `shader-after.log`.

## Lifetime, diagnostic retention and query setup

- **BLAS ownership:** resolving TLAS instances now acquires strong references while
  holding the BLAS registry lock. Each completed TLAS retains each distinct BLAS
  until destruction. Removing a scene's registry owner can no longer free geometry
  addressed by an in-progress TLAS build or another existing TLAS. A hardware test
  drops the original BLAS owner and input mesh, verifies retained ownership, then
  verifies release after the final TLAS is destroyed.
- **Explicit destruction:** `RtContext::destroy_tlas` is now `unsafe`: callers must
  own the ID and synchronize all use. This prevents the safe API from promising
  safety while allowing concurrent destruction of another owner's live structure.
- **Query TLAS reuse:** the acoustic pipeline retains one query TLAS for unchanged
  geometry. Its key uses stable scene-TLAS and diffraction-BLAS IDs, so separate
  scene managers with equal version numbers cannot alias. Tests alternate between
  obstructed and unobstructed scenes, change object activity, replace procedural
  edges, check numerical output, and check bounded registry counts and cleanup.
- **IR submission batching:** initialization and ray tracing share one submission,
  with a transfer-to-ray-tracing read/write dependency for float atomic updates.
  Validation exposed `VUID-vkCmdPipelineBarrier-dstAccessMask-06257` in the existing
  broad transfer helper when it was first used on this path. That helper now uses
  legal generic memory masks, and the IR path uses a dedicated narrow barrier.
  Both are exercised by the passing validation regressions.
- **Runtime diagnostics:** moved into `engine/diagnostics.rs` with adjacent tests.
  CSV and WAV outputs stream on a dedicated writer thread. Queue capacity accounts
  for an approximately 8 MiB payload budget including producer/writer records,
  with at most 1,024 queued blocks, plus bounded channel/file-buffer overhead.
  A full queue stops recording at a contiguous prefix and reports omitted blocks;
  it does not wait for disk space in the rendering loop or silently splice audio.
  WAV output stops at complete blocks before RIFF's 32-bit size limit. Metadata
  reports truncation and write failures. Canonicalized destination collisions are
  rejected before files are truncated. Eight CPU tests cover queue saturation,
  disconnected writers, file errors, frame-preserving limits and complete output.

Diagnostics still copy requested audio blocks on the producer thread. They are
optional and not a hard-real-time or durability guarantee. Startup opens outputs;
shutdown joins the writer and flushes buffered data. No live audio device was
needed for the file/queue tests.

The bounded empty-scene IR fixture (one direct ray, 1,024 stereo IR samples,
unchanged geometry, IR recomputed on every operation) measured a median
**284.250 µs before** and **143.106 µs after**. That is about a 50% reduction for
this fixture, not a claim about complex-scene tracing throughput. Four warm-up
queries precede five windows of 16 queries. The earlier executable and both logs
are preserved in `output/review-diagnostics-20260928/`.

## Input and compiler hardening

- **Vulkan input flags:** buffer creation rejects usages requiring disabled
  extensions or unknown bits. Acceleration builds reject feature/extension flags
  the backend does not enable and the contradictory fast-build/fast-trace pair,
  before the Vulkan size query/build. The latter violates
  [VUID-03796](https://docs.vulkan.org/refpages/latest/refpages/source/VkAccelerationStructureBuildGeometryInfoKHR.html).
  CPU tests cover accepted/rejected combinations; validation-backed tests exercise
  invalid buffer usages and TLAS build flags through their public entry points.
- **Shader compiler:** extracted process/file management into
  `gpu/shader/compiler.rs`. Compilation uses an exclusively created temporary
  directory with cleanup attempted on every return path. Unix permissions are
  owner-only; Windows inherits the user's temporary-directory permissions.
  Cleanup checks the resolved absolute directory against its owned path and parent.
  Verified bundled shaders compile from embedded source snapshots, preserving
  includes and eliminating a working-file edit between verification and compilation.
  Custom source paths are made absolute before becoming command arguments.
  Truncated/misaligned SPIR-V and invalid basic headers are rejected before Vulkan;
  this is not a complete SPIR-V validator. CPU tests cover workspace cleanup,
  snapshots and malformed output; a separate CLI test covers real compilation,
  bundled includes, a filename containing spaces/leading hyphen, and invalid GLSL.
- **Repeated teardown:** a serial hardware regression warms one shared backend,
  then performs 24 acoustic/auralization creation/drop cycles, each with three
  geometry/IR states, filter replacement and audio rendering. After every cycle,
  active VMA allocations return to **one allocation / 16,777,216 bytes** (the
  backend's staging buffer), and AS/RT/compute pipeline registries return to zero.
  Dropping the backend releases the final device owner. A deliberately retained
  buffer verifies that the allocation counter detects live storage. The run took
  48.96 seconds with validation enabled and emitted no validation-error markers.

The teardown test measures active VMA allocations, not allocator pool capacity,
driver-private memory, or VkFFT's internal raw allocations. Its one-backend scope
avoids the separately reproduced native multi-device stall. It does not establish
that arbitrarily long runs or every native error path are leak-free.

## Playback failures and scene synchronization

- **Playback failure reporting:** the first device/callback error is retained
  before cancellation wakes blocked producers. Healthy checks require no lock;
  repeated callers receive the same category/message and shared original cause.
  Explicit cancellation remains successful. Player push/finish/stop/wait and
  runtime request/shutdown paths now report the failure instead of treating it
  as normal completion. Runtime shutdown checks again after joining the worker,
  and `is_sound_player_running` observes actual worker termination.
- **Playback ownership:** a state now belongs to one stream. Its queue and first
  failure are installed once; restarting/replacing that queue cannot clear an
  error or strand producers on an old queue. Queue storage uses fallible reserve,
  returning a categorized error on capacity overflow/allocation failure. Failed
  allocation leaves setup retryable. Tests cover error retention and its original
  cause, cancellation, blocked producers, malformed callbacks and failed setup.
- **API change:** `SoundPlayer::wait_until_finished` now returns `SoundResult<()>`;
  the workspace example handles that result. `SoundPlayer::check_error` reports
  an asynchronous failure without consuming it. `SoundError` is cloneable with
  shared ownership of its underlying cause. No physical audio-device failure was
  injected; playback error checks use synthetic callbacks/state/queues.
- **Scene invalidation:** committed revisions distinguish surface geometry,
  material/object records and diffraction geometry. Material edits preserve
  surface/diffraction acceleration structures; object transforms preserve edge
  resources; edge edits preserve surface structures and shading buffers. Geometry
  edits also refresh object records because they contain addresses and transforms.
  The overall scene version still invalidates acoustic IR caches on every accepted
  nonempty update. Failed transactions preserve all resource revisions.
- **Organization and correctness:** GPU resources are grouped by their ownership
  and invalidation rules. A separate hardware test module verifies actual uploaded
  material indices/values, object addresses/transforms, edge records, instance
  changes and final allocation release. The acoustic regression changes an
  absorbing wall to a transmitting one: IR energy changes while the query TLAS is
  retained. The 24-cycle teardown check passes again with the grouped resources.

The serial scene-sync fixture uses one shared triangle mesh, 64 active objects and
16 diffraction edges. Each timed operation edits one record and synchronizes the
GPU scene; it does not trace rays. Four warm-up edits precede five windows of 16
operations. The baseline executable was saved before the scene changes. Two runs
of each binary, without validation enabled for timing, measured:

| Scene edit + GPU sync | Baseline median, µs | Updated median, µs |
| --- | ---: | ---: |
| Material replacement | 1,272–1,372 | 185–233 |
| Object transform | 1,219–1,382 | 356–541 |
| Diffraction edge replacement | 1,221–1,388 | 362–594 |

The paired material-update measurements improve by approximately 83–85% for this
fixture. These are setup/edit costs, not an end-to-end rendering speed claim.
Logs and the baseline binary are in `output/review-scene-playback-20260928/`.

## Worker startup, status and recovery

- **Recoverable startup:** the caller retains the engine core until the worker
  has initialized its platform audio stream. Thread creation, initialization
  errors and initialization panics return the original core instead of dropping
  it or leaving the public facade without an owner. Failed handoffs join the
  worker. The small `engine/worker.rs` helper supports thread-local state that
  cannot be transferred between threads; its CPU test uses a non-`Send` value.
- **Accurate voice status:** runtime voice queries now read a published atomic
  count instead of placeholder values. The worker publishes after rendering and
  before command replies. Normal worker exit retains the core's voices for
  recovery; a panicking worker clears the snapshot because its core is destroyed.
  A zero count describes completed voice rendering, not drained device audio.
- **Failure propagation and teardown:** the first asynchronous engine failure is
  published before queued command replies are disconnected, preserving the cause
  for waiting callers. A disconnected command channel stops active playback.
  Runtime destruction wakes a blocked producer and joins its worker, so dropping
  the internal handle cannot silently detach the engine/device owner.
- **Organization and verification:** startup handoff and runtime integration tests
  are in adjacent modules. The hardware regression uses a real Vulkan engine and
  simulated audio callbacks to cover failed startup, command acknowledgment
  snapshots, a full queue, device failure, a poisoned listener mailbox, recovery,
  restart, voice completion and final device-owner release. No physical audio
  device is opened by this test, and OS thread-creation failure is not injected.

Worker evidence is in `output/review-worker-20260928/`.

## Native Vulkan transfer cleanup

- **Verified native leak:** a CPU-only probe of the bundled VkFFT upload helper
  reproduces a mapping-failure exit retaining one Vulkan buffer and one dedicated
  memory allocation. Other early returns also abandon local command buffers.
  The readback helper checks an unrelated result after `vkMapMemory`, allowing a
  failed mapping to reach `memcpy`. These are source-level/native helper defects;
  they do not establish the cause of the reported machine crashes.
- **Scoped ownership:** `vkfft-rs/src/vulkan_memory.hpp` replaces the four Vulkan
  memory/transfer helpers. Buffer, allocation and command-buffer ownership stays
  local until success, with null outputs after failed allocation/binding and
  buffer destruction before memory release. Transfers use stack handle storage
  and preserve caller-owned staging resources. Failed mappings return before
  accessing or unmapping the output pointer.
- **Submission lifetime:** initialization transfers include dependencies around
  copying and a host-read dependency for readback. A timeout or host-allocation
  error while waiting does not establish GPU completion; resources remain alive
  until a later successful wait or device loss, then the first error is reported.
  As with the Rust synchronous path, a driver that never completes or reports
  device loss can still block the caller. No real device loss was injected.
- **Build and test organization:** the FFI explicitly includes the replacement
  before VkFFT, avoiding compiler-dependent include search order and leaving the
  pinned submodule unchanged. Cargo now tracks native headers and `VULKAN_SDK`
  changes and requests C++17 on both supported compiler families. A separate
  native test object is pulled into the Rust test executable only when referenced.
- **Verification:** 52 CPU cases cover upload/readback with owned/borrowed staging,
  successful copies and failures during allocation, binding, mapping, command
  recording, submission, waiting and fence reset. They check retained counts,
  cleanup order, pending work, untouched readback outputs and null failed-allocation
  outputs. A GPU regression checks FFT lengths 16, 17, 257 and 1009 against selected
  double-precision DFT bins and inverse round trips. Prime lengths exercise the
  library's LUT/recursive initialization paths. The full audit/teardown suite passes.

Evidence is under `output/review-native-memory-20260928/`.
`upstream-transfer-result.log` is the CPU-only upstream leak reproduction;
`cpu-debug.log` and `cpu-release.log` contain the replacement's failure tests.
The replacement covers transfer helpers, not every nested VkFFT initialization
failure path. The separate recursive cleanup findings are addressed below.

## Recursive VkFFT initialization

- **Reproduced ownership defects:** a CPU-only probe of the bundled Rader kernel
  generator injects an upload failure after temporary allocation. It leaves one
  buffer and one device-memory allocation alive, while freeing a CPU kernel still
  referenced by the parent application. Parent cleanup could therefore free the
  same allocation again. Separate command-error exits also leave command buffers
  allocated or delete a nested plan after a wait that did not prove completion.
- **Scoped temporary resources:** nested applications, phase-vector host storage
  and the temporary Rader buffer now have one scoped owner. The parent retains
  ownership of cached CPU kernels on both success and error. Nested plans are
  destroyed before their temporary buffer, and four repeated command submission
  blocks share a checked helper with the same completion/device-loss rule as the
  transfer path. FFT configuration, phase calculations and transform directions
  are preserved.
- **Maintainable vendor integration:** the reviewed unified patch lives in
  `vkfft-rs/patches/recursive_cleanup.patch`. The build copies the native headers
  into Cargo's output directory and applies hunks only at their exact original
  positions with matching context. A mismatch fails the build for review; there
  is no fuzzy patching and the source submodule is unchanged. This also avoids
  compiler-dependent selection of the original header through quoted includes.
- **Verification:** seventeen additional CPU cases execute the patched Rader
  generator with simulated initialization, upload, recording, readback and Vulkan
  failures. They verify temporary cleanup, no destruction while work is pending,
  and exactly one surviving parent-owned CPU kernel before explicit parent cleanup.
  The existing 52 transfer cases still pass. A patch-context regression rejects
  changed source. The GPU FFT test now also checks length 65,537, above the bundled
  Rader cutoff, exercising Bluestein setup against DFT bins and an inverse round trip.

Evidence is under `output/review-recursive-init-20260928/`.
`upstream-result.log` reproduces the unpatched ownership failure without touching
the driver. `native-cpu.log`, `cpu-debug.log` and `cpu-release.log` contain the
corrected CPU checks; `gpu-fft.log` records the numerical hardware regression.
This does not establish cleanup correctness for every compiler/native failure or
resolve the separately observed device-creation stall.

## Native compiler ownership and exception boundary

- **Reproduced compiler leak:** a CPU-only probe of the unmodified compiler
  simulates compute-pipeline creation failure. After its own error cleanup,
  two host allocations (generated source and SPIR-V) and one shader module remain
  live. The probe uses fake Vulkan functions and never enters the driver.
- **Scoped compilation:** glslang shader/program objects, temporary SPIR-V and
  shader modules now have scoped owners. Null compiler objects return allocation
  errors. SPIR-V length and pointer checks precede copying. Pipeline/layout
  ownership stays with the axis, including a partially returned pipeline on error.
  Failed shader-module/layout creation clears unspecified output handles before
  cleanup. A successful plan retains the binary only when serialization requests it.
- **Generated source:** the axis owns generated text until its caller releases it
  after successful compilation, or axis teardown releases it on failure/retention.
  Compiler error exits no longer delete their parent application while local
  resources still refer to it. The caller retains responsibility for parent cleanup.
- **FFI exceptions:** initialization and command recording translate C++ allocation
  exceptions and unexpected exceptions into existing VkFFT errors. Local owners
  unwind before initialization rollback runs. MSVC uses `/EHs`, preserving exception
  handling for glslang's C entry points; other compilers use `-fexceptions`.
- **Verification and integration:** 96 CPU cases run the actual patched compiler
  and axis teardown with simulated failures, exception throws from C entry points,
  retained/source/binary combinations and serialized-length checks. They check
  zero temporary retention, cleanup order and repeated cleanup. These patches use
  the same exact-context build integration as the recursive patch; the pinned
  source submodule remains unchanged. Serialized-length checks do not make the
  upstream raw-pointer deserializer safe for arbitrary untrusted storage; Rust's
  wrapper does not expose that feature.

Evidence is under `output/review-compiler-20260928/`. `upstream-result.log`
records the original leak; `cpu-debug-final.log` and `cpu-release-final.log`
contain the regression results. This pass covers compiler-owned resources;
the shader-generation findings from the subsequent pass are recorded below.

## Shader-generation memory and append bounds

- **Reproduced nested host leak:** the original `PfDeallocateContainer` resets the
  type tag before deciding whether to free complex/double-double children. A
  normal complex-container allocation creates four allocations and its original
  cleanup leaves three live. Cleanup now preserves the tag until children are
  released and tolerates partial allocation and repeated cleanup. Failed child
  allocation returns before dereferencing the failed output pointer.
- **Reproduced capacity overwrite:** `PfAppendLine` sets an insufficient-buffer
  error but continues writing, and its old limit excludes the terminating null.
  A canary-backed probe declares eight bytes of capacity and observes a write at
  offset eight after a nine-character append reports an error. The patched append
  rejects invalid lengths and insufficient capacity before writing, uses subtraction
  to avoid signed overflow, and copies the already-measured text without another
  formatting pass. Rejected appends preserve the destination and current length.
- **Parameter ownership:** original axis teardown leaves 46 parameter allocations
  live in a fixture with Rader names and workgroup/buffer offsets enabled. Teardown
  now releases API parameter storage before freeing its Rader array. Borrowing
  inverse Bluestein axes detach the shared array before cleanup because its owner
  may already have been destroyed. Successful R2C decomposition also releases
  its temporary API parameters, matching ordinary FFT plan creation.
- **Allocation failure handling:** parameter initializers now return the recorded
  allocation error before formatting names through null pointers. Cleanup works
  with partially initialized parameters and a missing Rader array. CPU regressions
  cover 24 container types, failure at each nested allocation, copy/swap ownership,
  append boundaries and signed-overflow inputs, failure at every allocation in the
  parameter fixture, repeated cleanup and owner-before-borrower destruction.

Evidence is under `output/review-codegen-20260928/`. `upstream-result.log`
records the original nested leak and capacity overwrite, while
`upstream-parameters-result.log` records the original axis-cleanup leak. These are
CPU-only reproductions. `native-final-debug.log` and `native-final-release.log`
contain the stricter final regressions; workspace CPU checks are in `cpu-debug.log`
and `cpu-release.log`. The subsequent register/temporary-formatting findings are
recorded below, followed by the container-name and copy-ownership findings.

## Register cleanup and bounded temporary formatting

- **Reproduced skipped cleanup:** the original register cleanup returns immediately
  when shader generation has already reported an error. A CPU-only fixture starts
  with 44 host allocations, reaches 100 after register initialization and retains
  all 100 after error cleanup: all 56 register allocations remain live.
- **Shared teardown:** both FFT and R2C register cleanup now share one release path.
  It walks initialized container tags, frees arrays and clears their pointers,
  including optional fields whose old allocation and cleanup conditions differed.
  It preserves the original error and supports partial initialization and repeated
  cleanup. Register counts use checked arithmetic; fixed-array index ranges are
  validated before initialization. Failed allocations stop before their output is
  dereferenced, and both generator entry points return after failed register setup.
- **Reproduced temporary overwrite:** an actual unmodified `PfDefine` call writes
  beyond a declared eight-byte temporary buffer while returning success. Formatting
  now uses the buffer's declared capacity and records truncation/invalid-capacity
  errors before an append can consume the result. A canary-backed regression checks
  the real rewritten call path as well as boundary lengths and preservation of a
  prior error.
- **Maintainable integration:** the build rewrites the exact temporary-buffer
  destination expression in 16 pinned headers (721 occurrences before other
  cleanup patches, including commented examples). Expected per-file counts and
  destination checks reject drift. Formatting strings and arguments remain intact;
  a small helper owns the bounds/error behavior. The regular unified patches still
  require exact context. Inspection of the generated release header tree found no
  remaining `sprintf` call targeting `sc->tempStr`.
- **Verification:** CPU cases inject failure at every register-allocation point in
  ordinary, convolution/Rader, R2C and double-double fixtures. They check restoration
  of the prior live-allocation count, preserved errors and repeated cleanup. Both
  actual generator entry points are exercised with first-register allocation failure.
  Hardware FFT numerical checks and the full teardown/ownership suite pass.

Evidence is under `output/review-registers-20260928/`. `upstream-result.log`
contains both original defect reproductions. `cpu-debug.log`, `cpu-release.log`,
`check.log`, `check-no-playback.log` and `clippy.log` contain final CPU/build results.
The initial `native-cases.log` run was interrupted: the synthetic full-generator
fixture omitted required layout fields and hit a CPU divide-by-zero before the
target allocation. `register-phases*.log` and `generator-phases.log` preserve that
diagnosis. The fixture now supplies stage, element-size and shared-bank metadata;
`register-fixture-final.log` records its focused passing run. These fixture failures
did not call the GPU and are not passing tests or evidence of the reported driver crash.

## Container names and copy ownership

- **Reproduced name overwrites:** the original `PfCopyContainer` checks length
  after `sprintf` and accepts a length equal to capacity, missing the terminator.
  `PfSetContainerName` also appends component suffixes without checking their
  allocations. CPU probes against the original headers report success while
  overwriting canaries beyond the declared capacity in both cases. The backing
  arrays are deliberately larger than the declared capacity, so the probes do
  not corrupt actual allocations.
- **Reproduced ownership corruption:** copying a numeric double-double container
  first copies its components, then overwrites its owning union pointer through
  the scalar-value branch. The destination then aliases the source's allocation,
  losing its previous owner. Complex double-double values have the same problem
  one level down. Copying now preserves every nested allocation; swapping
  preserves nested array addresses while exchanging values and name ownership.
  Type compatibility is checked before traversing children, and opaque type-name
  tags are no longer interpreted as containers with child storage.
- **Shared, bounded names:** a small helper owns name formatting and reports
  truncation before later generator operations can use it. Copies check the
  source terminator within its declared capacity and use overlap-safe `memmove`.
  Component suffixes respect each child's capacity. Push-constant prefixing
  checks capacity before moving bytes in place, removing the temporary string
  and its extra copy. Native parameter initializers now return recorded format
  errors rather than unconditionally returning success.
- **Integration and tests:** exact-context patches cover the math operations,
  parameter initializers, kernel constants, registers and push constants. The
  generated header tree has no remaining uncommented `sprintf` destination
  targeting a container name or temporary buffer. Regressions cover empty,
  exact-fit and too-long names, self-copy,
  suffix growth, unterminated sources, invalid capacities, preservation of an
  earlier error, opaque tags, numeric double-double ownership, a real initializer
  formatting error and allocation failure in `appendSinCos20`. That kernel helper
  previously wrote through the failed allocation's null name pointer.

Evidence is under `output/review-names-20260928/`. `upstream-probe.log` records
both boundary overwrites and the original destination/source pointer alias.
`native-cases.log` exercises the patched code paths without calling Vulkan.
`cpu-debug.log` and `cpu-release.log` record the passing workspace CPU suites.
The name operations avoid formatted output for plain copies and an intermediate
buffer for prefixing; no new runtime performance measurement is claimed.

## Permutation bounds and radix temporary allocations

- **Permutation bounds:** the old helper uses 33 stack entries without checking
  its requested count or permutation indices. The replacement checks count,
  indices, uniqueness, matching register types and strided capacity before any
  register changes. Its callers all supply bijective permutations. Invalid maps
  now return a math error; zero-length calls and prior errors remain harmless.
- **Allocation-free permutation:** cycle swaps rearrange existing names and their
  ownership without allocating temporary containers or copying their strings.
  Nested array addresses stay stable. The radix-14 kernel's strided seven-element
  permutations use the same helper, replacing another temporary array and two
  copy passes per permutation.
- **Unchecked constant allocation removed:** every inline radix kernel previously
  allocated two numeric components and then directly dereferenced them without
  checking the allocation result. The constant now owns those two components on
  the stack for the duration of code generation. This removes that failure path
  and one allocation/free pair per kernel call.
- **Output and allocation evidence:** a CPU-only comparison generates 128 cases:
  radices 2–16 and 32, single/double precision, two stage sizes and both angle
  signs. Every emitted shader is byte-identical before and after the change.
  Tracked allocation calls inside those kernel invocations fall from **6,368 to
  zero**. Setup allocations are excluded, and this is not an end-to-end speed or
  frame-time measurement.
- **Regressions:** tests exhaust all permutations through seven entries for
  complex and double-double names, local/register storage and strided storage.
  Additional cycles cover counts 32 and 33. Tests check nested ownership, untouched
  interleaved registers, invalid maps and zero allocation attempts. The 128 real
  kernel-generation cases also pass with the allocator configured to fail its
  next call. The hardware DFT/round-trip test now includes every small radix size
  from 2 through 16 and 32, retaining the prime/Rader/Bluestein cases.

Evidence is under `output/review-permutations-20260928/`. `comparison.log` records
the exact shader comparison and allocation counts; `baseline-result.log` and
`updated-result.log` record each case. `baseline/` and `updated/` retain the shader
text. The comparison selects the `p0`/`p1` single/double cases. Two older `p2`
files came from an exploratory double-double fixture that did not configure its
required LUT path; they are excluded from the comparison, not counted as passing
coverage of that mode. `native-cases.log` records the focused CPU regressions.

## Rader stage temporary ownership

- **Reproduced successful-path leak:** a CPU-only call to the actual Rader-17
  generator starts with 281 live host allocations, returns success with 345 and
  leaves 64 live after full axis teardown. The last forward stage frees its
  temporary array but skips destruction of the names and children in its elements.
  Earlier container cleanup fixes cannot recover owners whose array was discarded.
- **Scoped arrays:** a shared, non-copyable owner now releases all initialized
  elements and the array on every Rader return path, including partial allocation
  failure and C++ unwinding. Dimensions and products are checked before allocating.
  Construction preserves prior errors, and failed element initialization returns
  immediately. Internal linkage separates the test allocator from production code.
- **Unused snapshot removed:** the last forward stage consumes the registers
  directly and never reads its snapshot. It now skips that allocation and the
  associated name copies. The Rader-17 fixture drops from 260 to 195 allocation
  attempts per generation, returns to 281 live allocations and leaves zero after
  teardown. Its 38,468-byte generated shader is unchanged by this optimization.
  This establishes an allocation reduction for the fixture, not an end-to-end
  timing measurement or the cause of the originally reported driver crash.
- **Failure coverage:** native regressions inject failure at every temporary-array
  and element allocation in one-stage, two-stage, double-precision and strided
  fixtures. Each returns its original error and restores the live-allocation
  baseline. Code-buffer exhaustion, invalid/overflowing counts, optional storage,
  prior errors and exception unwinding are also covered.

Evidence is under `output/review-rader-20260928/`. `baseline-result.log` records
the original leak; `scoped-result.log` records cleanup before the optimization,
and `optimized-result.log` records final allocation counts. `shader-comparison.log`
compares the emitted source before and after skipping the unused snapshot.
`native-optimized.log` records the final focused regression run. The initial
`native-rader.log` is a failed build from a duplicated test-header include, which
was removed; it is not a passing run or a GPU failure.

## Lookup-table ownership and initialization

- **Reproduced R2C host leak:** CPU probes of the actual LUT builder show one
  host table left allocated whenever a forward real-to-complex plan borrows its
  inverse plan's table. This occurs in float, double and double-double modes;
  cleanup was confined to the branch that uploads a new table. Reuse now returns
  before generating a temporary table. The probes drop from one/two allocation
  attempts (ordinary/double-double precision) to zero, with no live allocations.
- **Reproduced uninitialized upload:** a single-stage transform with no twiddle
  entries still uploads its required fallback buffer. An allocator-poisoned CPU
  probe observes `0xa5` bytes in the original 4-, 8- and 16-byte fallback tables.
  All seven host LUT buffers now use a small scoped, zero-initialized owner, so
  fallback storage and unused padding cannot expose previous host data. The owner
  frees storage on returns and exceptions without accessing a potentially deleted
  application or axis. Device uploads remain synchronous.
- **Conversion failure:** numeric double-double conversion previously dereferenced
  its output immediately after an unchecked child allocation. It now returns on
  allocation failure. LUT generation uses a fixed pair of local numeric containers,
  removing that allocation entirely, and preserves prior specialization errors.
  CPU tests exercise failure in both allocating converter branches without a GPU.
- **Byte counts:** complex and R2C table allocation sizes reject element-to-byte
  overflow before allocation or filling. Empty R2C dimensions are also rejected.
  These final byte-conversion checks are complemented by the layout checks below;
  they do not establish bounds for every earlier native scheduler calculation.
- **Coverage:** the native regressions exercise all five precision configurations,
  ordinary/R2C/integer-Rader uploads, host/device/transfer failures, exception cleanup,
  borrowed tables, prior errors, fallback bytes and oversized dimensions. Uploaded
  twiddle values are checked against trigonometric references, and integer Rader
  tables against modular powers. Allocation canaries and live-allocation counts
  check bounds and cleanup. This is an allocation reduction, not an end-to-end
  performance measurement or proof of the reported system-crash cause.

Evidence is under `output/review-lut-20260928/`. `baseline-result.log` and
`fixed-result.log` record the before/after CPU probes. `native-release-final.log`
records the passing focused regressions. The initial `native-release.log` contains
a failed boundary assertion: the fixture incorrectly put a 64-bit size in a
32-bit radix field. The corrected fixture uses the native 64-bit DCT dimension;
this was a CPU test-fixture issue, not a GPU failure.

## Lookup-table layout arithmetic

- **Reproduced narrowing:** a CPU probe using stages `INT_MAX, 2, 2` computes
  6,442,450,941 complex entries but stores 2,147,483,645 in the signed 32-bit
  `maxStageSumLUT` field. A separate four-step probe multiplies two dimensions of
  2^32 in signed arithmetic; this build wraps the product and requests only the
  four-byte fallback table. Both probes force host allocation failure before any
  table filling, so neither huge allocations nor out-of-bounds writes are performed.
- **Checked shared calculations:** one small layout helper replaces three copies
  of the stage-size switch. It preserves the shader generator's reservation order,
  validates fixed-array counts before access, and checks multiplication and offset
  addition against the consuming field's range. Four-step products, DCT/DST
  extensions and appended Rader kernels are checked before allocating or filling.
  Prior successful table layouts retain their offsets.
- **Integer Rader tables:** offsets are computed and checked once for both new and
  cached tables. The total must fit native signed indexing and the host byte count;
  a cached allocation must also match the expected extent. Invalid axis/upload
  indices, missing Rader storage, invalid prime/generator values and missing FFT
  kernel storage are rejected by the relevant layout helper.
- **Failure behavior:** the offset-narrowing and four-step probes now return
  `VKFFT_ERROR_MATH_FAILED` without an allocation attempt. A third probe with an
  oversized integer-Rader layout does the same. This guards native planning
  invariants; it does not imply arbitrary corrupted owning pointers or array
  lengths can be safely destroyed, nor establish a trigger for the reported crash.
- **Regression coverage:** CPU tests cover signed count and offset boundaries,
  negative/overflowing dimensions, incompatible Rader reservations, cache extent
  mismatches and valid cached reuse with allocation disabled. Nested Rader-17 and
  Rader-31 tables are checked in all five precision configurations against expected
  forward/inverse offsets and independent trigonometric values. Canary checks
  verify the resulting host writes remain within their allocations. Seventy valid
  four-step/DCT/DST cases check extension sizes, offsets and endpoint values across
  the five precision configurations and both transform directions.

Evidence is under `output/review-lut-layout-20260928/`. `baseline-result.log` and
`fixed-result.log` record the guarded before/after probes. `native-layout.log`
records the focused passing regression run; `cpu-debug-final.log` and
`cpu-release-final.log` include the additional valid four-step/DCT/DST cases.
The probes call the actual LUT builders with mocked allocation/transfers; they
never invoke the Vulkan loader.

## Lookup-table reuse and teardown

- **Reproduced invalid reuse:** CPU probes show both complex and R2C builders
  returning success with a borrowed null buffer, or accepting a one-byte source
  for a required 64-byte table. The complex builder also allocates and generates
  a temporary table before deciding to reuse one; host allocation failure prevents
  otherwise valid reuse. These controlled probes do not submit invalid GPU work.
- **Compatible source selection:** one helper replaces three duplicated selection
  trees. It checks the actual upload slot, initialized buffer/memory handles and
  expected allocation extent. Complex tables additionally require matching stage
  counts/order, Rader layout/kernel identity, four-step geometry and relevant real
  transform parameters. Missing, incomplete or incompatible candidates cause an
  independently owned table to be generated. The old cross-axis path compared slot
  zero but could borrow a different slot without checking that slot's layout.
- **Earlier reuse:** expected byte sizing and compatibility checks now precede host
  allocation and trigonometry. The valid complex-table probe drops from one
  temporary allocation to zero and succeeds with host allocation disabled. R2C
  still avoids temporary work when reuse is valid, but now validates size and
  source storage first. The null/short-source probes instead perform one valid
  allocation/upload and leave `referenceLUT` clear. No end-to-end timing gain is
  claimed from these allocation counts.
- **Teardown:** Vulkan axis cleanup clears borrowed handles, byte size and the
  reference flag, as well as clearing owners after destruction. Tests execute
  the actual `deleteAxis` with mocked Vulkan destruction in both teardown orders
  and repeat cleanup, observing exactly one buffer/memory release. Borrower-first
  cleanup leaves its source intact. The application still owns the complete
  sharing graph and must finish submitted work before destruction.
- **Coverage:** CPU cases exercise inverse, Bluestein, cross-axis and R2C reuse
  across five precision configurations, invalid/null source metadata, fallback
  allocation failure, nested Rader mismatches and kernel identity. A new hardware
  regression covers six 2D/3D/4D shapes, including repeated prime axes and unequal
  axis lengths, comparing selected output bins to an independent multidimensional
  DFT and every inverse sample to the original input.

Evidence is under `output/review-lut-sharing-20260928/`. `baseline-result.log` and
`fixed-result.log` record the before/after CPU probes. `native-sharing-final.log`
records the focused native regressions, and `gpu-multidimensional.log` records
the initial passing hardware check under Vulkan synchronization validation.

## Measurements

`sound-engine/examples/runtime_costs.rs` is a bounded serial microbenchmark.
CPU tests use one mesh with 900,000 vertices and one object. GPU tests use a
2,048-point FFT, 1,024-frame stereo blocks and a one-tap IR. Setup is outside the
timed regions; each measurement warms up four operations and reports the median
of five windows. Windows contain 32 edits/audio blocks or 16 IR preparations.

The baseline binary was built before the implementation changes and saved under
`output/review-performance-20260928/`. Baseline and updated binaries were each
measured twice on the same machine, without validation layers enabled for timing.

| Operation | Baseline median, µs | Updated median, µs |
| --- | ---: | ---: |
| Scene transform, 900k vertices | 2,066–2,083 | 0.331–0.378 |
| Empty scene update | 1,977–2,073 | 0.009; essentially timer/loop overhead |
| Convolve a 1,024-frame block | 294–301 | 207–224 |
| Prepare a CPU-sourced IR | 263–269 | 208–233 |

The paired GPU runs reduced median convolution time by approximately 24–31%
and CPU-sourced IR preparation by 11–22%. These measurements establish gains for
this fixture, not an end-to-end engine improvement or a guarantee on other GPUs.
Scene edits still copy transaction metadata in proportion to record count; the
improvement removes the dependence on total vertex/index bytes.

Reproduce with:

```powershell
cargo run -p sound-engine --example runtime_costs --release --offline
cargo run -p sound-engine --example runtime_costs --release --offline -- --gpu
```

## Radix-stage and shuffle storage review

- Reproduced exception cleanup leaks in the ordinary radix stage and shuffle
  generators using the CPU-only allocator. Throwing during the fourth allocation
  left three allocations alive after axis teardown in each sampled path. Scoped
  container arrays now own all eight temporary-array sites across the strided and
  non-strided generators. Allocation/copy errors return immediately and keep the
  original error. These are injected native exceptions, not reproduced GPU faults.
- The exhaustive double-double fixture exposed another temporary outside that
  ownership: an exception during conversion left one allocation alive. Move,
  negate, add, subtract and multiply now scope their eight conversion temporaries
  through a small shared owner. Conversion failures return before accessing the
  temporary's child storage. The existing public conversion API is unchanged.
- Added shared checks for the radix-table index, physical/logical register counts,
  signed-int allocation/index products, divisors, stage-size products and the
  shuffle's final-stage lookup. Invalid fixtures fail before allocation or shader
  generation. This does not complete the audit of upstream scheduler arithmetic.
- Each ordinary radix stage now allocates one reusable register batch. Float and
  double fixtures drop from 68 or 136 allocation attempts to 17 (75% or 87.5%
  fewer), depending on register boost. Across all 48 stage/shuffle fixtures the
  count drops from 25,752 to 22,752; shuffle snapshots retain their original work.
  This is an allocation-count improvement, not an end-to-end timing claim.
- The 48 fixtures cover both layouts, ordinary/coalesced LUT stages, boosted
  stages, ordinary/final boosted shuffles, spare registers and float/double/
  double-double precision. Original and final generated shader text **and register
  mappings** are byte-identical: 1,339,153 bytes, SHA-256
  `47b11251182488cb66754ac8e8d2d105ab88ee56dc8db7037863c7c0c0006f14`.
  Every allocation position is tested with both a null return and an exception:
  45,504 injected failures, plus code-buffer exhaustion and invalid-layout cases.
  Final tests preserve the fixture's live allocation count and fully release it
  at teardown. Baseline and final probes are saved under
  `output/review-radix-stage-20260928/`.
- The changes remain local helpers and exact-context build patches; the pinned
  VkFFT submodule is unchanged. Fixture code is in the adjacent native test header.

## Verification

- All workspace targets/features compile with `cargo check --workspace --all-targets --all-features --offline`.
- The engine library also compiles with `--no-default-features`, without playback.
- Debug and release library suites each pass **137 CPU tests**: 112 engine tests and
  twenty-five VkFFT tests. Twenty-one hardware tests and one compiler-CLI test are excluded
  from ordinary test runs; the compiler test passes when run explicitly.
- **Nineteen serial release GPU tests pass** with Khronos validation and synchronization
  validation requested: seventeen `audit_` regressions, staging growth and engine
  scalar-update/voice-refresh integration. No Vulkan validation error markers were
  reported. The two long diffraction physics tests were not rerun in this pass.
- Callback and cancellation tests use synthetic buffers/queues; they do not establish
  live-device behavior under arbitrary audio-driver failures.
- Clippy completes for all targets/features. Style/API-shape warnings remain;
  this is not a warning-free result.
- `git diff --check` passes.

Earlier evidence is under `output/review-performance-20260928/` and
`output/review-runtime-20260928/`. The preceding pass's evidence is under
`output/review-diagnostics-20260928/`; `gpu-audit-final.log`, `gpu-integration.log`
and `gpu-staging.log` contain the passing hardware checks. The earlier
`gpu-audit.log` intentionally remains as evidence of the barrier validation error
before its fix. Compiler/flag/native evidence is under `output/review-hardening-20260928/`,
including `gpu-soak.log`, `gpu-audit.log` and the compiler/CPU/build logs.
Earlier runtime logs also include interrupted stalled diagnostics; those runs
are not passing tests.

Latest scene/playback evidence is under `output/review-scene-playback-20260928/`.
`gpu-audit.log` records all fourteen passing audit regressions, including the soak;
`gpu-integration.log` and `gpu-staging.log` record the other passing hardware tests.
The initial `gpu-scene-sync.log` contains a rejected transfer requested by the test
fixture from a buffer lacking transfer-source usage; the fixture now checks the
uploaded object's address instead. `gpu-scene-sync-final.log` is the passing run.

Worker evidence is under `output/review-worker-20260928/`.
`gpu-audit.log` records all fifteen passing audit regressions, including runtime
recovery and the 24-cycle teardown check. `gpu-staging.log` and
`gpu-integration.log` contain the other two passing hardware checks.
`cpu-debug-final.log`, `cpu-release.log`, `check-final.log`,
`check-no-playback-final.log` and `clippy-final.log` contain the corresponding
CPU/build results. The runtime test also has a focused `gpu-runtime-final.log`.

Native transfer-helper evidence is under `output/review-native-memory-20260928/`.
`gpu-audit.log` contains sixteen passing audit tests; `gpu-staging.log` and
`gpu-integration.log` record the other two GPU checks. `gpu-fft-final.log` is the
focused FFT regression. CPU/build checks are `cpu-debug.log`, `cpu-release.log`,
`check.log`, `check-no-playback.log` and `clippy.log`.

Recursive-initialization evidence is under `output/review-recursive-init-20260928/`.
`gpu-audit.log` contains all sixteen passing audit regressions; `gpu-staging.log`
and `gpu-integration.log` contain the two additional passing GPU checks. CPU and
build results use the same filenames as the preceding native-helper directory.

Compiler evidence is under `output/review-compiler-20260928/`.
`gpu-audit.log` contains sixteen passing audit regressions, including the 24-cycle
teardown check and the FFT length-65,537 numerical check. `gpu-staging.log` and
`gpu-integration.log` contain the additional hardware checks. `check.log`,
`check-no-playback.log`, `clippy.log`, `cpu-debug-final.log` and
`cpu-release-final.log` record build/CPU verification.

Shader-generation evidence is under `output/review-codegen-20260928/`.
`gpu-audit.log` contains all sixteen passing audit tests, including FFT numerical
checks and the 24-cycle teardown soak. `gpu-staging.log` and `gpu-integration.log`
contain the two additional passing hardware checks. No validation error markers
were reported. Build checks are in `check.log`, `check-no-playback.log` and
`clippy.log`; CPU checks and native leak reproductions are described above.

Current register/formatting evidence is under `output/review-registers-20260928/`.
`gpu-audit.log` records sixteen passing audit regressions, including the numerical
FFT checks and 24-cycle teardown soak. `gpu-staging.log` and `gpu-integration.log`
record the other two passing hardware tests. No validation error markers were reported.

Current name/copy-ownership evidence is under `output/review-names-20260928/`.
`gpu-audit.log`, `gpu-staging.log` and `gpu-integration.log` record all eighteen
passing serial hardware regressions, with no Vulkan validation error markers.
`generated-header-scan.log` records inspection of the built native headers,
excluding comments while preserving string literals. Legacy `&sc->tempStr`
examples remain only inside upstream comments and are not compiled.

Current permutation/radix evidence is under `output/review-permutations-20260928/`.
`cpu-debug.log` and `cpu-release.log` each record 127 passing CPU tests.
`gpu-audit.log`, `gpu-staging.log` and `gpu-integration.log` record all eighteen
passing serial hardware regressions with no Vulkan validation error markers.
The numerical FFT audit now covers twenty transform lengths. `check.log`,
`check-no-playback.log` and `clippy.log` record the final successful build checks;
the existing Clippy style/API warnings remain.

Current Rader-stage evidence is under `output/review-rader-20260928/`.
`cpu-debug.log` and `cpu-release.log` each record 128 passing CPU tests.
`gpu-audit.log`, `gpu-staging.log` and `gpu-integration.log` record all eighteen
passing serial hardware regressions, with no Vulkan validation error markers.
`check.log`, `check-no-playback.log` and `clippy.log` record successful final
checks; existing Clippy style/API warnings remain.

Current LUT-storage evidence is under `output/review-lut-20260928/`.
`cpu-debug.log` and `cpu-release.log` each record 131 passing CPU tests.
`gpu-audit.log`, `gpu-staging.log` and `gpu-integration.log` record all eighteen
passing serial hardware regressions, with no Vulkan validation error markers.
The audit group completed in 66.96 seconds. `check.log`, `check-no-playback.log`,
`clippy.log` and `diff-check.log` record successful final checks; existing Clippy
style/API warnings and Git line-ending advisories remain.

Current LUT-layout evidence is under `output/review-lut-layout-20260928/`.
`cpu-debug-final.log` and `cpu-release-final.log` each record 133 passing CPU tests.
`gpu-audit.log`, `gpu-staging.log` and `gpu-integration.log` record all eighteen
passing serial hardware regressions with no Vulkan validation error markers.
The audit group completed in 70.90 seconds. Workspace/all-feature checks, the
no-playback check, Clippy and `git diff --check` pass; their logs are alongside
the tests. Existing style warnings and line-ending advisories remain.

Current LUT-sharing evidence is under `output/review-lut-sharing-20260928/`.
`cpu-debug.log` and `cpu-release.log` each record 135 passing CPU tests.
`gpu-audit.log`, `gpu-staging.log` and `gpu-integration.log` record nineteen
passing serial hardware regressions with no Vulkan validation error markers.
The seventeen-test audit group completed in 68.58 seconds. Workspace/all-feature
checks, the no-playback check, Clippy and `git diff --check` pass; their logs are
alongside the tests. Existing style warnings and line-ending advisories remain.

Current radix-stage evidence is under `output/review-radix-stage-20260928/`.
`cpu-debug.log` and `cpu-release.log` each record 137 passing CPU tests.
`gpu-audit.log`, `gpu-staging.log` and `gpu-integration.log` record nineteen
passing serial hardware regressions with no Vulkan validation error markers.
The seventeen-test audit group completed in 71.94 seconds. Workspace/all-feature
checks, the no-playback check, Clippy and `git diff --check` pass; their logs are
alongside the tests. Existing style warnings and line-ending advisories remain.
`baseline-result.log` and `final-result.log` retain allocation/exception evidence;
`shader-hashes.log` records unchanged shader/register output. `test-result.log`
retains the intermediate double-double exception leak that motivated the
arithmetic-temporary fix; it is not a passing final result.

## Unresolved native device-creation stall

On this RTX 3060 / NVIDIA 591.86 installation, creating another backend **on a
worker thread after acoustic setup, while the first acoustic engine remains alive**
reproducibly stalled inside `vkCreateDevice`. In the initial reproduction, the
first query had already returned.
It occurred with synchronization validation, without explicit validation, and
with implicit layers disabled for the diagnostic process. The latter used the
[documented Vulkan loader filter](https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderInterfaceArchitecture.md);
no machine/driver settings were changed.

Creating both devices before acoustic setup succeeds. Dropping the acoustic engine
before creating the second device also succeeds. The standalone probe succeeds
on the main thread but its `--worker-thread` variant times out after 20 seconds
at second-device creation. A 16 MiB worker stack also timed out.
The newer `--pipeline-only` variant omits the initial IR query and still stalls:
an acoustic pipeline and its sentinel BLAS have been created, but ray tracing
is not required to trigger the observed ordering problem.

Windows wait-chain inspection and brief stack capture of the diagnostic child
show the constructor worker waiting on a critical section owned by a native
thread. That thread waits on an SRW lock with frames in `nvoglv64.dll`,
`nvcuda64.dll` and `nvrtum64.dll`. The SRW owner is not exposed, so these captures
do not prove a closed deadlock cycle. Export-based symbol names are approximate.
The stall persists with loader-confirmed disabling of all implicit layers and
both Overwolf disable variables; those captures contain no Overwolf frames in
the constructor chain. Captures are `native-no-overlay-stacks.log` and
`native-pipeline-only-stacks.log` under `output/review-hardening-20260928/`. Only owned
diagnostic children were inspected/terminated; no global settings were changed.

This is a native stall observation, **not proof of a
specific driver defect or the cause of the originally reported system crashes**.
The ownership regression creates both backends first so it can test rejection
without hanging in the native constructor.

`sound-engine/examples/device_lifecycle_probe.rs` preserves the small reproduction.
Build it with `cargo build -p sound-engine --example device_lifecycle_probe --release --features playback --offline`.
Modes are `--early-device`, `--late-device` and `--after-drop`; add `--worker-thread`
to reproduce the failing thread context, or `--pipeline-only` to skip the first
IR query. Run the late worker case under an
external process timeout. Logs include `device-probe-worker-late.log` and the
`gpu-cache-*` phase diagnostics. The diagnostic does not reset the GPU or inject
GPU faults. Its hung processes were terminated externally.

## Review still in progress

The following are concrete next inspection areas, not claims of verified defects
in every listed path:

1. Investigate the native worker-thread device-creation stall above, including
   whether a supported ownership/initialization change can avoid it safely.
2. Remaining playback/runtime error paths and integration under repeated scene/voice
   changes, including actual device failure/recovery beyond synthetic callbacks.
3. Continue checking stage dispatch and shuffle-normalization invariants, native
   scheduler arithmetic and allocation failures before these layout helpers, remaining
   native configuration/data invariants, retention outside VMA statistics and
   repeated complete backend creation/destruction.
4. Extend the bounded soak toward the originally reported workload if its trigger
   can be isolated. No system crash has been reproduced, and no deliberate GPU
   fault injection has been performed.

Unrelated benchmark work and existing user edits have been preserved. Completed
audit fixes are being committed locally at the user's request; the review remains
active. No dependency upgrades, driver settings or OS watchdog changes were made.
