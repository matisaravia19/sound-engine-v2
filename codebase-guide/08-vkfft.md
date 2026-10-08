# VkFFT: the native boundary behind convolution

[Guide index](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/codebase-guide/README.md)

The engine does not implement its FFT kernels in Rust. It uses VkFFT through a small workspace crate, `vkfft-rs`. That crate turns Rust configuration into a native FFT plan, then records the plan's operations into an existing Vulkan command buffer.

For the doorway example, a plan transforms the dry input block, another transforms a replacement IR, and the signal plan also records the inverse transform. These plans are reused across blocks. Their construction cost is different from the cost of executing a prepared transform.

## The public surface is deliberately narrow

Start with [plan.rs](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/vkfft-rs/src/plan.rs). The wrapper supports the case the engine needs: a single-batch, one-dimensional, complex `f32` transform whose length is a power of two.

`FFTPlanBuilder` receives Vulkan handles, a buffer size, and a transform length. It checks that required handles are non-null, that the length is a nonzero power of two, and that the buffer can hold the complex samples. Checked arithmetic prevents a byte-count overflow from passing validation. Normalization can be requested for the inverse transform.

These checks cannot establish whether a non-null Vulkan handle is live or belongs to the right device. Building and recording therefore have explicit unsafe contracts. The caller must preserve resource lifetimes, serialize use of shared Vulkan objects, and complete submitted work before destruction.

VkFFT itself supports more modes than this wrapper exposes. Native files dealing with other layouts or scheduling cases are support for the bundled implementation, not evidence of additional public engine features.

## Owning a plan is not owning its device

The central distinction is between **handle storage** and **the resources those handles identify**.

VkFFT retains pointers to configuration values. The builder places the device, queue, command pool, fence, buffer, and related values in a boxed `HandleStorage`. The completed plan keeps that box, so moving the Rust plan does not move the pointed-to values.

The plan owns its opaque native application and that stable storage. It borrows the actual Vulkan resources: it does not destroy the engine's device or audio buffer when dropped.

The engine satisfies this contract in [convolver/gpu.rs](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/sound-engine/src/auralization/convolver/gpu.rs). Its FFT workspace declares the plan before its buffer, giving the plan an earlier drop order. The enclosing owner retains the backend until after its plans and other dependent resources are released.

The wrapper permits moving a plan to another thread with `Send`. That does not authorize simultaneous recording through the same plan or unsynchronized access to its queue.

## Recording and submitting are separate

`append` and `append_inverse` append FFT commands to the caller's command buffer. They do not submit it and do not wait for completion.

The convolver supplies the surrounding sequence:

~~~text
upload or prepare input
  → record forward FFT
  → make FFT writes visible to multiplication
  → multiply spectra
  → make multiplication writes visible to inverse FFT
  → record inverse FFT
  → submit, wait, and download
~~~

The Vulkan chapter explains the barriers and waits. The wrapper's job is the FFT portion of that sequence.

Plan creation and destruction take a process-wide compiler mutex. VkFFT initializes and finalizes shared glslang state during these operations, so independent Rust owners alone would not make concurrent construction safe. Queue access is also serialized at the engine integration point; these locks protect different shared state.

## Errors must stop at the C boundary

[ffi.rs](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/vkfft-rs/src/ffi.rs) defines the Rust side of the C interface. [native/ffi.cpp](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/vkfft-rs/native/ffi.cpp) translates it into VkFFT configuration and entry-point calls. Their field layouts and types must agree.

C++ exceptions cannot unwind through that interface into Rust. [ffi_guard.hpp](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/vkfft-rs/native/ffi_guard.hpp) catches allocation failures and other exceptions, converting them into result codes. Initialization failure also rolls back partially constructed native state after local owners have unwound.

Recording failure is different: the enclosing caller must abandon the failed recording. Returning an error does not make partially recorded commands a valid completed FFT.

[FftError](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/vkfft-rs/src/error.rs) translates native status into Rust errors. It preserves failure information without making resource validity a property that the error type can guarantee.

## Why there are patches and native helper directories

[build.rs](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/vkfft-rs/build.rs) copies the pinned vendor header tree into Cargo's build directory and applies the repository's patches to that copy. It does not rewrite the vendor submodule. Patch application checks its expected input and fails if a change no longer matches, rather than silently building a different integration.

The native helpers group responsibilities:

| Directory | Responsibility |
| --- | --- |
| [vulkan](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/vkfft-rs/native/vulkan) | Vulkan allocation, initialization, and shader compilation ownership. |
| [scheduler](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/vkfft-rs/native/scheduler) | Transform sizes, stage planning, limits, and checked arithmetic. |
| [lut](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/vkfft-rs/native/lut) | Lookup-table layout, sharing, storage, and lifetime. |
| [codegen](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/vkfft-rs/native/codegen) | Generated-kernel names, register storage, permutations, and stage policy. |

Reading every helper is unnecessary for understanding audio convolution. Begin at the Rust plan and C bridge, then follow a patch only when investigating the behavior it changes.

The build requires a C++ toolchain and the Vulkan SDK, including the compiler libraries expected by the build script. The optional `native-tests` feature adds C++ fault-injection harnesses; it is not needed for ordinary engine use.

## Evidence to follow

[Wrapper tests](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/vkfft-rs/tests/unit/plan.rs) check configuration constraints. [Build-support tests](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/vkfft-rs/tests/unit/build_support.rs) check patch handling. [Native harness tests](C:/Users/matis/OneDrive/Documentos/Fing/Tesis/codigo-v2/vkfft-rs/tests/unit/native.rs) exercise allocation, compilation, code generation, and cleanup failures.

Those checks complement the engine's GPU convolution tests. A mock failure test can establish that cleanup occurs on a particular path; an actual convolution comparison tests the numerical result produced by the whole integration. Neither replaces the other.
