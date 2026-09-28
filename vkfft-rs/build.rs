use std::env;
use std::path::PathBuf;
mod build_support;

fn main() {
    println!("cargo:rerun-if-changed=src/ffi.cpp");
    println!("cargo:rerun-if-changed=src/vulkan_memory.hpp");
    println!("cargo:rerun-if-changed=src/vulkan_init.hpp");
    println!("cargo:rerun-if-changed=src/vulkan_compiler.hpp");
    println!("cargo:rerun-if-changed=src/ffi_guard.hpp");
    println!("cargo:rerun-if-changed=src/vulkan_memory_tests.cpp");
    println!("cargo:rerun-if-changed=src/vulkan_compiler_tests.cpp");
    println!("cargo:rerun-if-changed=src/vulkan_lut_tests.cpp");
    println!("cargo:rerun-if-changed=src/lut_storage.hpp");
    println!("cargo:rerun-if-changed=src/lut_layout.hpp");
    println!("cargo:rerun-if-changed=src/lut_layout_tests.hpp");
    println!("cargo:rerun-if-changed=src/lut_sharing.hpp");
    println!("cargo:rerun-if-changed=src/lut_sharing_tests.hpp");
    println!("cargo:rerun-if-changed=src/codegen_tests.cpp");
    println!("cargo:rerun-if-changed=src/codegen_format.hpp");
    println!("cargo:rerun-if-changed=src/codegen_names.hpp");
    println!("cargo:rerun-if-changed=src/codegen_permutation.hpp");
    println!("cargo:rerun-if-changed=src/codegen_storage.hpp");
    println!("cargo:rerun-if-changed=src/codegen_temporary.hpp");
    println!("cargo:rerun-if-changed=src/codegen_stage_tests.hpp");
    println!("cargo:rerun-if-changed=src/codegen_stage_layout.hpp");
    println!("cargo:rerun-if-changed=src/codegen_stage_policy.hpp");
    println!("cargo:rerun-if-changed=src/codegen_stage_policy_tests.hpp");
    println!("cargo:rerun-if-changed=src/codegen_radix_stage_tests.hpp");
    println!("cargo:rerun-if-changed=src/codegen_permutation_tests.hpp");
    println!("cargo:rerun-if-changed=src/codegen_name_tests.hpp");
    println!("cargo:rerun-if-changed=src/codegen_registers.hpp");
    println!("cargo:rerun-if-changed=src/recursive_init_tests.hpp");
    println!("cargo:rerun-if-changed=patches");
    println!("cargo:rerun-if-changed=build_support.rs");
    println!("cargo:rerun-if-changed=lib/vkfft/vkFFT");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    let vulkan_sdk = env::var("VULKAN_SDK")
        .expect("VULKAN_SDK environment variable not set. Install Vulkan SDK.");

    let vulkan_include = PathBuf::from(&vulkan_sdk).join("Include");
    let glslang_include = vulkan_include.join("glslang").join("Include");
    let vkfft_include = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("vkfft-patched");
    copy_headers(&manifest_dir.join("lib/vkfft/vkFFT"), &vkfft_include);
    // One literal rewrite keeps hundreds of identical call-site edits reviewable.
    for &(header, count) in build_support::TEMP_FORMAT_HEADERS {
        let path = vkfft_include.join("vkFFT").join(header);
        let bounded =
            build_support::bound_temporary_formats(&std::fs::read_to_string(&path).unwrap(), count)
                .unwrap_or_else(|error| panic!("VkFFT format rewrite for {header}: {error}"));
        std::fs::write(path, bounded).unwrap();
    }
    for (header, patch) in [
        (
            "vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_RecursiveFFTGenerators.h",
            include_str!("patches/recursive_cleanup.patch"),
        ),
        (
            "vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_CompileKernel.h",
            include_str!("patches/compiler_cleanup.patch"),
        ),
        (
            "vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_DeletePlan.h",
            include_str!("patches/axis_cleanup.patch"),
        ),
        (
            "vkFFT_CodeGen/vkFFT_MathUtils/vkFFT_MathUtils.h",
            include_str!("patches/container_cleanup.patch"),
        ),
        (
            "vkFFT_CodeGen/vkFFT_StringManagement/vkFFT_StringManager.h",
            include_str!("patches/string_bounds.patch"),
        ),
        (
            "vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_InitAPIParameters.h",
            include_str!("patches/parameter_failures.patch"),
        ),
        (
            "vkFFT_PlanManagement/vkFFT_Plans/vkFFT_Plan_R2C.h",
            include_str!("patches/r2c_cleanup.patch"),
        ),
        (
            "vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_MemoryManagement/vkFFT_MemoryInitialization/vkFFT_Registers.h",
            include_str!("patches/register_cleanup.patch"),
        ),
        (
            "vkFFT_CodeGen/vkFFT_KernelsLevel2/vkFFT_FFT.h",
            include_str!("patches/register_failure_fft.patch"),
        ),
        (
            "vkFFT_CodeGen/vkFFT_KernelsLevel2/vkFFT_R2C_even_decomposition.h",
            include_str!("patches/register_failure_r2c.patch"),
        ),
        (
            "vkFFT_CodeGen/vkFFT_MathUtils/vkFFT_MathUtils.h",
            include_str!("patches/names_containers.patch"),
        ),
        (
            "vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_InitAPIParameters.h",
            include_str!("patches/names_parameters.patch"),
        ),
        (
            "vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_KernelUtils.h",
            include_str!("patches/names_kernel.patch"),
        ),
        (
            "vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_MemoryManagement/vkFFT_MemoryInitialization/vkFFT_Registers.h",
            include_str!("patches/names_registers.patch"),
        ),
        (
            "vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_MemoryManagement/vkFFT_MemoryInitialization/vkFFT_PushConstants.h",
            include_str!("patches/names_push.patch"),
        ),
        (
            "vkFFT_CodeGen/vkFFT_MathUtils/vkFFT_MathUtils.h",
            include_str!("patches/permutation_bounds.patch"),
        ),
        (
            "vkFFT_CodeGen/vkFFT_KernelsLevel1/vkFFT_RadixKernels.h",
            include_str!("patches/permutation_radix.patch"),
        ),
        (
            "vkFFT_CodeGen/vkFFT_KernelsLevel1/vkFFT_RaderKernels.h",
            include_str!("patches/rader_storage.patch"),
        ),
        (
            "vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_ManageLUT.h",
            include_str!("patches/lut_storage.patch"),
        ),
        (
            "vkFFT_CodeGen/vkFFT_MathUtils/vkFFT_MathUtils.h",
            include_str!("patches/conversion_failure.patch"),
        ),
        (
            "vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_ManageLUT.h",
            include_str!("patches/lut_layout.patch"),
        ),
        (
            "vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_ManageLUT.h",
            include_str!("patches/lut_sharing.patch"),
        ),
        (
            "vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_DeletePlan.h",
            include_str!("patches/lut_teardown.patch"),
        ),
        (
            "vkFFT_CodeGen/vkFFT_KernelsLevel1/vkFFT_RadixStage.h",
            include_str!("patches/radix_stage_storage.patch"),
        ),
        (
            "vkFFT_CodeGen/vkFFT_KernelsLevel1/vkFFT_RadixShuffle.h",
            include_str!("patches/radix_shuffle_storage.patch"),
        ),
        (
            "vkFFT_CodeGen/vkFFT_MathUtils/vkFFT_MathUtils.h",
            include_str!("patches/arithmetic_temporary.patch"),
        ),
        (
            "vkFFT_CodeGen/vkFFT_KernelsLevel1/vkFFT_RadixStage.h",
            include_str!("patches/stage_dispatch.patch"),
        ),
        (
            "vkFFT_CodeGen/vkFFT_KernelsLevel1/vkFFT_RadixShuffle.h",
            include_str!("patches/shuffle_policy.patch"),
        ),
        (
            "vkFFT_CodeGen/vkFFT_KernelsLevel1/vkFFT_RaderKernels.h",
            include_str!("patches/rader_policy.patch"),
        ),
    ] {
        let path = vkfft_include.join("vkFFT").join(header);
        let patched =
            build_support::apply_header_patch(&std::fs::read_to_string(&path).unwrap(), patch)
                .unwrap_or_else(|error| {
                    panic!("VkFFT patch for {header} no longer matches: {error}")
                });
        std::fs::write(path, patched).unwrap();
    }

    let mut build = cc::Build::new();
    build.cpp(true);
    build.file("src/ffi.cpp");
    // Separate archive objects are linked only by Rust's CPU regression tests.
    build.file("src/vulkan_memory_tests.cpp");
    build.file("src/vulkan_compiler_tests.cpp");
    build.file("src/vulkan_lut_tests.cpp");
    build.file("src/codegen_tests.cpp");

    build
        .include(&vulkan_include)
        .include(manifest_dir.join("src"))
        .include(&glslang_include)
        .include(&vkfft_include);

    build.std("c++17");
    // Local owners must unwind before an exception is translated at the C ABI.
    if build.get_compiler().is_like_msvc() {
        // glslang's C entry points call C++ allocation code that can throw.
        build.flag("/EHs");
    } else {
        build.flag("-fexceptions");
    }

    build.compile("ffi");
    println!("cargo:rustc-link-lib=static=ffi");

    println!("cargo:rustc-link-search=native={}/Lib", vulkan_sdk);

    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=dylib=vulkan-1");
        println!("cargo:rustc-link-lib=static=glslang");
        println!("cargo:rustc-link-lib=static=MachineIndependent");
        println!("cargo:rustc-link-lib=static=OSDependent");
        println!("cargo:rustc-link-lib=static=GenericCodeGen");
        println!("cargo:rustc-link-lib=static=SPIRV");
        println!("cargo:rustc-link-lib=static=glslang-default-resource-limits");
        println!("cargo:rustc-link-lib=static=SPIRV-Tools");
        println!("cargo:rustc-link-lib=static=SPIRV-Tools-opt");
    } else if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=vulkan");
    }
}

// A complete header tree makes quoted includes deterministic on MSVC and GCC.
// Patches affect only Cargo's build output, never the pinned source submodule.
fn copy_headers(source: &std::path::Path, destination: &std::path::Path) {
    std::fs::create_dir_all(destination).unwrap();
    for entry in std::fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        let kind = entry.file_type().unwrap();
        let target = destination.join(entry.file_name());
        if kind.is_dir() {
            copy_headers(&entry.path(), &target);
        } else if kind.is_file() {
            std::fs::copy(entry.path(), target).unwrap();
        } else {
            panic!("Unsupported file in native header tree: {:?}", entry.path());
        }
    }
}
