use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/ffi.cpp");
    println!("cargo:rerun-if-changed=build.rs");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    let vulkan_sdk = env::var("VULKAN_SDK")
        .expect("VULKAN_SDK environment variable not set. Install Vulkan SDK.");

    let vulkan_include = PathBuf::from(&vulkan_sdk).join("Include");
    let glslang_include = vulkan_include.join("glslang").join("Include");
    let vkfft_include = manifest_dir.join("lib").join("vkfft").join("vkFFT");

    let mut build = cc::Build::new();
    build.cpp(true);
    build.file("src/ffi.cpp");

    build.include(&vulkan_include).include(&glslang_include).include(&vkfft_include);

    build.flag_if_supported("-std=c++17");

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