//! Vulkan FFT plans with explicit external resource lifetimes.
#[cfg(test)]
#[path = "../build_support.rs"]
mod build_support;
mod ffi;
/// Plan configuration, ownership, and command recording.
pub mod plan;
