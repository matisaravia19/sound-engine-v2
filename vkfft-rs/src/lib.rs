//! Power-of-two audio FFT plans with explicit external resource lifetimes.
#![warn(missing_docs)]

#[cfg(test)]
#[path = "../build_support.rs"]
mod build_support;
mod error;
mod ffi;

#[cfg(all(test, feature = "native-tests"))]
mod native_tests;

pub use error::FftError;
/// Plan configuration, ownership, and command recording.
pub mod plan;
