//! Power-of-two audio FFT plans with explicit external resource lifetimes.
#![warn(missing_docs)]

mod error;
mod ffi;

pub use error::FftError;
/// Plan configuration, ownership, and command recording.
pub mod plan;

#[cfg(test)]
#[path = "../build_support.rs"]
mod build_support;

#[cfg(all(test, feature = "native-tests"))]
#[path = "../tests/unit/native.rs"]
mod native_tests;
