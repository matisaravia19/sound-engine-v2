//! Shared scene builders and measured-data fixtures for benchmarks and the sandbox.
//! This crate is development tooling; the engine does not depend on it at runtime.
#![warn(missing_docs)]

pub mod bell_box;
pub mod bras;
pub mod geometry;

mod text;
pub use text::{PositionPair, read_positions, read_scene};
