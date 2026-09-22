mod cache;
mod engine;
mod ir;
mod pipeline;

#[cfg(test)]
mod diffraction_tests;

pub use cache::{IrCache, IrCacheConfig, IrCacheQuery};
pub use engine::AcousticsEngine;
pub(crate) use engine::canonicalize_listener;
pub use ir::{GpuIrSnapshot, IrSample, IrSnapshot};
pub use pipeline::{AcousticPipeline, AcousticQuery};
