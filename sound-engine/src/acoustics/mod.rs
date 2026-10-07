mod cache;
mod config;
mod engine;
mod export;
mod ir;
mod pipeline;
mod spatial;

pub use cache::{IrCache, IrCacheConfig};
pub use config::AcousticPipelineConfig;
pub use engine::AcousticsEngine;
pub use export::export_ir;
pub use ir::{GpuIrSnapshot, IrSample, IrSnapshot};
pub use pipeline::AcousticPipeline;
pub use spatial::{AcousticQuery, ListenerPose, PointSource};
pub(crate) use spatial::{canonicalize_listener, validate_source};

#[cfg(test)]
#[path = "../../tests/unit/acoustics/diffraction.rs"]
mod diffraction_tests;
