//! GPU support built on Vulkan.
//!
//! This module keeps raw Vulkan setup and resource ownership behind small
//! wrappers so simulation and auralization code can record work at a higher
//! level.

use std::error::Error;

pub mod backend;
pub mod compute;
pub mod memory;
pub mod rt;
pub mod shader;

/// Common error type for GPU setup, resource creation, and command recording.
pub type GpuError = Box<dyn Error>;
