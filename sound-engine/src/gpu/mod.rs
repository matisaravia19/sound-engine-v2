//! GPU support built on Vulkan.
//!
//! This module keeps raw Vulkan setup and resource ownership behind small
//! wrappers so simulation and auralization code can record work at a higher
//! level.

/// Vulkan instance, device, and queue ownership.
pub mod backend;
/// Compute pipelines, command submission, and synchronization.
pub mod compute;
/// GPU allocations, mapped transfers, and buffer lifetime management.
pub mod memory;
/// Acceleration structures, shader binding tables, and ray dispatch.
pub mod rt;
/// GLSL compilation and owned Vulkan shader modules.
pub mod shader;
