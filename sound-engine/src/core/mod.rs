/// Engine configuration and acoustic budgets.
pub mod config;
/// Error categories, conversions, and runtime validation.
pub mod error;
/// Checked identifier allocation shared by runtime resource registries.
pub(crate) mod id;

/// Shared CPU geometry validation for scenes and GPU build inputs.
pub(crate) mod geometry;
