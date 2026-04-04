use std::error::Error;

pub mod backend;
pub mod memory;
pub mod shader;
pub mod sync;

pub type GpuError = Box<dyn Error>;
