use std::error::Error;

pub mod backend;
pub mod compute;
pub mod memory;
pub mod shader;

pub type GpuError = Box<dyn Error>;
