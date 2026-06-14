//! Sound engine prototype with GPU acceleration for convolution and ray tracing.

pub mod acoustics;
pub mod auralization;
pub mod core;
pub mod gpu;
pub mod scene;

/// Placeholder helper kept by the initial crate template.
pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
