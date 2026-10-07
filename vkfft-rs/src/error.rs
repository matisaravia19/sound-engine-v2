//! Native VkFFT status codes without assumptions about the set of returned values.

use std::fmt;

/// A nonzero status returned by VkFFT or by plan configuration validation.
/// Unknown native codes are retained so dependency changes remain representable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FftError(i32);

impl FftError {
    /// Native allocation failed before an application was created.
    pub(crate) const EMPTY_APP: Self = Self(2015);
    /// No physical device was supplied.
    pub(crate) const INVALID_PHYSICAL_DEVICE: Self = Self(1001);
    /// No logical device was supplied.
    pub(crate) const INVALID_DEVICE: Self = Self(1002);
    /// No initialization queue was supplied.
    pub(crate) const INVALID_QUEUE: Self = Self(1003);
    /// No initialization command pool was supplied.
    pub(crate) const INVALID_COMMAND_POOL: Self = Self(1004);
    /// No initialization fence was supplied.
    pub(crate) const INVALID_FENCE: Self = Self(1005);
    /// The transform has zero length.
    pub(crate) const EMPTY_SIZE: Self = Self(2002);
    /// The transform length is not supported by this wrapper.
    pub(crate) const UNSUPPORTED_LENGTH: Self = Self(3002);
    /// The transform buffer is smaller than the requested data extent.
    pub(crate) const EMPTY_BUFFER_SIZE: Self = Self(2003);
    /// No transform buffer was supplied.
    pub(crate) const EMPTY_BUFFER: Self = Self(2004);
    /// The requested transform extent overflows its byte representation.
    pub(crate) const INSUFFICIENT_STORAGE: Self = Self(3);

    /// Returns the unmodified native status code.
    pub fn code(self) -> i32 {
        self.0
    }

    /// Converts the C status convention into a Rust result, retaining unknown codes.
    pub(crate) fn check(status: i32) -> Result<(), Self> {
        if status == 0 {
            Ok(())
        } else {
            Err(Self(status))
        }
    }
}

impl fmt::Display for FftError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let reason = match *self {
            Self::EMPTY_APP => "application allocation failed",
            Self::INVALID_PHYSICAL_DEVICE => "missing physical device",
            Self::INVALID_DEVICE => "missing logical device",
            Self::INVALID_QUEUE => "missing initialization queue",
            Self::INVALID_COMMAND_POOL => "missing initialization command pool",
            Self::INVALID_FENCE => "missing initialization fence",
            Self::EMPTY_SIZE => "zero transform length",
            Self::UNSUPPORTED_LENGTH => "unsupported transform length",
            Self::EMPTY_BUFFER => "missing transform buffer",
            Self::EMPTY_BUFFER_SIZE => "insufficient transform buffer capacity",
            Self::INSUFFICIENT_STORAGE => "transform storage exceeds representable bounds",
            _ => "native operation failed",
        };
        write!(formatter, "VkFFT {reason} (status {})", self.0)
    }
}

impl std::error::Error for FftError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_conversion_preserves_unknown_native_errors() {
        assert_eq!(FftError::check(0), Ok(()));
        assert_eq!(FftError::check(9876).unwrap_err().code(), 9876);
        assert_eq!(FftError::check(-1).unwrap_err().code(), -1);
    }
}
