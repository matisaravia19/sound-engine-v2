use std::error::Error;
use std::ffi::NulError;
use std::fmt;
use std::sync::Arc;
use std::time::SystemTimeError;

/// Stable error categories for sound-engine failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    /// An input violates the API contract.
    InvalidArgument,
    /// The requested operation is invalid in the current state.
    InvalidState,
    /// A requested handle or resource does not exist.
    NotFound,
    /// A required device, driver, or backend is unavailable.
    BackendUnavailable,
    /// The backend does not support a requested feature.
    UnsupportedFeature,
    /// A host or device resource could not be allocated.
    ResourceAllocationFailed,
    /// Shader compilation or validation failed.
    ShaderCompileFailed,
    /// File or stream input/output failed.
    Io,
    /// A dependency reported an error.
    External,
    /// A synchronization lock was poisoned by a panic.
    PoisonedLock,
}

/// Library-wide error type. Clones retain shared ownership of the original cause.
#[derive(Debug, Clone)]
pub struct SoundError {
    code: ErrorCode,
    message: String,
    source: Option<Arc<dyn Error + Send + Sync + 'static>>,
}

/// Library-wide result alias.
pub type SoundResult<T> = Result<T, SoundError>;

impl SoundError {
    /// Creates an error without an underlying cause.
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            source: None,
        }
    }

    /// Owns a thread-safe underlying cause alongside the engine error.
    pub fn with_source(
        code: ErrorCode,
        message: impl Into<String>,
        source: impl Error + Send + Sync + 'static,
    ) -> Self {
        Self {
            code,
            message: message.into(),
            source: Some(Arc::new(source)),
        }
    }

    /// Returns the stable category for programmatic handling.
    pub fn code(&self) -> ErrorCode {
        self.code
    }

    /// Reports an input-contract violation.
    pub fn invalid_argument(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::InvalidArgument, message)
    }

    /// Reports an operation incompatible with current state.
    pub fn invalid_state(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::InvalidState, message)
    }

    /// Reports a missing handle or resource.
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::NotFound, message)
    }

    /// Reports an unavailable backend or device.
    pub fn backend_unavailable(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::BackendUnavailable, message)
    }

    /// Reports an unsupported backend capability.
    pub fn unsupported_feature(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::UnsupportedFeature, message)
    }

    /// Reports a failed host or device allocation.
    pub fn resource_allocation_failed(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::ResourceAllocationFailed, message)
    }

    /// Reports a failed shader compilation.
    pub fn shader_compile_failed(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::ShaderCompileFailed, message)
    }

    /// Reports file or stream input/output failure.
    pub fn io(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::Io, message)
    }

    /// Wraps a dependency error message.
    pub fn external(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::External, message)
    }

    /// Reports a lock poisoned by a prior panic.
    pub fn poisoned_lock(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::PoisonedLock, message)
    }
}

impl fmt::Display for SoundError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:?}: {}", self.code, self.message)
    }
}

impl Error for SoundError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source
            .as_deref()
            .map(|source| source as &(dyn Error + 'static))
    }
}

impl From<std::io::Error> for SoundError {
    fn from(source: std::io::Error) -> Self {
        Self::with_source(ErrorCode::Io, source.to_string(), source)
    }
}

impl From<NulError> for SoundError {
    fn from(source: NulError) -> Self {
        Self::with_source(ErrorCode::InvalidArgument, source.to_string(), source)
    }
}

impl From<SystemTimeError> for SoundError {
    fn from(source: SystemTimeError) -> Self {
        Self::with_source(ErrorCode::External, source.to_string(), source)
    }
}

impl From<ash::LoadingError> for SoundError {
    fn from(source: ash::LoadingError) -> Self {
        Self::with_source(ErrorCode::BackendUnavailable, source.to_string(), source)
    }
}

impl From<ash::vk::Result> for SoundError {
    fn from(source: ash::vk::Result) -> Self {
        Self::external(format!("Vulkan call failed: {source:?}"))
    }
}

/// Returns a categorized error when an API precondition fails in any build.
#[macro_export]
macro_rules! validate {
    ($condition:expr, $code:expr, $($message:tt)+) => {{
        if !$condition {
            return Err($crate::core::error::SoundError::new($code, format!($($message)+)));
        }
    }};
}
