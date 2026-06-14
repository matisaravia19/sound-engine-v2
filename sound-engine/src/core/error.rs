use std::error::Error;
use std::ffi::NulError;
use std::fmt;
use std::time::SystemTimeError;

/// Stable error categories for sound-engine failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    InvalidArgument,
    InvalidState,
    NotFound,
    BackendUnavailable,
    UnsupportedFeature,
    ResourceAllocationFailed,
    ShaderCompileFailed,
    Io,
    External,
    PoisonedLock,
}

/// Library-wide error type.
#[derive(Debug)]
pub struct SoundError {
    code: ErrorCode,
    message: String,
    source: Option<Box<dyn Error + Send + Sync + 'static>>,
}

/// Library-wide result alias.
pub type SoundResult<T> = Result<T, SoundError>;

impl SoundError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            source: None,
        }
    }

    pub fn with_source(
        code: ErrorCode,
        message: impl Into<String>,
        source: impl Error + Send + Sync + 'static,
    ) -> Self {
        Self {
            code,
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    pub fn code(&self) -> ErrorCode {
        self.code
    }

    pub fn invalid_argument(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::InvalidArgument, message)
    }

    pub fn invalid_state(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::InvalidState, message)
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::NotFound, message)
    }

    pub fn backend_unavailable(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::BackendUnavailable, message)
    }

    pub fn unsupported_feature(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::UnsupportedFeature, message)
    }

    pub fn resource_allocation_failed(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::ResourceAllocationFailed, message)
    }

    pub fn shader_compile_failed(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::ShaderCompileFailed, message)
    }

    pub fn io(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::Io, message)
    }

    pub fn external(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::External, message)
    }

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
        self.source.as_deref().map(|source| source as &(dyn Error + 'static))
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

#[macro_export]
macro_rules! debug_validate {
    ($condition:expr, $code:expr, $($message:tt)+) => {{
        #[cfg(debug_assertions)]
        {
            if !$condition {
                return Err($crate::core::error::SoundError::new($code, format!($($message)+)));
            }
        }
    }};
}
