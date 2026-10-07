use super::*;

#[test]
fn status_conversion_preserves_unknown_native_errors() {
    assert_eq!(FftError::check(0), Ok(()));
    assert_eq!(FftError::check(9876).unwrap_err().code(), 9876);
    assert_eq!(FftError::check(-1).unwrap_err().code(), -1);
}
