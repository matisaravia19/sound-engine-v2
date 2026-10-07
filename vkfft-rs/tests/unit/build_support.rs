use super::*;

#[test]
fn format_rewrite_preserves_arguments_and_rejects_dependency_drift() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("lib/vkfft/vkFFT/vkFFT");
    for &(header, count) in TEMP_FORMAT_HEADERS {
        let source = std::fs::read_to_string(root.join(header)).unwrap();
        bound_temporary_formats(&source, count).unwrap();
    }
    assert_eq!(
        bound_temporary_formats("x = sprintf(sc->tempStr, \"%s:%d\", name, n);", 1).unwrap(),
        "x = PfFormatTemp(sc, \"%s:%d\", name, n);"
    );
    assert!(bound_temporary_formats("sprintf(sc->tempStr, x)", 2).is_err());
    assert!(
        bound_temporary_formats("sprintf(sc->tempStr, x); sprintf(sc->tempStr + 1, x)", 1).is_err()
    );
}

#[test]
fn patches_only_exact_context_and_rejects_changed_dependency() {
    let patch = "--- a\n+++ b\n@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n";
    assert_eq!(
        apply_header_patch("a\r\nb\r\nc\r\n", patch).unwrap(),
        "a\nB\nc\n"
    );
    assert!(apply_header_patch("a\nchanged\nc\n", patch).is_err());
    assert!(apply_header_patch("a\nb\nc\n", "").is_err());
}
