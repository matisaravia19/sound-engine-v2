//! Reviewed source transformations for the pinned native headers.

/// Pinned call counts, including commented examples, for the mechanical format rewrite.
pub const TEMP_FORMAT_HEADERS: &[(&str, usize)] = &[
    ("vkFFT_CodeGen/vkFFT_MathUtils/vkFFT_MathUtils.h", 363),
    (
        "vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_KernelStartEnd.h",
        74,
    ),
    ("vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_KernelUtils.h", 27),
    ("vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_Zeropad.h", 1),
    ("vkFFT_CodeGen/vkFFT_KernelsLevel1/vkFFT_RaderKernels.h", 46),
    ("vkFFT_CodeGen/vkFFT_KernelsLevel1/vkFFT_ReadWrite.h", 5),
    ("vkFFT_CodeGen/vkFFT_KernelsLevel1/vkFFT_RadixShuffle.h", 6),
    (
        "vkFFT_CodeGen/vkFFT_KernelsLevel1/PrePostProcessing/vkFFT_Convolution.h",
        2,
    ),
    (
        "vkFFT_CodeGen/vkFFT_KernelsLevel1/PrePostProcessing/vkFFT_R2C.h",
        1,
    ),
    (
        "vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_MemoryManagement/vkFFT_MemoryTransfers/vkFFT_Transfers.h",
        83,
    ),
    (
        "vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_MemoryManagement/vkFFT_MemoryInitialization/vkFFT_Constants.h",
        35,
    ),
    (
        "vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_MemoryManagement/vkFFT_MemoryInitialization/vkFFT_InputOutput.h",
        4,
    ),
    (
        "vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_MemoryManagement/vkFFT_MemoryInitialization/vkFFT_InputOutputLayout.h",
        11,
    ),
    (
        "vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_MemoryManagement/vkFFT_MemoryInitialization/vkFFT_PushConstants.h",
        11,
    ),
    (
        "vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_MemoryManagement/vkFFT_MemoryInitialization/vkFFT_Registers.h",
        20,
    ),
    (
        "vkFFT_CodeGen/vkFFT_KernelsLevel0/vkFFT_MemoryManagement/vkFFT_MemoryInitialization/vkFFT_SharedMemory.h",
        32,
    ),
];

/// Replaces only the reviewed destination expression; format strings/arguments stay intact.
/// A changed call count or destination form requires review of the pinned dependency.
pub fn bound_temporary_formats(source: &str, expected: usize) -> Result<String, String> {
    const CALL: &str = "sprintf(sc->tempStr,";
    let found = source.matches(CALL).count();
    if found != expected || expected == 0 {
        return Err(format!(
            "Expected {expected} temporary format calls, found {found}"
        ));
    }
    let bounded = source.replace(CALL, "PfFormatTemp(sc,");
    if bounded.contains("sprintf(sc->tempStr") {
        return Err("Unreviewed temporary format destination".into());
    }
    Ok(bounded)
}

/// Applies reviewed hunks at their original line positions and rejects changed context.
/// No fuzz or offset search is allowed: a dependency update requires patch review.
pub fn apply_header_patch(source: &str, patch: &str) -> Result<String, String> {
    let source: Vec<_> = source.lines().collect();
    let mut output = Vec::new();
    let mut cursor = 0;
    let mut in_hunk = false;
    for line in patch.lines() {
        if line.starts_with("@@ ") {
            let start = line
                .split_whitespace()
                .nth(1)
                .and_then(|range| range.strip_prefix('-'))
                .and_then(|range| range.split(',').next())
                .and_then(|start| start.parse::<usize>().ok())
                .and_then(|start| start.checked_sub(1))
                .ok_or_else(|| format!("Invalid hunk header: {line}"))?;
            if start < cursor || start > source.len() {
                return Err(format!("Overlapping or out-of-bounds hunk: {line}"));
            }
            output.extend_from_slice(&source[cursor..start]);
            cursor = start;
            in_hunk = true;
        } else if in_hunk {
            let (prefix, content) = line.split_at_checked(1).ok_or("Empty patch line")?;
            match prefix {
                " " | "-" => {
                    if source.get(cursor).copied() != Some(content) {
                        return Err(format!(
                            "Native patch context mismatch at line {}",
                            cursor + 1
                        ));
                    }
                    cursor += 1;
                    if prefix == " " {
                        output.push(content);
                    }
                }
                "+" => output.push(content),
                _ => return Err(format!("Unsupported patch line: {line}")),
            }
        } else if !line.starts_with("--- ") && !line.starts_with("+++ ") {
            return Err(format!("Unexpected patch header: {line}"));
        }
    }
    if !in_hunk {
        return Err("Native patch has no hunks".into());
    }
    output.extend_from_slice(&source[cursor..]);
    Ok(output.join("\n") + "\n")
}

#[cfg(test)]
#[path = "tests/unit/build_support.rs"]
mod tests;
