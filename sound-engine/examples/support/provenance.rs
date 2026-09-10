//! Runtime provenance for benchmark CSVs.
use std::process::Command;

/// Identifies the checkout actually being measured, including tracked edits.
/// Returns `unknown` outside a Git checkout rather than inventing a revision.
pub fn revision() -> String {
    let run = |args: &[&str]| {
        Command::new("git")
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .args(args)
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
    };
    let Some(mut revision) = run(&["rev-parse", "HEAD"]) else {
        return "unknown".into();
    };
    if run(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|status| !status.is_empty()) {
        revision.push_str("+dirty");
    }
    revision
}
