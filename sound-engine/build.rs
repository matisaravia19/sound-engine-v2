use std::{env, fs, io, path::Path};

// Retain the exact shader source tree associated with the host executable.
// GLSL is still compiled by the existing runtime compiler, but changing an
// include must rebuild Rust and a stale executable must reject new source files.
fn collect_shaders(directory: &Path, root: &Path, paths: &mut Vec<String>) -> io::Result<()> {
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_shaders(&path, root, paths)?;
        } else if path
            .extension()
            .is_some_and(|extension| extension == "glsl")
        {
            paths.push(
                path.strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/"),
            );
        }
    }
    Ok(())
}

fn main() -> io::Result<()> {
    let root = env::var_os("CARGO_MANIFEST_DIR").expect("Cargo supplies the manifest directory");
    let root = Path::new(&root);
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=build.rs");
    let mut paths = Vec::new();
    collect_shaders(&root.join("src"), root, &mut paths)?;
    paths.sort();
    let mut manifest = String::from("const BUNDLED_SHADER_SOURCES: &[(&str, &str)] = &[\n");
    for path in paths {
        manifest.push_str(&format!(
            "({path:?}, include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/\", {path:?}))),\n"
        ));
    }
    manifest.push_str("];\n");
    fs::write(
        Path::new(&env::var_os("OUT_DIR").unwrap()).join("shader_sources.rs"),
        manifest,
    )
}
