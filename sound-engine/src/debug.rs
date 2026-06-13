use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::acoustics::IrSnapshot;
use crate::error::SoundResult;

/// Writes an acoustic impulse response snapshot to a CSV debug file.
///
/// The file owns no engine resources after export; it is a plain-text snapshot
/// with metadata comments followed by `sample_index,time_seconds,amplitude`
/// rows. Existing files at `path` are replaced.
pub fn export_ir<P: AsRef<Path>>(ir: &IrSnapshot, path: P) -> SoundResult<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "# sound-engine ir export")?;
    writeln!(writer, "# sample_rate={}", ir.sample_rate)?;
    writeln!(writer, "# scene_version={}", ir.scene_version)?;
    writeln!(writer, "# query_id={}", ir.query_id)?;
    writeln!(writer, "# energy={}", ir.energy)?;
    writeln!(writer, "# samples={}", ir.samples.len())?;
    writeln!(writer, "sample_index,time_seconds,amplitude")?;

    for (sample_index, sample) in ir.samples.iter().enumerate() {
        let time_seconds = sample_index as f64 / ir.sample_rate as f64;
        writeln!(writer, "{sample_index},{time_seconds:.9},{sample:.9}")?;
    }

    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use crate::acoustics::IrSnapshot;

    use super::*;

    #[test]
    fn exports_ir_metadata_and_samples() {
        let path = std::env::temp_dir().join(format!("sound_engine_ir_export_test_{}.csv", std::process::id()));
        let ir = IrSnapshot {
            sample_rate: 2,
            samples: vec![0.0, 0.5],
            energy: 0.25,
            scene_version: 3,
            query_id: 7,
        };

        export_ir(&ir, &path).unwrap();

        let exported = fs::read_to_string(&path).unwrap();
        fs::remove_file(&path).unwrap();
        assert!(exported.contains("# sample_rate=2"));
        assert!(exported.contains("# scene_version=3"));
        assert!(exported.contains("sample_index,time_seconds,amplitude"));
        assert!(exported.contains("1,0.500000000,0.500000000"));
    }
}
