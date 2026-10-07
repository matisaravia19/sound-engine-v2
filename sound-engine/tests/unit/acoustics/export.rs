use std::fs;

use crate::acoustics::{IrSample, IrSnapshot};

use super::*;

#[test]
fn exports_ir_metadata_and_samples() {
    let path = std::env::temp_dir().join(format!(
        "sound_engine_ir_export_test_{}.csv",
        std::process::id()
    ));
    let ir = IrSnapshot {
        sample_rate: 2,
        samples: vec![IrSample::new(0.0, 0.0), IrSample::new(0.5, 0.25)],
        energy: 0.25,
        scene_version: 3,
        query_id: 7,
    };

    export_ir(&ir, &path).unwrap();

    let exported = fs::read_to_string(&path).unwrap();
    fs::remove_file(&path).unwrap();
    assert!(exported.contains("# sample_rate=2"));
    assert!(exported.contains("# scene_version=3"));
    assert!(exported.contains("sample_index,time_seconds,left,right"));
    assert!(exported.contains("1,0.500000000,0.500000000,0.250000000"));
}
