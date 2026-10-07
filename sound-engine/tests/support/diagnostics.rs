use super::{BlockMeasurement, BlockRecord, OutputPaths, RenderMetrics};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

/// Owns a unique directory containing only this test's diagnostic outputs.
pub(super) struct TestDirectory(
    /// Directory used for custom destinations and failure injection.
    pub(super) PathBuf,
);

impl TestDirectory {
    /// Creates an empty directory for one diagnostic test.
    pub(super) fn new() -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "sound-diagnostics-{}-{}",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }

    /// Returns separate destinations for timing and stereo capture.
    pub(super) fn paths(&self) -> OutputPaths {
        OutputPaths {
            timing: Some(self.0.join("timing.csv")),
            capture: Some(self.0.join("capture.wav")),
        }
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        for name in [
            "timing.csv",
            "capture.wav",
            "timing.audio.json",
            "capture.audio.json",
        ] {
            let _ = std::fs::remove_file(self.0.join(name));
        }
        let _ = std::fs::remove_dir(&self.0);
    }
}

/// Builds fixed measurements so file-format expectations do not depend on wall clocks.
pub(super) fn record(samples: Vec<f32>) -> BlockRecord {
    BlockRecord {
        measurement: Some(BlockMeasurement {
            elapsed: 0.5,
            render_ms: 1.0,
            rms: [0.25, 0.5],
            metrics: metrics(),
        }),
        samples,
    }
}

/// Returns representative render counters shared by recording and file-format tests.
pub(super) fn metrics() -> RenderMetrics {
    RenderMetrics {
        trace_ms: 0.1,
        ir_prepare_ms: 0.2,
        convolution_ms: 0.3,
        spatial_updates: 2,
        spatial_update_age_ms: 3.0,
        spatial_deferred: false,
    }
}
