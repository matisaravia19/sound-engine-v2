use super::*;
use std::sync::atomic::{AtomicU64, Ordering};

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "sound-diagnostics-{}-{}",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn paths(&self) -> OutputPaths {
        OutputPaths {
            timing: Some(self.0.join("timing.csv")),
            capture: Some(self.0.join("capture.wav")),
        }
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        for name in ["timing.csv", "capture.wav"] {
            let _ = std::fs::remove_file(self.0.join(name));
        }
        let _ = std::fs::remove_dir(&self.0);
    }
}

fn record(samples: Vec<f32>) -> BlockRecord {
    BlockRecord {
        measurement: Some(BlockMeasurement {
            elapsed: 0.5,
            render_ms: 1.0,
            rms: [0.25, 0.5],
            stages: [0.0, 0.0, 1.0],
            spatial_updates: 2,
            spatial_update_age_ms: 3.0,
            spatial_deferred: false,
        }),
        samples,
    }
}

#[test]
fn full_diagnostic_queue_stops_at_a_prefix_without_waiting_or_resuming() {
    let (sender, receiver) = mpsc::sync_channel(1);
    let mut queue = RecordQueue {
        sender: Some(sender),
        accepted: 0,
        omitted: 0,
    };
    queue.push(|| record(vec![1.0, -1.0]));
    queue.push(|| record(vec![2.0, -2.0]));
    assert_eq!(receiver.recv().unwrap().samples, [1.0, -1.0]);
    queue.push(|| panic!("stopped diagnostics must not allocate or compute another record"));
    assert!(receiver.recv().is_err());
    assert_eq!((queue.accepted, queue.omitted), (1, 2));
}

#[test]
fn disconnected_writer_stops_retention() {
    let (sender, receiver) = mpsc::sync_channel(1);
    drop(receiver);
    let mut queue = RecordQueue {
        sender: Some(sender),
        accepted: 0,
        omitted: 0,
    };
    queue.push(|| record(vec![0.0, 0.0]));
    assert!(queue.sender.is_none());
    assert_eq!(queue.omitted, 1);
}

#[test]
fn asynchronous_writer_flushes_complete_timing_and_audio() {
    let dir = TestDirectory::new();
    let paths = dir.paths();
    let mut recorder = Recorder::new(paths.clone(), 48_000, 2, 2).unwrap();
    recorder.queue.push(|| record(vec![0.25, -0.25]));
    recorder.queue.push(|| record(vec![0.5, -0.5]));
    let (summary, omitted, error) = recorder.finish();
    assert!(error.is_none());
    assert_eq!(omitted, 0);
    assert_eq!(summary.blocks_written, 2);
    assert_eq!(summary.capture_samples_written, 4);
    assert!(!summary.capture_truncated);
    let csv = std::fs::read_to_string(paths.timing.unwrap()).unwrap();
    let rows = csv.lines().collect::<Vec<_>>();
    assert_eq!(rows.len(), 3);
    assert!(rows[0].starts_with("block,elapsed_seconds,render_ms"));
    assert!(rows[1].starts_with("0,0.5,1,0.25,0.5,"));
    assert!(rows[2].starts_with("1,0.5,1,0.25,0.5,"));
    let mut wav = hound::WavReader::open(paths.capture.unwrap()).unwrap();
    assert_eq!(wav.spec().sample_rate, 48_000);
    assert_eq!(wav.spec().channels, 2);
    assert_eq!(
        wav.samples::<f32>().collect::<Result<Vec<_>, _>>().unwrap(),
        [0.25, -0.25, 0.5, -0.5]
    );
}

#[test]
fn wav_capacity_keeps_complete_frames_and_continues_timing() {
    let dir = TestDirectory::new();
    let paths = dir.paths();
    let mut writer = OutputWriter::new(&paths, 48_000, 2).unwrap();
    writer.max_capture_samples = 3;
    writer.write(record(vec![0.25, -0.25])).unwrap();
    writer.write(record(vec![0.5, -0.5])).unwrap();
    writer.write(record(vec![0.75, -0.75])).unwrap();
    writer.finish().unwrap();
    assert!(writer.summary.capture_truncated);
    assert_eq!(writer.summary.capture_samples_written, 2);
    assert_eq!(writer.summary.blocks_written, 3);
    let mut wav = hound::WavReader::open(paths.capture.unwrap()).unwrap();
    assert_eq!(
        wav.samples::<f32>().collect::<Result<Vec<_>, _>>().unwrap(),
        [0.25, -0.25]
    );
    assert_eq!(
        std::fs::read_to_string(paths.timing.unwrap())
            .unwrap()
            .lines()
            .count(),
        4
    );
}

#[test]
fn invalid_layout_and_oversized_capture_blocks_fail_before_opening_outputs() {
    let dir = TestDirectory::new();
    for (channels, samples) in [
        (0, 2),
        (3, 3),
        (2, 3),
        (2, QUEUE_BYTES),
        (2, usize::MAX - 1),
    ] {
        assert!(Recorder::new(dir.paths(), 48_000, channels, samples).is_err());
    }
    assert!(!dir.paths().capture.unwrap().exists());
    assert!(!dir.paths().timing.unwrap().exists());
}

#[test]
fn output_open_failure_is_reported() {
    let dir = TestDirectory::new();
    let paths = OutputPaths {
        timing: Some(dir.0.join("missing/record.csv")),
        capture: None,
    };
    assert!(Recorder::new(paths, 48_000, 2, 2).is_err());
}

#[test]
fn colliding_destinations_are_rejected_without_truncating_existing_files() {
    let dir = TestDirectory::new();
    let path = dir.0.join("timing.csv");
    std::fs::write(&path, "keep this content").unwrap();
    let aliases = OutputPaths {
        timing: Some(path.clone()),
        capture: Some(dir.0.join("./timing.csv")),
    };
    assert!(Recorder::new(aliases, 48_000, 2, 2).is_err());
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "keep this content");
    let metadata_collision = OutputPaths {
        timing: Some(path.clone()),
        capture: Some(path.with_extension("audio.json")),
    };
    assert!(Recorder::new(metadata_collision, 48_000, 2, 2).is_err());
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "keep this content");
}

#[test]
fn timing_flush_error_still_finalizes_the_audio_header() {
    let dir = TestDirectory::new();
    let paths = dir.paths();
    let mut writer = OutputWriter::new(&paths, 48_000, 2).unwrap();
    writer.timing.as_mut().unwrap().flush().unwrap();
    writer.timing = Some(BufWriter::new(
        File::open(paths.timing.as_ref().unwrap()).unwrap(),
    ));
    writer.write(record(vec![0.25, -0.25])).unwrap();
    assert!(writer.finish().is_err());
    let mut wav = hound::WavReader::open(paths.capture.unwrap()).unwrap();
    assert_eq!(
        wav.samples::<f32>().collect::<Result<Vec<_>, _>>().unwrap(),
        [0.25, -0.25]
    );
}
