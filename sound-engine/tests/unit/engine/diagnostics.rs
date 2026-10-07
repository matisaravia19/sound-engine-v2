use super::test_support::{TestDirectory, metrics, record};
use super::*;

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
    assert_eq!(rows[1], "0,0.5,1,0.25,0.5,0.1,0.2,0.3,2,3,0");
    assert_eq!(rows[2], "1,0.5,1,0.25,0.5,0.1,0.2,0.3,2,3,0");
    let mut wav = hound::WavReader::open(paths.capture.unwrap()).unwrap();
    assert_eq!(wav.spec().sample_rate, 48_000);
    assert_eq!(wav.spec().channels, 2);
    assert_eq!(
        wav.samples::<f32>().collect::<Result<Vec<_>, _>>().unwrap(),
        [0.25, -0.25, 0.5, -0.5]
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
fn runtime_records_optional_outputs_and_saves_metadata_after_flushing() {
    for (channels, timing, capture) in [(1, true, false), (2, false, true), (2, true, true)] {
        let dir = TestDirectory::new();
        let mut paths = dir.paths();
        if !timing {
            paths.timing = None;
        }
        if !capture {
            paths.capture = None;
        }
        let samples = if channels == 1 {
            [0.25, -0.25, 0.25, -0.25]
        } else {
            [0.25, -0.5, 0.25, -0.5]
        };
        let (player, _control) = SoundPlayer::test_player(samples.len());
        let mut diagnostics = RuntimeDiagnostics {
            player: &player,
            recorder: Some(Recorder::new(paths.clone(), 48_000, channels, samples.len()).unwrap()),
            started: Instant::now(),
            channels,
        };
        diagnostics.record(&samples, Instant::now(), metrics());
        drop(diagnostics);

        let metadata = std::fs::read_to_string(paths.summary_path()).unwrap();
        let captured_samples = if capture {
            samples.len()
        } else {
            0
        };
        assert_eq!(
            metadata,
            format!(
                "{{\"underrun_callbacks\":0,\"missing_frames\":0,\
                 \"first_underrun_seconds_since_player_creation\":null,\
                 \"last_underrun_seconds_since_player_creation\":null,\
                 \"diagnostic_blocks_written\":1,\"diagnostic_blocks_omitted\":0,\
                 \"capture_samples_written\":{captured_samples},\
                 \"capture_truncated\":false,\"diagnostic_write_failed\":false}}\n"
            )
        );
        if let Some(path) = paths.timing {
            let csv = std::fs::read_to_string(path).unwrap();
            let rows = csv.lines().collect::<Vec<_>>();
            assert_eq!(rows.len(), 2);
            let values = rows[1]
                .split(',')
                .map(|value| value.parse::<f64>().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(values[0], 0.0);
            assert!(
                values[1..3]
                    .iter()
                    .all(|time| time.is_finite() && *time >= 0.0)
            );
            let right_rms = if channels == 1 {
                0.0
            } else {
                0.5
            };
            assert_eq!(
                &values[3..],
                [0.25, right_rms, 0.1, 0.2, 0.3, 2.0, 3.0, 0.0]
            );
        }
        if let Some(path) = paths.capture {
            let mut wav = hound::WavReader::open(path).unwrap();
            assert_eq!(wav.spec().channels as usize, channels);
            assert_eq!(wav.spec().sample_rate, 48_000);
            assert_eq!(
                wav.samples::<f32>().collect::<Result<Vec<_>, _>>().unwrap(),
                samples
            );
        }
        assert_eq!(dir.paths().timing.unwrap().exists(), timing);
        assert_eq!(dir.paths().capture.unwrap().exists(), capture);
    }
}
