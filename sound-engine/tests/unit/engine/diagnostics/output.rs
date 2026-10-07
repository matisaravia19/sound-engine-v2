use super::super::test_support::{TestDirectory, record};
use super::*;

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
fn output_open_failure_is_reported() {
    let dir = TestDirectory::new();
    let paths = OutputPaths {
        timing: Some(dir.0.join("missing/record.csv")),
        capture: None,
    };
    assert!(OutputWriter::new(&paths, 48_000, 2).is_err());
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
    assert!(OutputWriter::new(&aliases, 48_000, 2).is_err());
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "keep this content");
    let metadata_collision = OutputPaths {
        timing: Some(path.clone()),
        capture: Some(path.with_extension("audio.json")),
    };
    assert!(OutputWriter::new(&metadata_collision, 48_000, 2).is_err());
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

#[test]
fn metadata_preserves_underruns_and_incomplete_capture_counts() {
    let dir = TestDirectory::new();
    let path = dir.paths().summary_path();
    let summary = WriterSummary {
        blocks_written: 3,
        capture_samples_written: 4,
        capture_truncated: true,
    };
    summary
        .write_metadata(
            &path,
            PlaybackSummary {
                underruns: 2,
                missing: 64,
                starvation_times: Some((0.125, 0.5)),
            },
            7,
            true,
        )
        .unwrap();
    assert_eq!(
        std::fs::read_to_string(path).unwrap(),
        "{\"underrun_callbacks\":2,\"missing_frames\":64,\
         \"first_underrun_seconds_since_player_creation\":0.125,\
         \"last_underrun_seconds_since_player_creation\":0.5,\
         \"diagnostic_blocks_written\":3,\"diagnostic_blocks_omitted\":7,\
         \"capture_samples_written\":4,\"capture_truncated\":true,\
         \"diagnostic_write_failed\":true}\n"
    );
}
