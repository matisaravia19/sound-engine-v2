//! Optional runtime diagnostics with bounded retention and off-thread file I/O.

use super::direct::RenderMetrics;
use crate::playback::SoundPlayer;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, SyncSender};
use std::thread::{self, JoinHandle};
use std::time::Instant;

// Include space for the record being produced and the one being written.
const QUEUE_BYTES: usize = 8 * 1024 * 1024;
const MAX_QUEUED_BLOCKS: usize = 1024;
// RIFF uses 32-bit chunk sizes; reserve ample room for hound's format headers.
const MAX_WAV_SAMPLES: u64 = (u32::MAX as u64 - 4096) / size_of::<f32>() as u64;

/// Owns a diagnostic writer while borrowing the player's final underrun counters.
/// Dropping closes its bounded queue and joins the writer before saving metadata.
pub(super) struct RuntimeDiagnostics<'a> {
    player: &'a SoundPlayer,
    recorder: Option<Recorder>,
    started: Instant,
    channels: usize,
}

impl<'a> RuntimeDiagnostics<'a> {
    /// Starts requested environment-configured outputs; failures leave playback usable.
    pub(super) fn new(
        player: &'a SoundPlayer,
        sample_rate: u32,
        channels: usize,
        block_samples: usize,
    ) -> Self {
        let paths = OutputPaths {
            timing: std::env::var_os("SOUND_ENGINE_TIMING_CSV").map(PathBuf::from),
            capture: std::env::var_os("SOUND_ENGINE_CAPTURE_WAV").map(PathBuf::from),
        };
        let recorder = if paths.timing.is_none() && paths.capture.is_none() {
            None
        } else {
            match Recorder::new(paths, sample_rate, channels, block_samples) {
                Ok(recorder) => Some(recorder),
                Err(error) => {
                    eprintln!("Could not start runtime diagnostics: {error}");
                    None
                }
            }
        };
        Self {
            player,
            recorder,
            started: Instant::now(),
            channels,
        }
    }

    /// Records one rendered block without waiting for file writes or queue space.
    pub(super) fn record(
        &mut self,
        samples: &[f32],
        render_started: Instant,
        metrics: RenderMetrics,
    ) {
        let Some(recorder) = &mut self.recorder else {
            return;
        };
        let timing = recorder.paths.timing.is_some();
        let capture = recorder.paths.capture.is_some();
        recorder.queue.push(|| {
            let measurement = timing.then(|| {
                let elapsed = self.started.elapsed().as_secs_f64();
                let render_ms = render_started.elapsed().as_secs_f64() * 1000.0;
                let mut squares = [0.0_f64; 2];
                for frame in samples.chunks_exact(self.channels) {
                    for (channel, sample) in frame.iter().enumerate() {
                        squares[channel] += (*sample as f64).powi(2);
                    }
                }
                let frames = samples.len() / self.channels;
                BlockMeasurement {
                    elapsed,
                    render_ms,
                    rms: squares.map(|sum| (sum / frames as f64).sqrt()),
                    metrics,
                }
            });
            BlockRecord {
                measurement,
                samples: if capture {
                    samples.to_vec()
                } else {
                    Vec::new()
                },
            }
        });
    }
}

impl Drop for RuntimeDiagnostics<'_> {
    fn drop(&mut self) {
        let Some(recorder) = self.recorder.take() else {
            return;
        };
        // Snapshot before joining: device callbacks can still run while files flush.
        let (underruns, missing) = self.player.starvation_counts();
        let starvation_times = self.player.starvation_time_range_seconds();
        let summary_path = recorder.paths.summary_path();
        let (summary, omitted, error) = recorder.finish();
        if let Some(error) = &error {
            eprintln!("Runtime diagnostics incomplete: {error}");
        }
        if omitted > 0 || summary.capture_truncated {
            eprintln!(
                "Runtime diagnostics retained a prefix: {omitted} omitted blocks; WAV limit reached: {}",
                summary.capture_truncated
            );
        }
        let result = std::fs::write(
            summary_path,
            format!(
                "{{\"underrun_callbacks\":{underruns},\
                 \"missing_frames\":{missing},\
                 \"first_underrun_seconds_since_player_creation\":{},\
                 \"last_underrun_seconds_since_player_creation\":{},\
                 \"diagnostic_blocks_written\":{},\
                 \"diagnostic_blocks_omitted\":{omitted},\
                 \"capture_samples_written\":{},\
                 \"capture_truncated\":{},\
                 \"diagnostic_write_failed\":{}}}\n",
                starvation_times.map_or_else(|| "null".to_owned(), |(first, _)| first.to_string()),
                starvation_times.map_or_else(|| "null".to_owned(), |(_, last)| last.to_string()),
                summary.blocks_written,
                summary.capture_samples_written,
                summary.capture_truncated,
                error.is_some(),
            ),
        );
        if let Err(error) = result {
            eprintln!("Could not save runtime diagnostic metadata: {error}");
        }
    }
}

#[derive(Clone)]
struct OutputPaths {
    timing: Option<PathBuf>,
    capture: Option<PathBuf>,
}

impl OutputPaths {
    fn validate(&self) -> io::Result<()> {
        if self.timing.is_none() && self.capture.is_none() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "No diagnostic output configured",
            ));
        }
        let mut paths = Vec::new();
        for path in self
            .timing
            .iter()
            .chain(self.capture.iter())
            .chain(std::iter::once(&self.summary_path()))
        {
            let key = output_path_key(path)?;
            if paths.contains(&key) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Timing, WAV and metadata destinations must be distinct",
                ));
            }
            paths.push(key);
        }
        Ok(())
    }

    fn summary_path(&self) -> PathBuf {
        self.timing
            .as_ref()
            .or(self.capture.as_ref())
            .expect("one diagnostic output is enabled")
            .with_extension("audio.json")
    }
}

// Resolve existing files and existing parent directories before any output is
// truncated. Windows destinations must also compare without case distinctions.
fn output_path_key(path: &Path) -> io::Result<String> {
    let absolute = std::path::absolute(path)?;
    let normalized = match absolute.canonicalize() {
        Ok(existing) => existing,
        Err(_) => {
            let parent = absolute.parent().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Diagnostic destination has no parent",
                )
            })?;
            let name = absolute.file_name().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Diagnostic destination must be a file",
                )
            })?;
            parent.canonicalize()?.join(name)
        }
    };
    Ok(if cfg!(windows) {
        normalized.to_string_lossy().to_lowercase()
    } else {
        normalized.to_string_lossy().into_owned()
    })
}

struct BlockMeasurement {
    elapsed: f64,
    render_ms: f64,
    rms: [f64; 2],
    metrics: RenderMetrics,
}

struct BlockRecord {
    measurement: Option<BlockMeasurement>,
    samples: Vec<f32>,
}

/// Stops at the first dropped block so captured audio never hides a time gap.
struct RecordQueue {
    sender: Option<SyncSender<BlockRecord>>,
    accepted: u64,
    omitted: u64,
}

impl RecordQueue {
    fn push(&mut self, make_record: impl FnOnce() -> BlockRecord) {
        if self
            .sender
            .as_ref()
            .is_some_and(|sender| sender.try_send(make_record()).is_ok())
        {
            self.accepted += 1;
        } else {
            self.sender.take();
            self.omitted += 1;
        }
    }
}

struct Recorder {
    paths: OutputPaths,
    queue: RecordQueue,
    worker: JoinHandle<(WriterSummary, io::Result<()>)>,
}

impl Recorder {
    fn new(
        paths: OutputPaths,
        sample_rate: u32,
        channels: usize,
        block_samples: usize,
    ) -> io::Result<Self> {
        if channels == 0
            || channels > 2
            || sample_rate == 0
            || block_samples == 0
            || !block_samples.is_multiple_of(channels)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid diagnostic audio shape",
            ));
        }
        let sample_bytes = if paths.capture.is_some() {
            block_samples.checked_mul(size_of::<f32>())
        } else {
            Some(0)
        };
        let record_bytes = sample_bytes
            .and_then(|bytes| bytes.checked_add(size_of::<BlockRecord>()))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Diagnostic block size overflow",
                )
            })?;
        let capacity = (QUEUE_BYTES / record_bytes)
            .saturating_sub(2)
            .min(MAX_QUEUED_BLOCKS);
        if capacity == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "One diagnostic block exceeds the recording memory budget",
            ));
        }
        paths.validate()?;
        // Opening files happens before playback produces blocks, not in its callback.
        let mut outputs = OutputWriter::new(&paths, sample_rate, channels)?;
        let (sender, receiver) = mpsc::sync_channel(capacity);
        let worker = thread::Builder::new()
            .name("sound-engine-diagnostics".into())
            .spawn(move || {
                let result = (|| {
                    for record in receiver {
                        outputs.write(record)?;
                    }
                    outputs.finish()
                })();
                (outputs.summary, result)
            })?;
        Ok(Self {
            paths,
            queue: RecordQueue {
                sender: Some(sender),
                accepted: 0,
                omitted: 0,
            },
            worker,
        })
    }

    fn finish(mut self) -> (WriterSummary, u64, Option<io::Error>) {
        self.queue.sender.take();
        match self.worker.join() {
            Ok((summary, result)) => {
                let omitted =
                    self.queue.omitted + self.queue.accepted.saturating_sub(summary.blocks_written);
                (summary, omitted, result.err())
            }
            Err(_) => (
                WriterSummary::default(),
                self.queue.omitted + self.queue.accepted,
                Some(io::Error::other("Diagnostic writer panicked")),
            ),
        }
    }
}

#[derive(Default)]
struct WriterSummary {
    blocks_written: u64,
    capture_samples_written: u64,
    capture_truncated: bool,
}

struct OutputWriter {
    timing: Option<BufWriter<File>>,
    capture: Option<hound::WavWriter<BufWriter<File>>>,
    summary: WriterSummary,
    max_capture_samples: u64,
}

impl OutputWriter {
    fn new(paths: &OutputPaths, sample_rate: u32, channels: usize) -> io::Result<Self> {
        let timing = paths
            .timing
            .as_ref()
            .map(|path| {
                let mut file = BufWriter::new(File::create(path)?);
                writeln!(
                    file,
                    concat!(
                        "block,elapsed_seconds,render_ms,left_rms,right_rms,",
                        "trace_ms,ir_prepare_ms,convolution_ms,",
                        "spatial_updates,spatial_update_age_ms,spatial_deferred",
                    )
                )?;
                Ok::<_, io::Error>(file)
            })
            .transpose()?;
        let capture = paths
            .capture
            .as_ref()
            .map(|path| {
                hound::WavWriter::create(
                    path,
                    hound::WavSpec {
                        channels: channels as u16,
                        sample_rate,
                        bits_per_sample: 32,
                        sample_format: hound::SampleFormat::Float,
                    },
                )
                .map_err(io::Error::other)
            })
            .transpose()?;
        Ok(Self {
            timing,
            capture,
            summary: WriterSummary::default(),
            max_capture_samples: MAX_WAV_SAMPLES,
        })
    }

    fn write(&mut self, record: BlockRecord) -> io::Result<()> {
        if let Some(capture) = &mut self.capture {
            if record.samples.len() as u64
                > self
                    .max_capture_samples
                    .saturating_sub(self.summary.capture_samples_written)
            {
                self.summary.capture_truncated = true;
            }
            if !self.summary.capture_truncated {
                for sample in record.samples {
                    capture.write_sample(sample).map_err(io::Error::other)?;
                }
                self.summary.capture_samples_written = capture.len() as u64;
            }
        }
        if let (Some(file), Some(block)) = (&mut self.timing, record.measurement) {
            writeln!(
                file,
                "{},{},{},{},{},{},{},{},{},{},{}",
                self.summary.blocks_written,
                block.elapsed,
                block.render_ms,
                block.rms[0],
                block.rms[1],
                block.metrics.trace_ms,
                block.metrics.ir_prepare_ms,
                block.metrics.convolution_ms,
                block.metrics.spatial_updates,
                block.metrics.spatial_update_age_ms,
                u8::from(block.metrics.spatial_deferred)
            )?;
        }
        self.summary.blocks_written += 1;
        Ok(())
    }

    fn finish(&mut self) -> io::Result<()> {
        // Attempt both flushes even if one output has failed.
        let timing = self.timing.as_mut().map_or(Ok(()), Write::flush);
        let capture = self
            .capture
            .take()
            .map_or(Ok(()), |writer| writer.finalize().map_err(io::Error::other));
        timing.and(capture)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/engine/diagnostics.rs"]
mod tests;
