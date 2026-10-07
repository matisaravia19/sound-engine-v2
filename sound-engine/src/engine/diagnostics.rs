//! Optional runtime diagnostics with bounded retention and off-thread file I/O.

mod output;

use super::direct::RenderMetrics;
use crate::playback::SoundPlayer;
use output::{OutputPaths, OutputWriter, WriterSummary};
use std::io;
use std::path::PathBuf;
use std::sync::mpsc::{self, SyncSender};
use std::thread::{self, JoinHandle};
use std::time::Instant;

// Include space for the record being produced and the one being written.
const QUEUE_BYTES: usize = 8 * 1024 * 1024;
const MAX_QUEUED_BLOCKS: usize = 1024;

/// Owns a diagnostic writer while borrowing the player's final underrun counters.
/// Dropping closes its bounded queue and joins the writer before saving metadata.
pub(super) struct RuntimeDiagnostics<'a> {
    /// Callback counters remain available until the diagnostic writer has joined.
    player: &'a SoundPlayer,
    /// Bounded recording session, absent when disabled or initialization failed.
    recorder: Option<Recorder>,
    /// Monotonic origin for block timing rows.
    started: Instant,
    /// Validated channel count used to measure each interleaved block.
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
        let playback = PlaybackSummary {
            underruns,
            missing,
            starvation_times: self.player.starvation_time_range_seconds(),
        };
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
        let result = summary.write_metadata(&summary_path, playback, omitted, error.is_some());
        if let Err(error) = result {
            eprintln!("Could not save runtime diagnostic metadata: {error}");
        }
    }
}

/// Callback counters sampled before joining, while the device stream is still alive.
struct PlaybackSummary {
    underruns: u64,
    missing: u64,
    starvation_times: Option<(f64, f64)>,
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

/// Owns the bounded producer queue and joins its file writer when recording finishes.
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
        // Opening files happens before playback produces blocks, not in its callback.
        let mut outputs = OutputWriter::new(&paths, sample_rate, channels)?;
        let (sender, receiver) = mpsc::sync_channel(capacity);
        let worker = thread::Builder::new()
            .name("sound-engine-diagnostics".into())
            .spawn(move || {
                let result = receiver
                    .into_iter()
                    .try_for_each(|record| outputs.write(record));
                // Finalize both outputs even when a preceding block write failed.
                let finalized = outputs.finish();
                (outputs.summary, result.and(finalized))
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

#[cfg(test)]
#[path = "../../tests/unit/engine/diagnostics.rs"]
mod tests;

#[cfg(test)]
#[path = "../../tests/support/diagnostics.rs"]
mod test_support;
