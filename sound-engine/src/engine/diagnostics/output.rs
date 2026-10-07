//! Diagnostic destinations and file formats, independent of playback and queue scheduling.

use super::{BlockRecord, PlaybackSummary};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

// RIFF uses 32-bit chunk sizes; reserve ample room for hound's format headers.
const MAX_WAV_SAMPLES: u64 = (u32::MAX as u64 - 4096) / size_of::<f32>() as u64;

/// Requested timing and audio destinations, checked before any file is truncated.
#[derive(Clone)]
pub(super) struct OutputPaths {
    /// Optional CSV destination for per-block measurements.
    pub(super) timing: Option<PathBuf>,
    /// Optional WAV destination for interleaved rendered samples.
    pub(super) capture: Option<PathBuf>,
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

    /// Derives the metadata destination from the first enabled output.
    pub(super) fn summary_path(&self) -> PathBuf {
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

/// Completed output counts, including whether the WAV reached its RIFF limit.
#[derive(Default)]
pub(super) struct WriterSummary {
    /// Blocks fully processed by every enabled output.
    pub(super) blocks_written: u64,
    /// Interleaved samples retained in the audio capture.
    pub(super) capture_samples_written: u64,
    /// Whether the WAV sample limit stopped further audio capture.
    pub(super) capture_truncated: bool,
}

impl WriterSummary {
    /// Saves callback counters and recording completeness after the writer has joined.
    pub(super) fn write_metadata(
        &self,
        summary_path: &Path,
        playback: PlaybackSummary,
        omitted: u64,
        write_failed: bool,
    ) -> io::Result<()> {
        let PlaybackSummary {
            underruns,
            missing,
            starvation_times,
        } = playback;
        std::fs::write(
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
                self.blocks_written,
                self.capture_samples_written,
                self.capture_truncated,
                write_failed,
            ),
        )
    }
}

/// Owns the CSV and WAV files used exclusively by the diagnostic writer thread.
/// Explicit finalization flushes both outputs and reports errors before metadata is saved.
pub(super) struct OutputWriter {
    timing: Option<BufWriter<File>>,
    capture: Option<hound::WavWriter<BufWriter<File>>>,
    /// Counts advanced only after successful writes.
    pub(super) summary: WriterSummary,
    max_capture_samples: u64,
}

impl OutputWriter {
    /// Opens distinct destinations after the caller has validated the audio shape.
    pub(super) fn new(paths: &OutputPaths, sample_rate: u32, channels: usize) -> io::Result<Self> {
        paths.validate()?;
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

    /// Writes one complete block, keeping timing rows after the WAV limit is reached.
    pub(super) fn write(&mut self, record: BlockRecord) -> io::Result<()> {
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

    /// Flushes CSV data and finalizes the WAV header, attempting both on failure.
    pub(super) fn finish(&mut self) -> io::Result<()> {
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
#[path = "../../../tests/unit/engine/diagnostics/output.rs"]
mod tests;
