use crate::core::error::{ErrorCode, SoundError, SoundResult};
use crate::playback::state::PlaybackState;
use cpal::traits::{DeviceTrait, HostTrait};
use std::sync::Arc;

/// Returns the sample rate advertised by the default output device.
///
/// The value comes from CPAL's default host and output device at call time, so
/// callers should treat it as a device capability probe rather than a cached
/// engine setting.
pub(super) fn default_output_sample_rate() -> SoundResult<u32> {
    let config = default_output_device_config()?;
    Ok(config.supported_config.sample_rate().0)
}

/// Builds a CPAL output stream that drains interleaved engine samples.
///
/// The stream callback shares ownership of the state and its sample queue. Samples
/// are read as `f32`, clamped to the target sample format, and mapped onto the
/// first `output_channels` of the device; any remaining device channels are
/// filled with silence.
pub(super) fn build_output_stream(
    sample_rate: u32,
    output_channels: usize,
    state: Arc<PlaybackState>,
) -> SoundResult<cpal::Stream> {
    let config = default_output_device_config()?;
    validate_output_channels(config.device_channels, output_channels)?;

    let mut stream_config: cpal::StreamConfig = config.supported_config.clone().into();
    stream_config.sample_rate = cpal::SampleRate(sample_rate);

    build_stream_for_sample_format(
        &config.device,
        &stream_config,
        config.supported_config.sample_format(),
        config.device_channels,
        output_channels,
        state,
    )
}

struct OutputDeviceConfig {
    device: cpal::Device,
    supported_config: cpal::SupportedStreamConfig,
    device_channels: usize,
}

/// Opens the default CPAL output device and captures its default configuration.
fn default_output_device_config() -> SoundResult<OutputDeviceConfig> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| SoundError::backend_unavailable("No default output device available"))?;
    let supported_config = device
        .default_output_config()
        .map_err(|source| SoundError::external(source.to_string()))?;
    let device_channels = supported_config.channels() as usize;

    Ok(OutputDeviceConfig {
        device,
        supported_config,
        device_channels,
    })
}

/// Ensures the selected device can carry every interleaved engine channel.
fn validate_output_channels(device_channels: usize, output_channels: usize) -> SoundResult<()> {
    if device_channels < output_channels {
        return Err(SoundError::unsupported_feature(format!(
            "Default output device exposes {device_channels} channel(s), but playback needs {output_channels}"
        )));
    }

    Ok(())
}

/// Selects the concrete callback sample type required by the output device.
fn build_stream_for_sample_format(
    device: &cpal::Device,
    stream_config: &cpal::StreamConfig,
    sample_format: cpal::SampleFormat,
    device_channels: usize,
    output_channels: usize,
    state: Arc<PlaybackState>,
) -> SoundResult<cpal::Stream> {
    match sample_format {
        cpal::SampleFormat::F32 => build_typed_output_stream::<f32>(
            device,
            stream_config,
            device_channels,
            output_channels,
            state,
        ),
        cpal::SampleFormat::I16 => build_typed_output_stream::<i16>(
            device,
            stream_config,
            device_channels,
            output_channels,
            state,
        ),
        cpal::SampleFormat::U16 => build_typed_output_stream::<u16>(
            device,
            stream_config,
            device_channels,
            output_channels,
            state,
        ),
        other => {
            return Err(SoundError::unsupported_feature(format!(
                "Unsupported output sample format: {other:?}"
            )));
        }
    }
    .map_err(|source| SoundError::external(source.to_string()))
}

/// Installs the realtime callback for one CPAL sample type.
///
/// Samples are converted directly into the device buffer without scratch
/// allocation. A busy producer queue yields silence until the next callback.
fn build_typed_output_stream<T>(
    device: &cpal::Device,
    stream_config: &cpal::StreamConfig,
    device_channels: usize,
    output_channels: usize,
    state: Arc<PlaybackState>,
) -> Result<cpal::Stream, cpal::BuildStreamError>
where
    T: cpal::Sample + cpal::SizedSample + cpal::FromSample<f32>,
{
    let error_state = state.clone();
    let err_fn = move |err| {
        // A failed device may never issue another callback to drain the queue.
        // Retain its cause before waking producers so failure cannot look like
        // a successful user-requested stop.
        error_state.report_failure(SoundError::with_source(
            ErrorCode::External,
            format!("Audio stream failed: {err}"),
            err,
        ));
    };
    device.build_output_stream(
        stream_config,
        move |data: &mut [T], _| write_output(data, device_channels, output_channels, &state),
        err_fn,
        None,
    )
}

/// Drains queued engine samples into one device callback buffer.
///
/// The queue stores only engine channels, so each output frame copies those
/// channels first and writes silence to any extra physical device channels.
pub(super) fn write_output<T>(
    data: &mut [T],
    device_channels: usize,
    output_channels: usize,
    state: &PlaybackState,
) where
    T: cpal::Sample + cpal::FromSample<f32>,
{
    if let Err(error) = validate_callback_sample_shape(data.len(), device_channels, output_channels)
    {
        state.report_failure(error);
        silence_output(data);
        return;
    }

    silence_output(data);
    let frames = data.len() / device_channels;
    if !state.ready_to_play() || state.is_stop_requested() {
        return;
    }
    let required_samples = frames * output_channels;
    // If finish races the drain, the producer may still enqueue its last block.
    // Observe completion first so an empty tail proves no later enqueue remains.
    let producer_done = state.is_producer_done();
    let drained = match state.queue().drain_into(required_samples, |index, input| {
        let output_index = (index / output_channels) * device_channels + index % output_channels;
        // DSP overflow must never reach the device or integer sample conversion.
        let finite = if input.is_finite() {
            input.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        data[output_index] = T::from_sample(finite);
    }) {
        Ok(Some(drained)) => drained,
        Ok(None) => {
            // Contention leaves queued audio intact, even after producer shutdown.
            // Only an observed empty tail can establish playback completion.
            state.record_callback(0, frames);
            return;
        }
        Err(error) => {
            state.report_failure(error);
            return;
        }
    };
    if let Err(error) = validate_drained_sample_shape(drained, output_channels) {
        state.report_failure(error);
        silence_output(data);
        return;
    }

    state.record_callback(drained / output_channels, frames);
    let missing_sample = drained < required_samples;

    // An underrun means "finished" only after the producer has explicitly
    // stopped writing; otherwise it is just a temporary starvation.
    if missing_sample && producer_done {
        state.mark_playback_done();
    }
}
fn validate_callback_sample_shape(
    data_len: usize,
    device_channels: usize,
    output_channels: usize,
) -> SoundResult<()> {
    crate::validate!(
        device_channels > 0,
        ErrorCode::InvalidArgument,
        "device channel count must be > 0"
    );
    crate::validate!(
        output_channels > 0,
        ErrorCode::InvalidArgument,
        "output channel count must be > 0"
    );
    if output_channels > device_channels {
        return Err(SoundError::invalid_argument(
            "Output channel count exceeds the device callback layout",
        ));
    }
    crate::validate!(
        data_len.is_multiple_of(device_channels),
        ErrorCode::InvalidArgument,
        "CPAL output buffer length {} must be divisible by device channel count {}",
        data_len,
        device_channels
    );

    Ok(())
}
fn validate_drained_sample_shape(drained: usize, output_channels: usize) -> SoundResult<()> {
    crate::validate!(
        drained.is_multiple_of(output_channels),
        ErrorCode::InvalidState,
        "playback queue drained {} sample(s), which is not divisible by output channel count {}",
        drained,
        output_channels
    );

    Ok(())
}

fn silence_output<T>(data: &mut [T])
where
    T: cpal::Sample + cpal::FromSample<f32>,
{
    for sample in data {
        *sample = T::from_sample(0.0);
    }
}

#[cfg(test)]
#[path = "../../tests/unit/playback/device.rs"]
mod tests;
