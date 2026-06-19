use crate::core::error::{ErrorCode, SoundError, SoundResult};
use crate::playback::queue::SampleQueue;
use crate::playback::state::PlaybackState;
use cpal::traits::{DeviceTrait, HostTrait};
use std::sync::Arc;
use std::sync::atomic::Ordering;

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
/// The stream callback owns `Arc` handles to `queue` and `state`. Samples
/// are read as `f32`, clamped to the target sample format, and mapped onto the
/// first `output_channels` of the device; any remaining device channels are
/// filled with silence.
pub(super) fn build_output_stream(
    sample_rate: u32,
    output_channels: usize,
    queue: Arc<SampleQueue>,
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
        queue,
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
    queue: Arc<SampleQueue>,
    state: Arc<PlaybackState>,
) -> SoundResult<cpal::Stream> {
    match sample_format {
        cpal::SampleFormat::F32 => {
            build_typed_output_stream::<f32>(device, stream_config, device_channels, output_channels, queue, state)
        }
        cpal::SampleFormat::I16 => {
            build_typed_output_stream::<i16>(device, stream_config, device_channels, output_channels, queue, state)
        }
        cpal::SampleFormat::U16 => {
            build_typed_output_stream::<u16>(device, stream_config, device_channels, output_channels, queue, state)
        }
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
/// The scratch buffer is allocated once and then reused by the callback so the
/// audio thread does not allocate on each device wakeup.
fn build_typed_output_stream<T>(
    device: &cpal::Device,
    stream_config: &cpal::StreamConfig,
    device_channels: usize,
    output_channels: usize,
    queue: Arc<SampleQueue>,
    state: Arc<PlaybackState>,
) -> Result<cpal::Stream, cpal::BuildStreamError>
where
    T: cpal::Sample + cpal::SizedSample + cpal::FromSample<f32>,
{
    let mut scratch = Vec::new();
    let err_fn = |err| eprintln!("Audio stream error: {err}");
    device.build_output_stream(
        stream_config,
        move |data: &mut [T], _| write_output(data, device_channels, output_channels, &queue, &state, &mut scratch),
        err_fn,
        None,
    )
}

/// Drains queued engine samples into one device callback buffer.
///
/// The queue stores only engine channels, so each output frame copies those
/// channels first and writes silence to any extra physical device channels.
fn write_output<T>(
    data: &mut [T],
    device_channels: usize,
    output_channels: usize,
    queue: &SampleQueue,
    state: &PlaybackState,
    scratch: &mut Vec<f32>,
) where
    T: cpal::Sample + cpal::FromSample<f32>,
{
    if let Err(error) = validate_callback_sample_shape(data.len(), device_channels, output_channels) {
        eprintln!("Audio callback shape error: {error}");
        let _ = state.request_stop();
        silence_output(data);
        return;
    }

    let frames = data.len() / device_channels;
    let required_samples = frames * output_channels;

    // The queue is packed as engine-channel samples only; scratch mirrors that
    // smaller shape before samples are expanded into the physical device buffer.
    scratch.resize(required_samples, 0.0);
    let drained = match queue.pop_into(scratch) {
        Ok(drained) => drained,
        Err(error) => {
            eprintln!("Audio queue error: {error}");

            // A poisoned queue cannot be recovered from inside the realtime
            // callback, so request shutdown and keep the device fed with silence.
            let _ = state.request_stop();
            silence_output(data);
            return;
        }
    };

    if let Err(error) = validate_drained_sample_shape(drained, output_channels) {
        eprintln!("Audio queue shape error: {error}");
        let _ = state.request_stop();
        silence_output(data);
        return;
    }

    let rendered_frame_count = drained / output_channels;
    let mut device_frames = data.chunks_exact_mut(device_channels);
    let engine_frames = scratch[..drained].chunks_exact(output_channels);

    for (device_frame, engine_frame) in device_frames.by_ref().take(rendered_frame_count).zip(engine_frames) {
        let (rendered_channels, extra_device_channels) = device_frame.split_at_mut(output_channels);

        // Copy a complete rendered frame into the matching leading device
        // channels. Queue producers push complete interleaved blocks, so partial
        // rendered frames would indicate a broken playback contract.
        for (output_sample, input_sample) in rendered_channels.iter_mut().zip(engine_frame) {
            *output_sample = T::from_sample(input_sample.clamp(-1.0, 1.0));
        }

        // Devices can expose more channels than the engine renders. Those tail
        // channels must be cleared every callback to avoid stale output samples.
        silence_output(extra_device_channels);
    }

    let missing_sample = drained < required_samples;

    // `device_frames` has already advanced past every rendered frame above.
    // On underflow, only the unwritten tail of the device buffer is silenced.
    if missing_sample {
        for device_frame in device_frames {
            silence_output(device_frame);
        }
    }

    // An underrun means "finished" only after the producer has explicitly
    // stopped writing; otherwise it is just a temporary starvation.
    if missing_sample && state.producer_done().load(Ordering::Acquire) {
        state.mark_playback_done();
    }
}

fn validate_callback_sample_shape(data_len: usize, device_channels: usize, output_channels: usize) -> SoundResult<()> {
    crate::debug_validate!(
        device_channels > 0,
        ErrorCode::InvalidArgument,
        "device channel count must be > 0"
    );
    crate::debug_validate!(
        output_channels > 0,
        ErrorCode::InvalidArgument,
        "output channel count must be > 0"
    );
    crate::debug_validate!(
        data_len % device_channels == 0,
        ErrorCode::InvalidArgument,
        "CPAL output buffer length {} must be divisible by device channel count {}",
        data_len,
        device_channels
    );

    Ok(())
}

fn validate_drained_sample_shape(drained: usize, output_channels: usize) -> SoundResult<()> {
    crate::debug_validate!(
        drained % output_channels == 0,
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
mod tests {
    use super::*;

    #[test]
    fn write_output_maps_engine_channels_and_zeros_extra_device_channels() -> SoundResult<()> {
        let queue = SampleQueue::new(4);
        let state = PlaybackState::new();
        state.prepare_queue(4)?;
        assert!(queue.push_block(&[2.0, -2.0, 0.25, -0.25], state.stop_requested())?);
        let mut output = [1.0f32; 6];
        let mut scratch = Vec::new();

        write_output(&mut output, 3, 2, &queue, &state, &mut scratch);

        assert_eq!(output, [1.0, -1.0, 0.0, 0.25, -0.25, 0.0]);
        assert!(!state.is_playback_done());
        Ok(())
    }

    #[test]
    fn write_output_marks_done_after_final_underflow() -> SoundResult<()> {
        let queue = SampleQueue::new(2);
        let state = PlaybackState::new();
        state.prepare_queue(2)?;
        state.mark_producer_done()?;
        assert!(queue.push_block(&[0.5, -0.5], state.stop_requested())?);
        let mut output = [0.0f32; 4];
        let mut scratch = Vec::new();

        write_output(&mut output, 2, 2, &queue, &state, &mut scratch);

        assert_eq!(output, [0.5, -0.5, 0.0, 0.0]);
        assert!(state.is_playback_done());
        Ok(())
    }
}
