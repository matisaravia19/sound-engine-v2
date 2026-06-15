use crate::core::error::{SoundError, SoundResult};
use crate::playback::queue::SampleQueue;
use crate::playback::state::PlaybackState;
use cpal::traits::{DeviceTrait, HostTrait};
use std::sync::Arc;

pub(super) fn default_output_sample_rate() -> SoundResult<u32> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| SoundError::backend_unavailable("No default output device available"))?;
    let supported_config = device
        .default_output_config()
        .map_err(|source| SoundError::external(source.to_string()))?;
    Ok(supported_config.sample_rate().0)
}

pub(super) fn build_output_stream(
    sample_rate: u32,
    output_channels: usize,
    queue: Arc<SampleQueue>,
    state: Arc<PlaybackState>,
) -> SoundResult<cpal::Stream> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| SoundError::backend_unavailable("No default output device available"))?;
    let supported_config = device
        .default_output_config()
        .map_err(|source| SoundError::external(source.to_string()))?;
    let device_channels = supported_config.channels() as usize;
    if device_channels < output_channels {
        return Err(SoundError::unsupported_feature(format!(
            "Default output device exposes {device_channels} channel(s), but playback needs {output_channels}"
        )));
    }

    let mut stream_config: cpal::StreamConfig = supported_config.clone().into();
    stream_config.sample_rate = cpal::SampleRate(sample_rate);
    let err_fn = |err| eprintln!("Audio stream error: {err}");

    match supported_config.sample_format() {
        cpal::SampleFormat::F32 => {
            let mut scratch = Vec::new();
            device.build_output_stream(
                &stream_config,
                move |data: &mut [f32], _| {
                    write_output(data, device_channels, output_channels, &queue, &state, &mut scratch)
                },
                err_fn,
                None,
            )
        }
        cpal::SampleFormat::I16 => {
            let mut scratch = Vec::new();
            device.build_output_stream(
                &stream_config,
                move |data: &mut [i16], _| {
                    write_output(data, device_channels, output_channels, &queue, &state, &mut scratch)
                },
                err_fn,
                None,
            )
        }
        cpal::SampleFormat::U16 => {
            let mut scratch = Vec::new();
            device.build_output_stream(
                &stream_config,
                move |data: &mut [u16], _| {
                    write_output(data, device_channels, output_channels, &queue, &state, &mut scratch)
                },
                err_fn,
                None,
            )
        }
        other => {
            return Err(SoundError::unsupported_feature(format!(
                "Unsupported output sample format: {other:?}"
            )));
        }
    }
    .map_err(|source| SoundError::external(source.to_string()))
}

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
    let frames = data.len() / device_channels;
    let required_samples = frames * output_channels;
    scratch.resize(required_samples, 0.0);
    let drained = match queue.pop_into(scratch) {
        Ok(drained) => drained,
        Err(error) => {
            eprintln!("Audio queue error: {error}");
            let _ = state.request_stop();
            silence_output(data);
            return;
        }
    };
    let missing_sample = drained < required_samples;

    let mut sample_index = 0;
    for frame_start in (0..data.len()).step_by(device_channels) {
        let frame_end = (frame_start + device_channels).min(data.len());
        let output_end = (frame_start + output_channels).min(frame_end);
        for channel_index in frame_start..output_end {
            let sample = if sample_index < drained {
                scratch[sample_index].clamp(-1.0, 1.0)
            } else {
                0.0
            };
            data[channel_index] = T::from_sample(sample);
            sample_index += 1;
        }
        for channel_index in output_end..frame_end {
            data[channel_index] = T::from_sample(0.0);
        }
    }

    if missing_sample && state.producer_done().load(std::sync::atomic::Ordering::Acquire) {
        state.mark_playback_done();
    }
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
        let queue = SampleQueue::new(1);
        let state = PlaybackState::new();
        state.prepare_queue(1)?;
        state.mark_producer_done()?;
        assert!(queue.push_block(&[0.5], state.stop_requested())?);
        let mut output = [0.0f32; 2];
        let mut scratch = Vec::new();

        write_output(&mut output, 2, 2, &queue, &state, &mut scratch);

        assert_eq!(output, [0.5, 0.0]);
        assert!(state.is_playback_done());
        Ok(())
    }
}
