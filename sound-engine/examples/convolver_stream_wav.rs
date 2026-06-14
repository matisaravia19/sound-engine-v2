use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use glam::Vec3;
use hound::{SampleFormat, WavReader};
use sound_engine::acoustics::IrSample;
use sound_engine::auralization::ImpulseResponseId;
use sound_engine::auralization::convolver::PartitionedConvolver;
use sound_engine::core::config::{
    AcousticsConfig, AuralizationConfig, EngineConfig, IrCacheConfig, OutputChannels, SoundConfig,
};
use sound_engine::gpu::backend::VkBackend;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

const DEFAULT_INPUT: &str = "C:/Users/matis/OneDrive/Documentos/Fing/Tesis/circus.wav";
const IR_ID: ImpulseResponseId = 1;
const BLOCK_SIZE: usize = 1024;
const PREFILL_BLOCKS: usize = 8;
const MAX_QUEUED_BLOCKS: usize = 32;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let input_path = std::env::args().nth(1).unwrap_or_else(|| DEFAULT_INPUT.to_string());

    let (input_samples, sample_rate) = read_mono_wav_as_f32(&input_path)?;
    if input_samples.is_empty() {
        return Err(std::io::Error::other("Input WAV has no samples").into());
    }

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| std::io::Error::other("No default output device available"))?;
    let supported_config = device.default_output_config()?;

    if supported_config.sample_rate().0 != sample_rate {
        return Err(std::io::Error::other(format!(
            "Input WAV sample rate ({sample_rate} Hz) must match output device sample rate ({} Hz)",
            supported_config.sample_rate().0
        ))
        .into());
    }

    let channels = supported_config.channels() as usize;
    if channels < 2 {
        return Err(std::io::Error::other(format!(
            "Default output device must expose at least 2 channels for stereo playback, got {channels}"
        ))
        .into());
    }
    let stream_config: cpal::StreamConfig = supported_config.clone().into();
    let queue = Arc::new(SampleQueue::new(MAX_QUEUED_BLOCKS * BLOCK_SIZE * 2));
    let producer_done = Arc::new(AtomicBool::new(false));
    let playback_done = Arc::new(AtomicBool::new(false));

    let producer_queue = queue.clone();
    let producer_done_flag = producer_done.clone();
    let producer = thread::spawn(move || -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let result = run_convolver_producer(input_samples, sample_rate, producer_queue.clone());
        producer_done_flag.store(true, Ordering::Release);
        producer_queue.notify_all();
        result
    });

    queue.wait_until_len_or_done(PREFILL_BLOCKS * BLOCK_SIZE * 2, &producer_done);
    if producer_done.load(Ordering::Acquire) && queue.is_empty() {
        producer
            .join()
            .map_err(|_| std::io::Error::other("Convolver producer thread panicked"))??;
        return Ok(());
    }

    let callback_queue = queue.clone();
    let callback_producer_done = producer_done.clone();
    let callback_playback_done = playback_done.clone();
    let err_fn = |err| eprintln!("Audio stream error: {err}");

    let stream = match supported_config.sample_format() {
        cpal::SampleFormat::F32 => device.build_output_stream(
            &stream_config,
            move |data: &mut [f32], _| {
                write_output(
                    data,
                    channels,
                    &callback_queue,
                    &callback_producer_done,
                    &callback_playback_done,
                )
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_output_stream(
            &stream_config,
            move |data: &mut [i16], _| {
                write_output(
                    data,
                    channels,
                    &callback_queue,
                    &callback_producer_done,
                    &callback_playback_done,
                )
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::U16 => device.build_output_stream(
            &stream_config,
            move |data: &mut [u16], _| {
                write_output(
                    data,
                    channels,
                    &callback_queue,
                    &callback_producer_done,
                    &callback_playback_done,
                )
            },
            err_fn,
            None,
        )?,
        other => {
            return Err(std::io::Error::other(format!("Unsupported output sample format: {other:?}")).into());
        }
    };

    stream.play()?;

    while !playback_done.load(Ordering::Acquire) {
        thread::sleep(Duration::from_millis(20));
    }

    drop(stream);

    producer
        .join()
        .map_err(|_| std::io::Error::other("Convolver producer thread panicked"))??;

    Ok(())
}

fn run_convolver_producer(
    input_samples: Vec<f32>,
    sample_rate: u32,
    queue: Arc<SampleQueue>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let ir = build_sample_ir(sample_rate);

    let backend = Arc::new(VkBackend::new()?);
    let mut convolver = PartitionedConvolver::new(
        backend,
        engine_config(sample_rate, BLOCK_SIZE, ir.len() as u32, OutputChannels::Stereo),
    )?;
    convolver.register_impulse_response(IR_ID, &ir)?;
    let voice_id = convolver.start_sound(IR_ID)?;

    let block_size = BLOCK_SIZE;
    let mut input_block = vec![0.0f32; block_size];
    let mut output_block = vec![0.0f32; block_size * 2];

    for chunk in input_samples.chunks(block_size) {
        input_block.fill(0.0);
        input_block[..chunk.len()].copy_from_slice(chunk);

        output_block.fill(0.0);
        convolver.process_block(voice_id, &input_block, &mut output_block)?;
        queue.push_block(&output_block[..chunk.len() * 2]);
        if chunk.len() < block_size {
            queue.push_block(&output_block[chunk.len() * 2..]);
        }
    }

    while convolver.flush_tail_block(voice_id, &mut output_block)? {
        queue.push_block(&output_block);
    }

    Ok(())
}

fn write_output<T>(
    data: &mut [T],
    channels: usize,
    queue: &SampleQueue,
    producer_done: &AtomicBool,
    playback_done: &AtomicBool,
) where
    T: cpal::Sample + cpal::FromSample<f32>,
{
    let mut missing_sample = false;

    for frame in data.chunks_mut(channels) {
        let left = match queue.pop_sample() {
            Some(sample) => sample.clamp(-1.0, 1.0),
            None => {
                missing_sample = true;
                0.0
            }
        };
        let right = match queue.pop_sample() {
            Some(sample) => sample.clamp(-1.0, 1.0),
            None => {
                missing_sample = true;
                0.0
            }
        };

        frame[0] = T::from_sample(left);
        frame[1] = T::from_sample(right);
        for channel in &mut frame[2..] {
            *channel = T::from_sample(0.0);
        }
    }

    if missing_sample && producer_done.load(Ordering::Acquire) && queue.is_empty() {
        playback_done.store(true, Ordering::Release);
    }
}

struct SampleQueue {
    samples: Mutex<VecDeque<f32>>,
    available: Condvar,
    capacity: usize,
}

impl SampleQueue {
    fn new(capacity: usize) -> Self {
        Self {
            samples: Mutex::new(VecDeque::with_capacity(capacity)),
            available: Condvar::new(),
            capacity,
        }
    }

    fn push_block(&self, block: &[f32]) {
        loop {
            let mut samples = self.samples.lock().expect("sample queue lock poisoned");
            let free = self.capacity.saturating_sub(samples.len());
            if free >= block.len() {
                samples.extend(block);
                self.available.notify_all();
                return;
            }

            drop(samples);
            thread::sleep(Duration::from_millis(2));
        }
    }

    fn pop_sample(&self) -> Option<f32> {
        let mut samples = self.samples.lock().expect("sample queue lock poisoned");
        let sample = samples.pop_front();
        if sample.is_some() {
            self.available.notify_all();
        }
        sample
    }

    fn wait_until_len_or_done(&self, len: usize, done: &AtomicBool) {
        let mut samples = self.samples.lock().expect("sample queue lock poisoned");
        while samples.len() < len && !done.load(Ordering::Acquire) {
            let (next_samples, _) = self
                .available
                .wait_timeout(samples, Duration::from_millis(10))
                .expect("sample queue lock poisoned");
            samples = next_samples;
        }
    }

    fn notify_all(&self) {
        self.available.notify_all();
    }

    fn is_empty(&self) -> bool {
        self.samples.lock().expect("sample queue lock poisoned").is_empty()
    }
}

fn read_mono_wav_as_f32(path: &str) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error + Send + Sync>> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels != 1 {
        return Err(std::io::Error::other("Input must be mono WAV (1 channel)").into());
    }

    let samples = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Float, 32) => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        (SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|s| s.map(|x| x as f32 / i16::MAX as f32))
            .collect::<Result<Vec<_>, _>>()?,
        (SampleFormat::Int, 24) => reader
            .samples::<i32>()
            .map(|s| s.map(|x| x as f32 / 8_388_607.0))
            .collect::<Result<Vec<_>, _>>()?,
        (SampleFormat::Int, 32) => reader
            .samples::<i32>()
            .map(|s| s.map(|x| x as f32 / i32::MAX as f32))
            .collect::<Result<Vec<_>, _>>()?,
        _ => {
            return Err(std::io::Error::other(format!(
                "Unsupported WAV format: {:?} {} bits",
                spec.sample_format, spec.bits_per_sample
            ))
            .into());
        }
    };

    Ok((samples, spec.sample_rate))
}

// fn build_sample_ir(sample_rate: u32) -> Vec<IrSample> {
//     let size = sample_rate as usize;
//     let mut ir = vec![IrSample::zero(); size];
//     ir[0] = IrSample::new(1.0, 0.0);
//     ir[size - 1] = IrSample::new(0.0, 1.0);
//     // let d1 = (sample_rate as usize / 4).min(size - 1);
//     // let d2 = (sample_rate as usize / 2).min(size - 1);
//     // ir[d1] = 0.45;
//     // ir[d2] = 0.25;

//     // Add a light exponentially decaying tail.
//     // for (i, s) in ir.iter_mut().enumerate().skip(1) {
//     //     let t = i as f32 / sample_rate as f32;
//     //     *s += (-8.0 * t).exp() * 0.02 * (2.0 * std::f32::consts::PI * 180.0 * t).sin();
//     // }

//     ir
// }

// fn build_sample_ir(sample_rate: u32) -> Vec<IrSample> {
//     let size = sample_rate as usize; // 1 second IR

//     let mut ir = vec![IrSample::zero(); size];

//     // Direct sound
//     ir[0] = IrSample { left: 1.0, right: 0.8 };

//     // Early reflections
//     ir[(0.015 * sample_rate as f32) as usize] = IrSample { left: 0.5, right: 0.3 };

//     ir[(0.032 * sample_rate as f32) as usize] = IrSample { left: 0.25, right: 0.4 };

//     ir[(0.055 * sample_rate as f32) as usize] = IrSample {
//         left: 0.18,
//         right: 0.12,
//     };

//     // Late reverberation tail
//     for i in (0.08 * sample_rate as f32) as usize..size {
//         let t = i as f32 / sample_rate as f32;

//         let decay = (-4.0 * t).exp();

//         let left_mod = (t * 120.0).sin() * 0.02 + (t * 340.0).sin() * 0.01;

//         let right_mod = (t * 100.0).sin() * 0.02 + (t * 290.0).sin() * 0.01;

//         ir[i].left += decay * left_mod;
//         ir[i].right += decay * right_mod;
//     }

//     ir
// }

fn build_sample_ir(sample_rate: u32) -> Vec<IrSample> {
    let size = (2.0 * sample_rate as f32) as usize;
    let mut ir = vec![IrSample::zero(); size];

    let echoes = [
        (0.000, 1.00),
        (0.125, 0.90),
        (0.250, 0.70),
        (0.375, 0.50),
        (0.500, 0.30),
        (0.625, 0.18),
        (0.750, 0.12),
        (0.875, 0.08),
    ];

    for (i, (time, amplitude)) in echoes.iter().enumerate() {
        let sample = (time * sample_rate as f32) as usize;

        if sample >= size {
            continue;
        }

        // Ping-pong between ears.
        if i % 2 == 0 {
            ir[sample] = IrSample {
                left: *amplitude,
                right: 0.0,
            };
        } else {
            ir[sample] = IrSample {
                left: 0.0,
                right: *amplitude,
            };
        }
    }

    ir
}

fn engine_config(
    sample_rate: u32,
    block_size: usize,
    ir_num_samples: u32,
    output_channels: OutputChannels,
) -> EngineConfig {
    EngineConfig {
        sound: SoundConfig {
            output_channels,
            sample_rate,
            ir_num_samples,
        },
        acoustics: AcousticsConfig {
            listener_half_extent: Vec3::splat(0.2),
            rays_per_query: 1,
            max_bounces: 0,
        },
        cache: IrCacheConfig::default(),
        auralization: AuralizationConfig { block_size },
    }
}
