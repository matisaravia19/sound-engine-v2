use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use glam::Vec3;
use sound_engine::acoustics::IrSample;
use sound_engine::auralization::{AuralizationEngine, ImpulseResponseId, PlaySoundRequest};
use sound_engine::core::config::{
    AcousticsConfig, AuralizationConfig, EngineConfig, IrCacheConfig, OutputChannels, SoundConfig,
};
use sound_engine::gpu::backend::VkBackend;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

const DEFAULT_INPUT_A: &str = "C:/Users/matis/OneDrive/Documentos/Fing/Tesis/circus.wav";
const DEFAULT_INPUT_B: &str = "C:/Users/matis/OneDrive/Documentos/Fing/Tesis/music.wav";
const IR_A_ID: ImpulseResponseId = 1;
const IR_B_ID: ImpulseResponseId = 2;
const BLOCK_SIZE: usize = 1024;
const PREFILL_BLOCKS: usize = 8;
const MAX_QUEUED_BLOCKS: usize = 32;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut args = std::env::args().skip(1);
    let input_a = args.next().unwrap_or_else(|| DEFAULT_INPUT_A.to_string());
    let input_b = args.next().unwrap_or_else(|| DEFAULT_INPUT_B.to_string());

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| std::io::Error::other("No default output device available"))?;
    let supported_config = device.default_output_config()?;
    let sample_rate = supported_config.sample_rate().0;

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
        let result = run_engine_producer(input_a, input_b, sample_rate, producer_queue.clone());
        producer_done_flag.store(true, Ordering::Release);
        producer_queue.notify_all();
        result
    });

    queue.wait_until_len_or_done(PREFILL_BLOCKS * BLOCK_SIZE * 2, &producer_done);
    if producer_done.load(Ordering::Acquire) && queue.is_empty() {
        producer
            .join()
            .map_err(|_| std::io::Error::other("Auralization producer thread panicked"))??;
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
        .map_err(|_| std::io::Error::other("Auralization producer thread panicked"))??;

    Ok(())
}

fn run_engine_producer(
    input_a: String,
    input_b: String,
    sample_rate: u32,
    queue: Arc<SampleQueue>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let ir_a = build_ping_pong_ir(sample_rate);
    let ir_b = build_room_ir(sample_rate);
    let ir_num_samples = ir_a.len().max(ir_b.len()) as u32;

    let backend = Arc::new(VkBackend::new()?);
    let mut engine = AuralizationEngine::new(
        backend,
        engine_config(sample_rate, BLOCK_SIZE, ir_num_samples, OutputChannels::Stereo),
    )?;

    engine.register_impulse_response(IR_A_ID, &ir_a)?;
    engine.register_impulse_response(IR_B_ID, &ir_b)?;
    let sound_a = engine.load_wav_sound(input_a)?;
    let sound_b = engine.load_wav_sound(input_b)?;

    engine.play_sound(PlaySoundRequest::new(sound_a, IR_A_ID))?;
    engine.play_sound(PlaySoundRequest {
        sound_id: sound_b,
        impulse_response_id: IR_B_ID,
        gain: 0.8,
    })?;

    let mut output_block = vec![0.0f32; BLOCK_SIZE * 2];
    while engine.has_active_voices() {
        output_block.fill(0.0);
        engine.render_block(&mut output_block)?;
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

fn build_ping_pong_ir(sample_rate: u32) -> Vec<IrSample> {
    let size = (2.0 * sample_rate as f32) as usize;
    let mut ir = vec![IrSample::zero(); size];
    let echoes = [
        (0.000, 1.00),
        (0.125, 0.80),
        (0.250, 0.55),
        (0.375, 0.36),
        (0.500, 0.22),
        (0.625, 0.13),
    ];

    for (i, (time, amplitude)) in echoes.iter().enumerate() {
        let sample = (time * sample_rate as f32) as usize;
        if sample >= size {
            continue;
        }
        if i % 2 == 0 {
            ir[sample] = IrSample::new(*amplitude, 0.0);
        } else {
            ir[sample] = IrSample::new(0.0, *amplitude);
        }
    }

    ir
}

fn build_room_ir(sample_rate: u32) -> Vec<IrSample> {
    let size = (2.0 * sample_rate as f32) as usize;
    let mut ir = vec![IrSample::zero(); size];
    ir[0] = IrSample::new(0.8, 1.0);

    let reflections = [
        (0.018, 0.35, 0.20),
        (0.041, 0.18, 0.28),
        (0.073, 0.14, 0.12),
        (0.117, 0.08, 0.10),
    ];
    for (time, left, right) in reflections {
        let sample = (time * sample_rate as f32) as usize;
        if sample < size {
            ir[sample] = IrSample::new(left, right);
        }
    }

    for (i, sample) in ir.iter_mut().enumerate().skip((0.140 * sample_rate as f32) as usize) {
        let t = i as f32 / sample_rate as f32;
        let decay = (-3.5 * t).exp();
        sample.left += decay * 0.015 * (2.0 * std::f32::consts::PI * 173.0 * t).sin();
        sample.right += decay * 0.015 * (2.0 * std::f32::consts::PI * 211.0 * t).sin();
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
