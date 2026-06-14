use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use glam::Vec3;
use sound_engine::acoustics::IrSample;
use sound_engine::auralization::{AuralizationEngine, ImpulseResponseId, PlaySoundRequest};
use sound_engine::core::config::{
    AcousticsConfig, AuralizationConfig, EngineConfig, IrCacheConfig, OutputChannels, SoundConfig,
};
use sound_engine::gpu::backend::VkBackend;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader};
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
    let args = parse_args()?;

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| std::io::Error::other("No default output device available"))?;
    let supported_config = device.default_output_config()?;
    let sample_rate = ir_sample_rate(&args)?.unwrap_or_else(|| supported_config.sample_rate().0);

    let channels = supported_config.channels() as usize;
    if channels < 2 {
        return Err(std::io::Error::other(format!(
            "Default output device must expose at least 2 channels for stereo playback, got {channels}"
        ))
        .into());
    }

    let mut stream_config: cpal::StreamConfig = supported_config.clone().into();
    stream_config.sample_rate = cpal::SampleRate(sample_rate);
    let queue = Arc::new(SampleQueue::new(MAX_QUEUED_BLOCKS * BLOCK_SIZE * 2));
    let producer_done = Arc::new(AtomicBool::new(false));
    let playback_done = Arc::new(AtomicBool::new(false));

    let producer_queue = queue.clone();
    let producer_done_flag = producer_done.clone();
    let producer = thread::spawn(move || -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let result = run_engine_producer(
            args.input_a,
            args.input_b,
            args.ir_a_csv,
            args.ir_b_csv,
            args.play_second_sound,
            sample_rate,
            producer_queue.clone(),
        );
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
    ir_a_csv: Option<String>,
    ir_b_csv: Option<String>,
    play_second_sound: bool,
    sample_rate: u32,
    queue: Arc<SampleQueue>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let ir_a = match ir_a_csv {
        Some(path) => load_ir_csv(&path, sample_rate)?,
        None => build_ping_pong_ir(sample_rate),
    };
    let ir_b = if play_second_sound {
        Some(match ir_b_csv {
            Some(path) => load_ir_csv(&path, sample_rate)?,
            None => build_room_ir(sample_rate),
        })
    } else {
        None
    };
    let ir_num_samples = ir_b.as_ref().map_or(ir_a.len(), |ir_b| ir_a.len().max(ir_b.len())) as u32;

    let backend = Arc::new(VkBackend::new()?);
    let mut engine = AuralizationEngine::new(
        backend,
        engine_config(sample_rate, BLOCK_SIZE, ir_num_samples, OutputChannels::Stereo),
    )?;

    engine.register_impulse_response(IR_A_ID, &ir_a)?;
    let sound_a = engine.load_wav_sound(input_a)?;
    engine.play_sound(PlaySoundRequest::new(sound_a, IR_A_ID))?;

    if let Some(ir_b) = ir_b {
        engine.register_impulse_response(IR_B_ID, &ir_b)?;
        let sound_b = engine.load_wav_sound(input_b)?;
        engine.play_sound(PlaySoundRequest {
            sound_id: sound_b,
            impulse_response_id: IR_B_ID,
            gain: 0.8,
        })?;
    }

    let mut output_block = vec![0.0f32; BLOCK_SIZE * 2];
    while engine.has_active_voices() {
        output_block.fill(0.0);
        engine.render_block(&mut output_block)?;
        queue.push_block(&output_block);
    }

    Ok(())
}

struct ExampleArgs {
    input_a: String,
    input_b: String,
    ir_a_csv: Option<String>,
    ir_b_csv: Option<String>,
    play_second_sound: bool,
}

fn parse_args() -> Result<ExampleArgs, Box<dyn std::error::Error + Send + Sync>> {
    let mut positionals = Vec::new();
    let mut shared_ir_csv = None;
    let mut ir_a_csv = None;
    let mut ir_b_csv = None;
    let mut play_second_sound = true;
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            "--ir-csv" => {
                shared_ir_csv = Some(next_arg_value(&mut args, "--ir-csv")?);
            }
            "--ir-a-csv" => {
                ir_a_csv = Some(next_arg_value(&mut args, "--ir-a-csv")?);
            }
            "--ir-b-csv" => {
                ir_b_csv = Some(next_arg_value(&mut args, "--ir-b-csv")?);
            }
            "--one-sound" => {
                play_second_sound = false;
            }
            _ if arg.starts_with("--") => {
                return Err(
                    std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("Unknown argument: {arg}")).into(),
                );
            }
            _ => positionals.push(arg),
        }
    }

    if positionals.len() > 2 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Expected at most two positional WAV paths",
        )
        .into());
    }
    if !play_second_sound && positionals.len() > 1 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "--one-sound accepts only the first positional WAV path",
        )
        .into());
    }
    if !play_second_sound && ir_b_csv.is_some() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "--ir-b-csv cannot be used with --one-sound",
        )
        .into());
    }

    if let Some(path) = shared_ir_csv {
        ir_a_csv.get_or_insert_with(|| path.clone());
        if play_second_sound {
            ir_b_csv.get_or_insert(path);
        }
    }

    Ok(ExampleArgs {
        input_a: positionals
            .first()
            .cloned()
            .unwrap_or_else(|| DEFAULT_INPUT_A.to_string()),
        input_b: positionals
            .get(1)
            .cloned()
            .unwrap_or_else(|| DEFAULT_INPUT_B.to_string()),
        ir_a_csv,
        ir_b_csv,
        play_second_sound,
    })
}

fn next_arg_value(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    args.next()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("{flag} requires a path")).into())
}

fn print_usage() {
    eprintln!(
        "Usage: auralization_engine_stream_wav [input_a.wav] [input_b.wav] [--one-sound] [--ir-csv ir.csv] [--ir-a-csv ir.csv] [--ir-b-csv ir.csv]"
    );
}

fn ir_sample_rate(args: &ExampleArgs) -> Result<Option<u32>, Box<dyn std::error::Error + Send + Sync>> {
    let mut sample_rate = None;
    for path in args.ir_csv_paths() {
        let path_sample_rate = read_ir_csv_sample_rate(path)?;
        match sample_rate {
            Some(existing_sample_rate) if existing_sample_rate != path_sample_rate => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("IR CSV sample rates must match, got {existing_sample_rate} Hz and {path_sample_rate} Hz"),
                )
                .into());
            }
            Some(_) => {}
            None => sample_rate = Some(path_sample_rate),
        }
    }

    Ok(sample_rate)
}

impl ExampleArgs {
    fn ir_csv_paths(&self) -> impl Iterator<Item = &str> {
        self.ir_a_csv
            .iter()
            .chain(self.ir_b_csv.iter().filter(|_| self.play_second_sound))
            .map(String::as_str)
    }
}

fn read_ir_csv_sample_rate(path: &str) -> Result<u32, Box<dyn std::error::Error + Send + Sync>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        if let Some(value) = line.strip_prefix("# sample_rate=") {
            let sample_rate = value.trim().parse::<u32>()?;
            if sample_rate == 0 {
                return Err(
                    std::io::Error::new(std::io::ErrorKind::InvalidData, "IR CSV sample_rate must be > 0").into(),
                );
            }
            return Ok(sample_rate);
        }
        if !line.starts_with('#') && !line.trim().is_empty() {
            break;
        }
    }

    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("IR CSV {path} is missing required # sample_rate metadata"),
    )
    .into())
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

fn load_ir_csv(
    path: &str,
    expected_sample_rate: u32,
) -> Result<Vec<IrSample>, Box<dyn std::error::Error + Send + Sync>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut sample_rate = None;
    let mut header = None;
    let mut rows = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.starts_with('#') {
            if let Some(value) = line.strip_prefix("# sample_rate=") {
                sample_rate = Some(value.trim().parse::<u32>()?);
            }
            continue;
        }
        if line.trim().is_empty() {
            continue;
        }
        if header.is_none() {
            header = Some(line);
        } else {
            rows.push(line);
        }
    }

    let source_sample_rate = sample_rate.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("IR CSV {path} is missing required # sample_rate metadata"),
        )
    })?;
    if source_sample_rate != expected_sample_rate {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("IR CSV sample_rate={source_sample_rate} does not match engine sample_rate={expected_sample_rate}"),
        )
        .into());
    }

    let header =
        header.ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "IR CSV is missing a header row"))?;
    let columns: Vec<&str> = header.split(',').map(str::trim).collect();
    let left_column = find_csv_column(&columns, "left")?;
    let right_column = find_csv_column(&columns, "right")?;
    let required_columns = ["sample_index", "time_seconds", "left", "right"];
    for required_column in required_columns {
        find_csv_column(&columns, required_column)?;
    }

    let mut samples = Vec::with_capacity(rows.len());
    for (row_index, row) in rows.iter().enumerate() {
        let fields: Vec<&str> = row.split(',').map(str::trim).collect();
        if fields.len() != columns.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "IR CSV row {} has {} field(s), expected {}",
                    row_index + 2,
                    fields.len(),
                    columns.len()
                ),
            )
            .into());
        }

        samples.push(IrSample::new(
            fields[left_column].parse::<f32>()?,
            fields[right_column].parse::<f32>()?,
        ));
    }

    if samples.is_empty() {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "IR CSV contains no samples").into());
    }

    Ok(samples)
}

fn find_csv_column(columns: &[&str], name: &str) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    columns.iter().position(|column| *column == name).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("IR CSV is missing required column: {name}"),
        )
        .into()
    })
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
