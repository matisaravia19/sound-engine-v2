use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use sound_engine::auralization::convolver::PartitionedConvolver;
use sound_engine::auralization::{BLOCK_SIZE, ImpulseResponseId};
use sound_engine::gpu::backend::VkBackend;
use std::sync::Arc;

const DEFAULT_INPUT: &str = "C:/Users/matis/OneDrive/Documentos/Fing/Tesis/explosion.wav";
const DEFAULT_OUTPUT: &str = "C:/Users/matis/OneDrive/Documentos/Fing/Tesis/explosion-output.wav";
const IR_ID: ImpulseResponseId = 1;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut args = std::env::args().skip(1);
    let input_path = args.next().unwrap_or_else(|| DEFAULT_INPUT.to_string());
    let output_path = args.next().unwrap_or_else(|| DEFAULT_OUTPUT.to_string());

    let (input_samples, sample_rate) = read_mono_wav_as_f32(&input_path)?;
    if input_samples.is_empty() {
        return Err(std::io::Error::other("Input WAV has no samples").into());
    }

    let ir = build_sample_ir(sample_rate);

    let backend = Arc::new(VkBackend::new().map_err(|e| std::io::Error::other(e.to_string()))?);
    let mut convolver = PartitionedConvolver::new(backend)?;
    convolver.register_impulse_response(IR_ID, &ir)?;
    let voice_id = convolver.start_sound(IR_ID)?;

    let block_size = BLOCK_SIZE as usize;
    let mut processed = Vec::with_capacity(input_samples.len() + block_size);
    let mut input_block = vec![0.0f32; block_size];
    let mut output_block = vec![0.0f32; block_size];

    for chunk in input_samples.chunks(block_size) {
        input_block.fill(0.0);
        input_block[..chunk.len()].copy_from_slice(chunk);

        convolver.process_block(voice_id, &input_block, &mut output_block)?;
        processed.extend_from_slice(&output_block[..chunk.len()]);
    }

    // Flush overlap tail so the reverb/echo decay is not truncated.
    let flush_blocks = sound_engine::auralization::TAIL_SIZE.div_ceil(block_size);
    input_block.fill(0.0);
    for _ in 0..flush_blocks {
        output_block.fill(0.0);
        convolver.process_block(voice_id, &input_block, &mut output_block)?;
        processed.extend_from_slice(&output_block);
    }

    normalize_if_needed(&mut processed);
    write_mono_f32_wav(&output_path, sample_rate, &processed)?;
    println!("Convolved {} samples -> {}", processed.len(), output_path);
    Ok(())
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

fn write_mono_f32_wav(
    path: &str,
    sample_rate: u32,
    samples: &[f32],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut writer = WavWriter::create(path, spec)?;
    for &s in samples {
        writer.write_sample(s)?;
    }
    writer.finalize()?;
    Ok(())
}

fn build_sample_ir(sample_rate: u32) -> Vec<f32> {
    let size = sample_rate as usize;
    let mut ir = vec![0.0f32; size];
    ir[0] = 1.0;
    ir[size - 1] = 1.0;
    // let d1 = (sample_rate as usize / 4).min(size - 1);
    // let d2 = (sample_rate as usize / 2).min(size - 1);
    // ir[d1] = 0.45;
    // ir[d2] = 0.25;
    //
    // // Add a light exponentially decaying tail.
    // for (i, s) in ir.iter_mut().enumerate().skip(1) {
    //     let t = i as f32 / sample_rate as f32;
    //     *s += (-8.0 * t).exp() * 0.02 * (2.0 * std::f32::consts::PI * 180.0 * t).sin();
    // }

    ir
}

fn normalize_if_needed(samples: &mut [f32]) {
    let peak = samples.iter().map(|x| x.abs()).fold(0.0_f32, |a, b| a.max(b));

    if peak > 1.0 {
        let gain = 0.98 / peak;
        for s in samples {
            *s *= gain;
        }
    }
}
