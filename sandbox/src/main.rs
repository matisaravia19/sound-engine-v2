use std::error::Error;
use realfft::RealFftPlanner;

fn read_wav_mono_f32(path: &str) -> Result<(Vec<f32>, u32), Box<dyn Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let sr = spec.sample_rate;
    let channels = spec.channels as usize;

    // Read as i16 (common for PCM WAV). If your WAV is f32 or i32, adjust accordingly.
    let mut interleaved: Vec<f32> = Vec::new();
    for s in reader.samples::<i16>() {
        interleaved.push(s? as f32 / i16::MAX as f32);
    }

    // Downmix to mono (average channels). If you want stereo convolution, keep it interleaved.
    let mut mono = Vec::with_capacity(interleaved.len() / channels);
    for frame in interleaved.chunks_exact(channels) {
        let sum: f32 = frame.iter().copied().sum();
        mono.push(sum / channels as f32);
    }

    Ok((mono, sr))
}

pub fn convolve_fft_realfft(x: &[f32], h: &[f32]) -> Vec<f32> {
    let out_len = x.len() + h.len() - 1;
    let n = out_len.next_power_of_two();

    // Planner caches internally; reuse it if you convolve repeatedly.
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(n);
    let c2r = planner.plan_fft_inverse(n);

    // Prepare zero-padded input buffers
    let mut x_time = r2c.make_input_vec();
    let mut h_time = r2c.make_input_vec();

    x_time[..x.len()].copy_from_slice(x);
    h_time[..h.len()].copy_from_slice(h);

    // Frequency-domain buffers
    let mut x_freq = r2c.make_output_vec();
    let mut h_freq = r2c.make_output_vec();

    // Forward FFTs
    r2c.process(&mut x_time, &mut x_freq).expect("FFT x failed");
    r2c.process(&mut h_time, &mut h_freq).expect("FFT h failed");

    // Multiply spectra: Y[k] = X[k] * H[k]
    for (y, hk) in x_freq.iter_mut().zip(h_freq.iter()) {
        *y *= *hk;
    }

    // Inverse FFT back to time domain
    let mut y_time = c2r.make_output_vec();
    c2r.process(&mut x_freq, &mut y_time).expect("iFFT failed");

    // IMPORTANT: realfft/rustfft do not normalize either direction.
    // After FFT + iFFT, scale by 1/N.
    let scale = 1.0 / n as f32;
    for s in &mut y_time {
        *s *= scale;
    }

    // Trim to linear-convolution length
    y_time.truncate(out_len);
    y_time
}

pub fn save_wav_f32(path: &str, samples: &[f32], sample_rate: u32, ) -> Result<(), Box<dyn Error>> {
    let spec = hound::WavSpec {
        channels: 1, // mono
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = hound::WavWriter::create(path, spec)?;

    for &sample in samples {
        writer.write_sample(sample)?;
    }

    writer.finalize()?;
    Ok(())
}

fn main() {
    let (audio, sr) = read_wav_mono_f32("C:\\Users\\matis\\OneDrive\\Documentos\\Fing\\Tesis\\explosion.wav").expect("Failed to read audio file");

    let ir_length = sr as usize;
    let mut ir = Vec::from_iter(std::iter::repeat(0.0).take(ir_length));
    ir[0] = 1.0;
    ir[ir_length - 1] = 1.0;

    let start = std::time::Instant::now();
    let output = convolve_fft_realfft(&audio, &ir);
    let duration = start.elapsed();
    println!("Convolution took: {:?}", duration);

    save_wav_f32("output.wav", &output, sr).expect("Failed to save output WAV");
}
