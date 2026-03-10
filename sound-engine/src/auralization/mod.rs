use std::sync::Arc;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use realfft::num_complex::Complex;

pub struct Auralizer {
    block_size: usize,
    ir_size: usize,
    fft_size: usize,
    overlap_buffer: Vec<f32>,
    forward_fft: Arc<dyn RealToComplex<f32>>,
    inverse_fft: Arc<dyn ComplexToReal<f32>>,
    fft_input: Vec<f32>,
    fft_output: Vec<Complex<f32>>,
}

impl Auralizer {
    pub fn new(block_size: usize, ir_size: usize) -> Self {
        let fft_size = (block_size + ir_size - 1).next_power_of_two();

        let mut planner = RealFftPlanner::<f32>::new();
        let forward_fft = planner.plan_fft_forward(fft_size);
        let inverse_fft = planner.plan_fft_inverse(fft_size);

        let overlap_buffer = vec![0.0; ir_size - 1];
        let fft_input = forward_fft.make_input_vec();
        let fft_output = forward_fft.make_output_vec();

        Self {
            block_size,
            ir_size,
            fft_size,
            overlap_buffer,
            forward_fft,
            inverse_fft,
            fft_input,
            fft_output,
        }
    }

    pub fn process_block(&mut self, input_block: &[f32], ir_fft: &[Complex<f32>]) -> Vec<f32> {
        let mut x_padded = vec![0.0; self.fft_size];
        for (i, &sample) in input_block.iter().enumerate() {
            x_padded[i] = sample;
        }

        let mut x_fft = vec![Complex { re: 0.0, im: 0.0 }; self.fft_size / 2 + 1];
        self.forward_fft.process(&mut x_padded, &mut self.fft_output).expect("FFT forward failed");

        // 3) Multiplicación en el dominio de la frecuencia
        let mut y_fft = vec![Complex { re: 0.0, im: 0.0 }; self.fft_size / 2 + 1];
        for k in 0..self.fft_size / 2 + 1 {
            //y_fft[k] = x_fft[k] * self.h_fft[k];
        }

        // 4) FFT inversa al dominio del tiempo
        let mut y_padded = vec![0.0; self.fft_size];
        self.inverse_fft.process(&mut y_fft, &mut y_padded).expect("FFT inverse failed");

        // 5) Agregar el solapamiento del bloque anterior
        for i in 0..(self.ir_size - 1) {
            y_padded[i] += self.overlap_buffer[i];
        }

        // 6) Extraer los primeros L muestras (normalización incluida)
        let mut output = vec![0.0; self.block_size];
        for i in 0..self.block_size {
            output[i] = y_padded[i] / self.fft_size as f32;
        }

        // 7) Guardar la cola para el siguiente bloque
        for i in 0..(self.ir_size - 1) {
            self.overlap_buffer[i] = y_padded[self.block_size + i] / self.fft_size as f32;
        }

        output
    }
}
