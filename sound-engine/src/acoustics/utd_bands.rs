//! GPU synthesis of interpolated complex pressure carried by traced rays.
use super::GpuIrSnapshot;
use crate::core::error::{SoundError, SoundResult};
use crate::gpu::compute::{
    BufferBarrierSpec, ComputePipelineSpec, DescriptorBindingSpec, DescriptorWrite, DispatchSpec, MemoryBarrierSpec,
    PipelineId, PushConstantSpec,
};
use crate::gpu::{backend::VkBackend, memory::BufferHandle, shader::ShaderStage};
use ash::vk;
use std::sync::Arc;
use vkfft_rs::plan::{DeviceHandles, FFTPlan, FFTPlanBuilder};

/// Exclusive FFT scratch owner. The caller keeps its Vulkan backend alive and
/// serializes queries; no cached IR aliases these mutable work buffers.
pub(super) struct BandSynthesis {
    /// Forward plan borrows `scratch` and the caller's device handles.
    forward: FFTPlan,
    /// Normalized inverse plan borrows `combined` and the same device.
    inverse: FFTPlan,
    /// Contiguous zero-padded timelines/spectra for all bands and ears.
    scratch: BufferHandle,
    /// Accumulated stereo spectrum, packed as left + i right after inverse FFT.
    combined: BufferHandle,
    /// Extraction, spectral interpolation, and final frame assembly kernel.
    kernel: PipelineId,
    /// Fixed transform sizes and band definition for this engine.
    constants: Constants,
}

/// Shared std430 scalar layout, copied as one push-constant block.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Constants {
    /// Time-domain sample count in each ray-deposition plane.
    samples: u32,
    /// Zero-padded transform size.
    fft_size: u32,
    /// Engine sample rate in Hz.
    sample_rate: u32,
    /// Number of output ears (one for mono).
    channels: u32,
    /// Number of logarithmic interpolation nodes.
    bands: u32,
    /// Operation: extract, accumulate, or assemble.
    stage: u32,
    /// Last node frequency in Hz.
    maximum_frequency: f32,
}

impl BandSynthesis {
    /// Allocates reusable plans and work buffers during engine setup.
    pub(super) fn new(
        gpu: &VkBackend,
        samples: u32,
        sample_rate: u32,
        channels: u32,
        bands: u32,
        maximum_frequency: f32,
    ) -> SoundResult<Self> {
        let fft_size = samples
            .checked_next_power_of_two()
            .ok_or_else(|| SoundError::invalid_argument("IR FFT size overflow"))?;
        let scratch = gpu.memory().create_storage_buffer(u64::from(fft_size) * 8 * u64::from(bands) * u64::from(channels))?;
        let combined = gpu.memory().create_storage_buffer(fft_size as u64 * 8)?;
        let handles = DeviceHandles {
            physical_device: gpu.device().physical_device,
            device: gpu.device().device.handle(),
            // Borrowed only for plan setup; launches use synchronized submissions.
            queue: unsafe { gpu.compute().queue_handle()? },
            command_pool: unsafe { gpu.compute().command_pool_handle()? },
            fence: vk::Fence::null(),
        };
        // The plans borrow stable device/buffer handles. Field order drops both
        // plans before their buffers; the owning acoustic engine outlives them.
        let forward = FFTPlanBuilder::new(&handles)
            .with_single_dimension(fft_size as u64)
            .with_batch_count(u64::from(bands) * u64::from(channels))
            .with_buffer(scratch.buffer, scratch.size)
            .build()
            .map_err(|e| SoundError::external(format!("UTD band forward FFT: {e:?}")))?;
        let inverse = FFTPlanBuilder::new(&handles)
            .with_single_dimension(fft_size as u64)
            .with_buffer(combined.buffer, combined.size)
            .with_normalization()
            .build()
            .map_err(|e| SoundError::external(format!("UTD band inverse FFT: {e:?}")))?;
        let shader_id = gpu.shaders().load_glsl_file(
            ShaderStage::Compute,
            concat!(env!("CARGO_MANIFEST_DIR"), "/src/acoustics/shaders/utd_bands.comp.glsl"),
        )?;
        let kernel = gpu.compute().create_pipeline(ComputePipelineSpec {
            shader_id,
            descriptor_bindings: (0..4).map(DescriptorBindingSpec::storage_buffer).collect(),
            push_constant_ranges: vec![PushConstantSpec::new(0, std::mem::size_of::<Constants>() as u32)],
        })?;
        Ok(Self {
            forward,
            inverse,
            scratch,
            combined,
            kernel,
            constants: Constants {
                samples,
                fft_size,
                sample_rate,
                channels,
                bands,
                stage: 0,
                maximum_frequency,
            },
        })
    }

    /// Reconstructs both ears entirely on GPU; only ray-deposited pressure is used.
    pub(super) fn synthesize(&mut self, gpu: &VkBackend, raw: &GpuIrSnapshot) -> SoundResult<GpuIrSnapshot> {
        let input = raw
            .buffer
            .as_ref()
            .ok_or_else(|| SoundError::invalid_argument("missing ray band buffer"))?;
        let byte_size = self.constants.samples as usize * 8;
        let output = Arc::new(gpu.memory().create_storage_buffer(byte_size as u64)?);
        gpu.compute().submit_compute_and_wait(|command| {
            let mut constants = self.constants;
            self.dispatch(gpu, command, input, &output, constants)?;
            // One batched transform retains separate complex pressure for every
            // band/ear. Interpolation overwrites all bins, including zero bins.
            barrier(gpu, command);
            self.forward.append(command)
                .map_err(|e| SoundError::external(format!("UTD batched FFT launch: {e:?}")))?;
            barrier(gpu, command);
            constants.stage = 1;
            self.dispatch(gpu, command, input, &output, constants)?;
            barrier(gpu, command);
            self.inverse
                .append_inverse(command)
                .map_err(|e| SoundError::external(format!("UTD inverse launch: {e:?}")))?;
            barrier(gpu, command);
            let mut constants = self.constants;
            constants.stage = 2;
            self.dispatch(gpu, command, input, &output, constants)?;
            gpu.compute().record_buffer_barrier(
                command,
                &BufferBarrierSpec::compute_shader_write_to_compute_or_transfer_read(output.buffer),
            );
            Ok(())
        })?;
        Ok(GpuIrSnapshot {
            buffer: Some(output),
            byte_size,
            sample_count: self.constants.samples,
            sample_rate: raw.sample_rate,
            energy: None,
            scene_version: raw.scene_version,
            query_id: raw.query_id,
        })
    }

    /// Records a stage with fresh descriptors, avoiding mutation of in-flight sets.
    fn dispatch(
        &self,
        gpu: &VkBackend,
        command: vk::CommandBuffer,
        input: &BufferHandle,
        output: &BufferHandle,
        constants: Constants,
    ) -> SoundResult<()> {
        gpu.compute().record_dispatch(
            command,
            &DispatchSpec {
                pipeline_id: self.kernel,
                descriptor_writes: vec![
                    DescriptorWrite::storage_buffer(0, input.buffer),
                    DescriptorWrite::storage_buffer(1, self.scratch.buffer),
                    DescriptorWrite::storage_buffer(2, self.combined.buffer),
                    DescriptorWrite::storage_buffer(3, output.buffer),
                ],
                push_constants: bytemuck::bytes_of(&constants).to_vec(),
                push_constant_offset: 0,
                groups: [constants.fft_size.div_ceil(256),
                    if constants.stage == 0 { constants.bands * constants.channels } else { 1 }, 1],
            },
        )
    }
}

/// Orders dependent FFT and interpolation reads/writes on the compute queue.
fn barrier(gpu: &VkBackend, command: vk::CommandBuffer) {
    gpu.compute().record_memory_barrier(
        command,
        &MemoryBarrierSpec {
            src_stage: vk::PipelineStageFlags::COMPUTE_SHADER,
            dst_stage: vk::PipelineStageFlags::COMPUTE_SHADER,
            src_access: vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            dst_access: vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acoustics::IrSample;
    use realfft::num_complex::Complex32;

    #[test]
    #[ignore = "requires Vulkan and glslangValidator; run hardware tests serially"]
    fn batched_stereo_synthesis_matches_independent_long_ir() {
        // Synthetic band impulses isolate reconstruction from ray-sampling noise.
        // The runtime still obtains every band timeline exclusively from rays.
        let gpu = VkBackend::new().unwrap();
        let (samples, bands, channels, rate) = (96000_u32, 16_u32, 2_u32, 48000_u32);
        let fft_size = samples.next_power_of_two() as usize;
        let frequency_max = 20000.0_f32;
        let impulse = |band: usize, ear: usize| {
            (317 + 37*band + 11*ear, Complex32::new(0.05*(band+1) as f32,
                if (band+ear)%2 == 0 { 0.07 } else { -0.03 }))
        };
        let mut input = vec![IrSample::zero(); ((bands*channels+1)*samples) as usize];
        for band in 0..bands as usize {
            for ear in 0..channels as usize {
                let (time, pressure) = impulse(band, ear);
                input[(band*channels as usize+ear)*samples as usize+time] = IrSample::new(pressure.re, pressure.im);
            }
        }
        let surface_index = (bands*channels*samples) as usize;
        input[surface_index+17] = IrSample::new(0.25, -0.125);
        input[surface_index+samples as usize-1] = IrSample::new(-0.05, 0.1);
        let buffer = Arc::new(gpu.memory().create_storage_buffer((input.len()*8) as u64).unwrap());
        gpu.memory().upload_typed(&buffer, &input).unwrap();
        let raw = GpuIrSnapshot { buffer: Some(buffer), byte_size: input.len()*8,
            sample_rate: rate, sample_count: input.len() as u32, energy: None, scene_version: 1, query_id: 1 };
        let mut synthesis = BandSynthesis::new(&gpu, samples, rate, channels, bands, frequency_max).unwrap();
        let snapshot = synthesis.synthesize(&gpu, &raw).unwrap();
        let mut actual = vec![IrSample::zero(); samples as usize];
        gpu.memory().download_typed(snapshot.buffer.as_ref().unwrap(), actual.len(), &mut actual).unwrap();
        let inverse = realfft::RealFftPlanner::<f32>::new().plan_fft_inverse(fft_size);
        for ear in 0..channels as usize {
            let mut spectrum = inverse.make_input_vec();
            for (bin, value) in spectrum.iter_mut().enumerate().skip(1) {
                let frequency = bin as f32 * rate as f32 / fft_size as f32;
                if frequency > frequency_max { continue; }
                let node = (frequency/(rate as f32/fft_size as f32)).ln()
                    / (frequency_max/(rate as f32/fft_size as f32)).ln() * (bands-1) as f32;
                // CPU reference sums all triangular basis functions; the GPU
                // gathers only the two nonzero neighboring band spectra.
                for band in 0..bands as usize {
                    let weight = (1.0-(node-band as f32).abs()).max(0.0);
                    let (time, pressure) = impulse(band, ear);
                    let phase = -std::f64::consts::TAU * bin as f64 * time as f64 / fft_size as f64;
                    *value += pressure * Complex32::new(phase.cos() as f32, phase.sin() as f32) * weight;
                }
            }
            let mut expected = inverse.make_output_vec();
            inverse.process(&mut spectrum, &mut expected).unwrap();
            let (mut error, mut norm) = (0.0_f64, 0.0_f64);
            for i in 0..samples as usize {
                let direct = input[surface_index+i];
                let expected = expected[i]/fft_size as f32 + if ear == 0 { direct.left } else { direct.right };
                let actual = if ear == 0 { actual[i].left } else { actual[i].right };
                assert!(actual.is_finite());
                error += f64::from(actual-expected).powi(2);
                norm += f64::from(expected).powi(2);
            }
            let relative = (error/norm).sqrt();
            eprintln!("long stereo batched synthesis ear={ear} relative L2={relative}");
            assert!(relative < 2e-5, "batched FFT changed complex pressure");
        }
        // Repeated queries clear the complete scratch and assembled spectrum.
        let zero = vec![IrSample::zero(); input.len()];
        gpu.memory().upload_typed(raw.buffer.as_ref().unwrap(), &zero).unwrap();
        let silent = synthesis.synthesize(&gpu, &raw).unwrap();
        gpu.memory().download_typed(silent.buffer.as_ref().unwrap(), actual.len(), &mut actual).unwrap();
        assert!(actual.iter().all(|s| s.left == 0.0 && s.right == 0.0));
    }
}
