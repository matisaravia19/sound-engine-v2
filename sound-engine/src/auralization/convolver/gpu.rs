//! Synchronous GPU transforms and spectral multiplication for audio convolution.

use crate::acoustics::IrSample;
use crate::core::error::{SoundError, SoundResult};
use crate::gpu::StorageBufferWrite;
use crate::gpu::backend::VkBackend;
use crate::gpu::compute::{
    BufferBarrierSpec, ComputePipelineSpec, DispatchSpec, PipelineId, PushConstantSpec,
};
use crate::gpu::memory::BufferHandle;
use crate::gpu::resource::OwnedFence;
use crate::gpu::shader::ShaderStage;
use ash::vk;
use std::sync::Arc;
use vkfft_rs::plan::{DeviceHandles, FFTPlan, FFTPlanBuilder};

const MULTIPLY_SHADER_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/auralization/shaders/multiply.comp.glsl"
);

/// Owns the plans, scratch buffers, and pipeline for one convolution processor.
/// Mutable access reserves scratch storage across all synchronous submissions.
pub(super) struct GpuConvolution {
    /// Signal workspace with normalized inverse transforms.
    signal: FftWorkspace,
    /// Reusable forward workspace for preparing new impulse responses.
    ir: FftWorkspace,
    /// Retained after both plans because VkFFT borrows it for initialization uploads.
    _initialization_fence: OwnedFence,
    /// Backend-owned pipeline released when this processor is dropped.
    multiply_pipeline_id: PipelineId,
    /// Retains the borrowed device and command pool until every plan is destroyed.
    gpu: Arc<VkBackend>,
}

/// Keeps a VkFFT plan paired with its bound buffer, destroying the plan first.
/// The enclosing processor retains the backend and initialization fence.
struct FftWorkspace {
    plan: FFTPlan,
    buffer: BufferHandle,
}

impl GpuConvolution {
    /// Creates workspaces for a power-of-two length validated by the convolver.
    pub(super) fn new(gpu: Arc<VkBackend>, fft_size: usize) -> SoundResult<Self> {
        let initialization_fence = OwnedFence::new(gpu.device_context())?;
        let signal = FftWorkspace::new(&gpu, fft_size, true, initialization_fence.handle)?;
        // Reuse the IR plan: rebuilding it on every moving-source update would
        // put expensive native plan construction on the playback path.
        let ir = FftWorkspace::new(&gpu, fft_size, false, initialization_fence.handle)?;
        let multiply_pipeline_id = create_multiply_pipeline(&gpu)?;
        Ok(Self {
            signal,
            ir,
            _initialization_fence: initialization_fence,
            multiply_pipeline_id,
            gpu,
        })
    }

    /// Rejects spectra from another allocator or with a different FFT length.
    pub(super) fn validate_spectrum(&self, spectrum: &BufferHandle) -> SoundResult<()> {
        self.gpu.memory().validate_owner(spectrum)?;
        if spectrum.size != self.signal.buffer.size {
            return Err(SoundError::invalid_argument(
                "Impulse response FFT size must match this convolver",
            ));
        }
        Ok(())
    }

    /// Uploads a validated IR prefix, zeroes its padding, and returns an owned spectrum.
    pub(super) fn prepare_ir(&mut self, samples: &[IrSample]) -> SoundResult<BufferHandle> {
        self.gpu.memory().clear_buffer(&self.ir.buffer)?;
        self.gpu.memory().upload_typed(&self.ir.buffer, samples)?;
        self.transform_staged_ir()
    }

    /// Copies a validated time-domain prefix without a CPU readback and transforms it.
    pub(super) fn prepare_gpu_ir(
        &mut self,
        source: &BufferHandle,
        sample_count: u32,
    ) -> SoundResult<BufferHandle> {
        self.gpu.memory().validate_owner(source)?;
        let copy_bytes = u64::from(sample_count) * size_of::<IrSample>() as u64;
        self.gpu
            .compute()
            .submit_compute_and_wait(|command_buffer| unsafe {
                // The fill overlaps the copy, so explicitly order both transfer
                // writes before making the padded input available to the FFT.
                self.gpu
                    .memory()
                    .record_clear_buffer(command_buffer, &self.ir.buffer)?;
                self.gpu.compute().record_buffer_barrier(
                    command_buffer,
                    &BufferBarrierSpec::transfer_write_to_transfer_write(self.ir.buffer.buffer),
                );
                self.gpu.memory().record_copy_buffer(
                    command_buffer,
                    source,
                    &self.ir.buffer,
                    copy_bytes,
                    Some(
                        BufferBarrierSpec::transfer_write_to_compute_shader_read_write(
                            self.ir.buffer.buffer,
                        ),
                    ),
                )
            })?;
        self.transform_staged_ir()
    }

    /// Convolves a full padded complex block in place with a validated IR spectrum.
    /// The caller packs mono input as real components before entering this method.
    pub(super) fn convolve(
        &mut self,
        signal: &mut [[f32; 2]],
        spectrum: &BufferHandle,
    ) -> SoundResult<()> {
        // A full CPU staging block overwrites every bin, so a preceding GPU fill
        // is redundant. Upload ordering is handled by the allocator's transfer.
        debug_assert_eq!(size_of_val(signal) as u64, self.signal.buffer.size);
        self.gpu
            .memory()
            .upload_typed(&self.signal.buffer, signal)?;

        self.gpu
            .compute()
            .submit_compute_and_wait(|command_buffer| unsafe {
                // Keep forward FFT, multiplication, and inverse FFT in one ordered
                // submission. This owner retains both plans and all bound buffers.
                self.signal.plan.append(command_buffer).map_err(|error| {
                    SoundError::external(format!("Signal forward launch failed: {error}"))
                })?;
                self.gpu.compute().record_buffer_barrier(
                    command_buffer,
                    &BufferBarrierSpec::compute_shader_write_to_compute_shader_read_write(
                        self.signal.buffer.buffer,
                    ),
                );
                self.append_multiply(command_buffer, spectrum)?;
                self.gpu.compute().record_buffer_barrier(
                    command_buffer,
                    &BufferBarrierSpec::compute_shader_write_to_compute_shader_read_write(
                        self.signal.buffer.buffer,
                    ),
                );
                self.signal
                    .plan
                    .append_inverse(command_buffer)
                    .map_err(|error| {
                        SoundError::external(format!("Signal inverse launch failed: {error}"))
                    })?;
                Ok(())
            })?;

        self.gpu
            .memory()
            .download_typed(&self.signal.buffer, signal.len(), signal)
    }

    fn transform_staged_ir(&mut self) -> SoundResult<BufferHandle> {
        let spectrum = self
            .gpu
            .memory()
            .create_storage_buffer(self.ir.buffer.size)?;
        self.gpu
            .compute()
            .submit_compute_and_wait(|command_buffer| unsafe {
                // The queue lock serializes recording; the synchronous wait keeps
                // the plan and its scratch buffer alive through the following copy.
                self.ir.plan.append(command_buffer).map_err(|error| {
                    SoundError::external(format!("IR forward launch failed: {error}"))
                })?;
                self.gpu.compute().record_buffer_barrier(
                    command_buffer,
                    &BufferBarrierSpec::compute_shader_write_to_transfer_read(
                        self.ir.buffer.buffer,
                    ),
                );
                self.gpu.memory().record_copy_buffer(
                    command_buffer,
                    &self.ir.buffer,
                    &spectrum,
                    self.ir.buffer.size,
                    Some(
                        BufferBarrierSpec::transfer_write_to_compute_shader_read_write(
                            spectrum.buffer,
                        ),
                    ),
                )
            })?;
        Ok(spectrum)
    }

    fn append_multiply(
        &self,
        command_buffer: vk::CommandBuffer,
        spectrum: &BufferHandle,
    ) -> SoundResult<()> {
        let count = (self.signal.buffer.size / size_of::<[f32; 2]>() as u64) as u32;
        let dispatch = DispatchSpec {
            pipeline_id: self.multiply_pipeline_id,
            storage_buffers: vec![
                StorageBufferWrite::new(0, self.signal.buffer.buffer),
                StorageBufferWrite::new(1, spectrum.buffer),
            ],
            push_constants: count.to_ne_bytes().to_vec(),
            push_constant_offset: 0,
            groups: [count.div_ceil(256), 1, 1],
        };
        // The private caller records on this backend and retains both spectra
        // until the enclosing synchronous submission completes.
        unsafe {
            self.gpu
                .compute()
                .record_dispatch(command_buffer, &dispatch)
        }
    }
}

impl Drop for GpuConvolution {
    fn drop(&mut self) {
        // Synchronous submissions have finished; the shared backend should not
        // retain a private multiply pipeline after its processor is released.
        self.gpu
            .compute()
            .release_pipeline(self.multiply_pipeline_id);
    }
}

impl FftWorkspace {
    fn new(
        gpu: &VkBackend,
        fft_size: usize,
        normalize_inverse: bool,
        fence: vk::Fence,
    ) -> SoundResult<Self> {
        gpu.compute().with_queue(|queue, command_pool| {
            let handles = DeviceHandles {
                physical_device: gpu.device().physical_device,
                device: gpu.device().device.handle(),
                queue,
                command_pool,
                fence,
            };
            let buffer = gpu
                .memory()
                .create_storage_buffer(fft_size as u64 * size_of::<[f32; 2]>() as u64)?;
            let builder = FFTPlanBuilder::new(&handles)
                .with_length(fft_size as u64)
                .with_buffer(buffer.buffer, buffer.size);
            let builder = if normalize_inverse {
                builder.with_normalization()
            } else {
                builder
            };
            // VkFFT borrows these handles. The processor retains the backend and
            // fence; this workspace retains the buffer until the plan is destroyed.
            // with_queue serializes initialization submissions and command-pool use.
            let plan = unsafe { builder.build() }
                .map_err(|error| SoundError::external(format!("FFT plan build failed: {error}")))?;
            Ok(Self {
                plan,
                buffer,
            })
        })
    }
}

fn create_multiply_pipeline(gpu: &VkBackend) -> SoundResult<PipelineId> {
    let shader_id = gpu
        .shaders()
        .load_glsl_file(ShaderStage::Compute, MULTIPLY_SHADER_PATH)?;
    gpu.compute().create_pipeline(ComputePipelineSpec {
        shader_id,
        storage_buffer_bindings: vec![0, 1],
        push_constant_ranges: vec![PushConstantSpec::new(0, size_of::<u32>() as u32)],
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/auralization/convolver/gpu.rs"]
mod tests;
