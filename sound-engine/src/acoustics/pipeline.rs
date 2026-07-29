use crate::acoustics::{GpuIrSnapshot, IrSample, IrSnapshot};
use crate::core::config::{EngineConfig, OutputChannels};
use crate::core::error::{SoundError, SoundResult};
use crate::gpu::backend::VkBackend;
use crate::gpu::compute::BufferBarrierSpec;
use crate::gpu::rt::{
    BlasBuildSpec, BlasId, RtDescriptorBindingSpec, RtDescriptorWrite, RtInstanceSpec, RtMeshBuffers, RtMeshSpec,
    RtPipelineId, RtPipelineSpec, RtPushConstantSpec, RtShaderGroupSpec, RtShaderStageSpec, RtTraceSpec, TlasBuildSpec,
};
use crate::gpu::shader::ShaderStage;
use crate::scene::SceneManager;
use ash::vk;
use glam::{Mat4, Vec3};
use std::mem::size_of;
use std::path::PathBuf;
use std::sync::Arc;

const SHADER_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/acoustics/shaders");
const SPEED_OF_SOUND_METERS_PER_SECOND: f32 = 343.0;
#[cfg(test)]
const SURFACE_GEOMETRY_MASK: u8 = 0x01;
const DIFFRACTION_GEOMETRY_MASK: u8 = 0x02;
const DIFFRACTION_INSTANCE_CUSTOM_INDEX: u32 = u32::MAX - 1;
const SURFACE_CONTINUATION_HIT_GROUP_OFFSET: u32 = 0;
#[cfg(test)]
const SURFACE_VISIBILITY_HIT_GROUP_OFFSET: u32 = 1;
const DIFFRACTION_CONTINUATION_HIT_GROUP_OFFSET: u32 = 2;

/// Source/listener pair that should produce one impulse response.
#[derive(Debug, Clone, Copy)]
pub struct AcousticQuery {
    /// Application-defined query identifier copied into the returned IR snapshot.
    pub query_id: u32,
    /// Source position in world meters.
    pub source_position: Vec3,
    /// Listener position in world meters.
    pub listener_position: Vec3,
    /// Listener right-ear direction used for direction-based stereo panning.
    pub listener_right: Vec3,
    /// Reference source pressure amplitude.
    pub source_amplitude: f32,
}

/// Listener-originated acoustic path tracer that deposits pressure paths directly into a GPU IR.
///
/// The ray-generation invocation owns all path state and performs sequential
/// continuation and source-visibility traces. The RT pipeline has recursion
/// depth one because hit shaders only report compact traversal results.
pub struct AcousticPipeline {
    config: EngineConfig,
    listener_pipeline: RtPipelineId,
    /// Triangle storage kept alive for the mask-zero empty-scene sentinel BLAS.
    _empty_scene_mesh: RtMeshBuffers,
    empty_scene_blas: BlasId,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ListenerTracePushConstants {
    source_position: [f32; 3],
    source_amplitude: f32,
    listener_position: [f32; 3],
    speed_of_sound: f32,
    listener_right: [f32; 3],
    maximum_path_length: f32,
    listener_ray_count: u32,
    max_bounces: u32,
    sample_rate: u32,
    sample_count: u32,
    output_channels: u32,
    sampling_seed: u32,
    minimum_throughput: f32,
    _pad0: u32,
}

unsafe impl bytemuck::Zeroable for ListenerTracePushConstants {}
unsafe impl bytemuck::Pod for ListenerTracePushConstants {}

impl AcousticPipeline {
    /// Creates the listener-originated RT pipeline and the empty-scene sentinel BLAS.
    pub fn new(gpu: &VkBackend, config: EngineConfig) -> SoundResult<Self> {
        validate_config(config)?;

        let raygen = load_shader(gpu, ShaderStage::RayGeneration, "listener_backward.rgen.glsl")?;
        let continuation_miss = load_shader(gpu, ShaderStage::RayMiss, "continuation.rmiss.glsl")?;
        let visibility_miss = load_shader(gpu, ShaderStage::RayMiss, "visibility.rmiss.glsl")?;
        let surface_closest_hit = load_shader(gpu, ShaderStage::RayClosestHit, "listener_backward_surface.rchit.glsl")?;
        let visibility_closest_hit = load_shader(gpu, ShaderStage::RayClosestHit, "visibility.rchit.glsl")?;
        let diffraction_closest_hit = load_shader(
            gpu,
            ShaderStage::RayClosestHit,
            "listener_backward_diffraction.rchit.glsl",
        )?;
        let diffraction_intersection = load_shader(gpu, ShaderStage::RayIntersection, "diffraction.rint.glsl")?;

        let listener_pipeline = gpu.rt().create_pipeline(RtPipelineSpec {
            shaders: vec![
                RtShaderStageSpec::new(raygen),
                RtShaderStageSpec::new(continuation_miss),
                RtShaderStageSpec::new(visibility_miss),
                RtShaderStageSpec::new(surface_closest_hit),
                RtShaderStageSpec::new(visibility_closest_hit),
                RtShaderStageSpec::new(diffraction_closest_hit),
                RtShaderStageSpec::new(diffraction_intersection),
            ],
            // Hit records must remain in this order. Surface instances use SBT
            // offset 0, visibility traces add 1, and diffraction instances use
            // offset 2. The shaders use a zero SBT stride for both ray types.
            groups: vec![
                RtShaderGroupSpec::Raygen { shader: 0 },
                RtShaderGroupSpec::Miss { shader: 1 },
                RtShaderGroupSpec::Miss { shader: 2 },
                RtShaderGroupSpec::TrianglesHit { closest_hit_shader: 3 },
                RtShaderGroupSpec::TrianglesHit { closest_hit_shader: 4 },
                RtShaderGroupSpec::ProceduralHit {
                    closest_hit_shader: 5,
                    intersection_shader: 6,
                },
            ],
            // The compact descriptor set contains only traversal state, the
            // final IR, and immutable scene lookup tables. There is no
            // per-ray hit or contribution storage.
            descriptor_bindings: vec![
                RtDescriptorBindingSpec::acceleration_structure(0),
                RtDescriptorBindingSpec::storage_buffer(1),
                RtDescriptorBindingSpec::storage_buffer(2),
                RtDescriptorBindingSpec::storage_buffer(3),
                RtDescriptorBindingSpec::storage_buffer(4),
            ],
            push_constant_ranges: vec![RtPushConstantSpec::new(
                0,
                size_of::<ListenerTracePushConstants>() as u32,
            )],
            max_ray_recursion_depth: 1,
        })?;

        // RtContext currently rejects an empty TLAS. This masked triangle keeps
        // direct-path queries valid without participating in traversal.
        let empty_scene_mesh = gpu.rt().upload_mesh(RtMeshSpec {
            vertices: &[
                Vec3::new(-0.001, -0.001, 0.0),
                Vec3::new(0.001, -0.001, 0.0),
                Vec3::new(0.0, 0.001, 0.0),
            ],
            indices: None,
            opaque: true,
        })?;
        let empty_scene_blas = gpu.rt().build_blas(BlasBuildSpec {
            mesh: &empty_scene_mesh,
            flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
        })?;

        Ok(Self {
            config,
            listener_pipeline,
            _empty_scene_mesh: empty_scene_mesh,
            empty_scene_blas,
        })
    }

    /// Builds one CPU-resident impulse response for a source/listener query.
    pub fn build_ir(
        &mut self,
        gpu: &VkBackend,
        scene: &mut SceneManager,
        query: AcousticQuery,
    ) -> SoundResult<IrSnapshot> {
        let gpu_ir = self.build_ir_gpu(gpu, scene, query)?;
        self.download_gpu_ir(gpu, &gpu_ir)
    }

    /// Builds one GPU-resident time-domain pressure IR.
    pub fn build_ir_gpu(
        &mut self,
        gpu: &VkBackend,
        scene: &mut SceneManager,
        query: AcousticQuery,
    ) -> SoundResult<GpuIrSnapshot> {
        validate_query(query)?;
        let gpu_scene = scene.sync_gpu_if_needed(gpu)?;
        let tlas_id = self.build_query_tlas(gpu, gpu_scene.instances(), gpu_scene.diffraction_blas_id())?;
        let ir_buffer = Arc::new(
            gpu.memory()
                .create_storage_buffer(ir_buffer_size(self.config.sound.ir_num_samples) as u64)?,
        );
        // Every query receives independent zeroed pressure storage. Atomic
        // accumulation cannot inherit data from allocator memory reused by a
        // previous query.
        gpu.memory().clear_buffer(&ir_buffer)?;

        let constants = ListenerTracePushConstants {
            source_position: query.source_position.to_array(),
            source_amplitude: query.source_amplitude,
            listener_position: query.listener_position.to_array(),
            speed_of_sound: SPEED_OF_SOUND_METERS_PER_SECOND,
            listener_right: query.listener_right.to_array(),
            maximum_path_length: self.config.acoustics.maximum_path_length,
            listener_ray_count: self.config.acoustics.listener_ray_count,
            max_bounces: self.config.acoustics.max_bounces,
            sample_rate: self.config.sound.sample_rate,
            sample_count: self.config.sound.ir_num_samples,
            output_channels: output_channels_to_shader(self.config.sound.output_channels),
            sampling_seed: self.config.acoustics.sampling_seed,
            minimum_throughput: self.config.acoustics.minimum_throughput,
            _pad0: 0,
        };

        gpu.compute().submit_compute_and_wait(|command_buffer| {
            gpu.rt().record_trace(
                command_buffer,
                &RtTraceSpec {
                    pipeline_id: self.listener_pipeline,
                    descriptor_writes: vec![
                        RtDescriptorWrite::acceleration_structure(0, tlas_id),
                        RtDescriptorWrite::storage_buffer(1, ir_buffer.buffer),
                        RtDescriptorWrite::storage_buffer(2, gpu_scene.object_buffer().buffer),
                        RtDescriptorWrite::storage_buffer(3, gpu_scene.material_buffer().buffer),
                        RtDescriptorWrite::storage_buffer(4, gpu_scene.diffraction_edge_buffer().buffer),
                    ],
                    push_constants: bytemuck::bytes_of(&constants).to_vec(),
                    push_constant_offset: 0,
                    dimensions: [self.config.acoustics.listener_ray_count, 1, 1],
                    barrier_after_trace: false,
                },
            )?;
            // Ray-generation float atomics are shader writes. Make them visible
            // before this buffer is downloaded, cached, or copied into the
            // convolution pipeline by a later command submission.
            gpu.compute().record_buffer_barrier(
                command_buffer,
                &BufferBarrierSpec::ray_tracing_shader_write_to_compute_or_transfer_read(ir_buffer.buffer),
            );
            Ok(())
        })?;

        Ok(GpuIrSnapshot {
            buffer: Some(ir_buffer),
            byte_size: ir_buffer_size(self.config.sound.ir_num_samples),
            sample_rate: self.config.sound.sample_rate,
            sample_count: self.config.sound.ir_num_samples,
            energy: None,
            scene_version: gpu_scene.version(),
            query_id: query.query_id,
        })
    }

    /// Builds CPU IR snapshots for a batch of source/listener queries.
    pub fn build_irs(
        &mut self,
        gpu: &VkBackend,
        scene: &mut SceneManager,
        queries: &[AcousticQuery],
    ) -> SoundResult<Vec<IrSnapshot>> {
        let mut snapshots = Vec::with_capacity(queries.len());
        for query in queries {
            snapshots.push(self.build_ir(gpu, scene, *query)?);
        }
        Ok(snapshots)
    }

    /// Downloads a GPU-resident pressure IR into a CPU snapshot.
    pub fn download_gpu_ir(&self, gpu: &VkBackend, snapshot: &GpuIrSnapshot) -> SoundResult<IrSnapshot> {
        let ir_buffer = snapshot
            .buffer
            .as_ref()
            .ok_or_else(|| SoundError::invalid_argument("GPU IR snapshot has no buffer"))?;
        let mut samples = vec![IrSample::zero(); snapshot.sample_count as usize];
        gpu.memory()
            .download_typed::<IrSample>(ir_buffer, samples.len(), &mut samples)?;
        let energy = samples
            .iter()
            .map(|sample| sample.left * sample.left + sample.right * sample.right)
            .sum();

        Ok(IrSnapshot {
            sample_rate: snapshot.sample_rate,
            samples,
            energy,
            scene_version: snapshot.scene_version,
            query_id: snapshot.query_id,
        })
    }

    fn build_query_tlas(
        &self,
        gpu: &VkBackend,
        scene_instances: &[RtInstanceSpec],
        diffraction_blas_id: Option<BlasId>,
    ) -> SoundResult<crate::gpu::rt::TlasId> {
        let mut instances = Vec::with_capacity(scene_instances.len() + 1);
        instances.extend_from_slice(scene_instances);
        if let Some(blas_id) = diffraction_blas_id {
            // Mask 0x02 keeps proxy strips out of surface-only visibility rays;
            // SBT offset 2 routes continuation hits to the procedural record.
            instances.push(RtInstanceSpec {
                blas_id,
                transform: Mat4::IDENTITY,
                custom_index: DIFFRACTION_INSTANCE_CUSTOM_INDEX,
                mask: DIFFRACTION_GEOMETRY_MASK,
                sbt_record_offset: DIFFRACTION_CONTINUATION_HIT_GROUP_OFFSET,
            });
        }
        if instances.is_empty() {
            // The RT wrapper requires a non-empty TLAS. A zero mask makes this
            // sentinel unreachable while preserving direct empty-scene queries.
            instances.push(RtInstanceSpec {
                blas_id: self.empty_scene_blas,
                transform: Mat4::IDENTITY,
                custom_index: 0,
                mask: 0,
                sbt_record_offset: SURFACE_CONTINUATION_HIT_GROUP_OFFSET,
            });
        }

        gpu.rt().build_tlas(TlasBuildSpec {
            instances: &instances,
            flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
        })
    }
}

fn load_shader(gpu: &VkBackend, stage: ShaderStage, file: &str) -> SoundResult<crate::gpu::shader::ShaderId> {
    gpu.shaders().load_glsl_file(stage, shader_path(file))
}

fn shader_path(file: &str) -> PathBuf {
    PathBuf::from(SHADER_DIR).join(file)
}

fn validate_config(config: EngineConfig) -> SoundResult<()> {
    if config.sound.sample_rate == 0 || config.sound.ir_num_samples == 0 {
        return Err(SoundError::invalid_argument(
            "IR sample rate and sample count must be non-zero",
        ));
    }
    if config.acoustics.listener_ray_count == 0 {
        return Err(SoundError::invalid_argument("listener_ray_count must be non-zero"));
    }
    if !config.acoustics.maximum_path_length.is_finite() || config.acoustics.maximum_path_length <= 0.0 {
        return Err(SoundError::invalid_argument(
            "maximum_path_length must be finite and positive",
        ));
    }
    if !config.acoustics.minimum_throughput.is_finite() || config.acoustics.minimum_throughput < 0.0 {
        return Err(SoundError::invalid_argument(
            "minimum_throughput must be finite and non-negative",
        ));
    }
    Ok(())
}

fn validate_query(query: AcousticQuery) -> SoundResult<()> {
    if !query.source_position.is_finite() || !query.listener_position.is_finite() || !query.listener_right.is_finite() {
        return Err(SoundError::invalid_argument("acoustic query vectors must be finite"));
    }
    if !query.source_amplitude.is_finite() || query.source_amplitude < 0.0 {
        return Err(SoundError::invalid_argument(
            "source amplitude must be finite and non-negative",
        ));
    }
    Ok(())
}

fn ir_buffer_size(sample_count: u32) -> usize {
    sample_count as usize * size_of::<IrSample>()
}

fn output_channels_to_shader(output_channels: OutputChannels) -> u32 {
    match output_channels {
        OutputChannels::Mono => 0,
        OutputChannels::Stereo => 1,
        OutputChannels::Binaural => 2,
    }
}

#[cfg(test)]
fn power_cosine_pdf(cosine: f32, exponent: f32) -> f32 {
    if !cosine.is_finite() || !exponent.is_finite() || cosine <= 0.0 || exponent < 0.0 {
        return 0.0;
    }
    (exponent + 1.0) * cosine.powf(exponent) / (2.0 * std::f32::consts::PI)
}

#[cfg(test)]
fn pressure_branch_probabilities(energies: &[f32]) -> Vec<f32> {
    let weights = energies.iter().map(|energy| energy.max(0.0).sqrt()).collect::<Vec<_>>();
    let sum = weights.iter().sum::<f32>();
    if sum <= 0.0 || !sum.is_finite() {
        return vec![0.0; energies.len()];
    }
    weights.into_iter().map(|weight| weight / sum).collect()
}

#[cfg(test)]
fn deposit_fractional(samples: &mut [f32], sample_position: f32, amplitude: f32) {
    if !sample_position.is_finite() || !amplitude.is_finite() || sample_position < 0.0 {
        return;
    }
    let base = sample_position.floor() as usize;
    let fraction = sample_position - base as f32;
    if let Some(sample) = samples.get_mut(base) {
        *sample += amplitude * (1.0 - fraction);
    }
    if let Some(sample) = samples.get_mut(base.saturating_add(1)) {
        *sample += amplitude * fraction;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_cosine_pdf_integrates_to_one() {
        let exponent = 32.0;
        let steps = 100_000;
        let integral = (0..steps)
            .map(|index| {
                let cosine = (index as f32 + 0.5) / steps as f32;
                power_cosine_pdf(cosine, exponent) * 2.0 * std::f32::consts::PI / steps as f32
            })
            .sum::<f32>();
        assert!((integral - 1.0).abs() < 0.001);
    }

    #[test]
    fn branch_probabilities_follow_pressure_weights() {
        let probabilities = pressure_branch_probabilities(&[0.81, 0.09, 0.0]);
        assert!((probabilities[0] - 0.75).abs() < 1.0e-6);
        assert!((probabilities[1] - 0.25).abs() < 1.0e-6);
        assert_eq!(probabilities[2], 0.0);
    }

    #[test]
    fn fractional_deposition_preserves_signed_amplitude() {
        let mut samples = [0.0; 4];
        deposit_fractional(&mut samples, 1.25, -2.0);
        assert_eq!(samples, [0.0, -1.5, -0.5, 0.0]);
    }

    #[test]
    fn fractional_deposition_rejects_invalid_state() {
        let mut samples = [0.0; 2];
        deposit_fractional(&mut samples, f32::NAN, 1.0);
        deposit_fractional(&mut samples, 0.0, f32::INFINITY);
        assert_eq!(samples, [0.0, 0.0]);
    }

    #[test]
    fn half_period_two_tap_ir_cancels_a_sine() {
        let period = 32;
        let half_period = period / 2;
        let input = (0..256)
            .map(|index| (std::f32::consts::TAU * index as f32 / period as f32).sin())
            .collect::<Vec<_>>();
        let mut output = vec![0.0; input.len() + half_period];
        for (index, sample) in input.iter().copied().enumerate() {
            output[index] += sample;
            output[index + half_period] += sample;
        }
        let steady_peak = output[half_period..input.len()]
            .iter()
            .fold(0.0_f32, |peak, value| peak.max(value.abs()));
        assert!(steady_peak < 1.0e-5);
    }

    #[test]
    fn push_constants_match_shader_layout() {
        assert_eq!(size_of::<ListenerTracePushConstants>(), 80);
    }

    #[test]
    fn geometry_masks_do_not_overlap() {
        assert_eq!(SURFACE_GEOMETRY_MASK & DIFFRACTION_GEOMETRY_MASK, 0);
        assert_eq!(SURFACE_VISIBILITY_HIT_GROUP_OFFSET, 1);
    }
}
