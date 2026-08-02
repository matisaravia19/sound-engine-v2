use crate::acoustics::{GpuIrSnapshot, IrSample, IrSnapshot};
use crate::core::config::{EngineConfig, OutputChannels};
use crate::core::error::{SoundError, SoundResult};
use crate::gpu::backend::VkBackend;
use crate::gpu::compute::{
    BufferBarrierSpec, ComputePipelineSpec, DescriptorBindingSpec, DescriptorWrite, DispatchSpec, PipelineId,
    PushConstantSpec,
};
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
const BTM_SEGMENTS_PER_EDGE: u32 = 4_096;
const BTM_WORKGROUP_SIZE: u32 = 64;
const SURFACE_CONTINUATION_HIT_GROUP_OFFSET: u32 = 0;
#[cfg(test)]
const SURFACE_VISIBILITY_HIT_GROUP_OFFSET: u32 = 1;

/// Source/listener pair that should produce one impulse response.
#[derive(Debug, Clone, Copy)]
pub struct AcousticQuery {
    /// Application-defined query identifier copied into the returned IR snapshot.
    pub query_id: u32,
    /// Source position in world meters.
    pub source_position: Vec3,
    /// Listener position in world meters.
    pub listener_position: Vec3,
    /// Listener forward direction in world space.
    pub listener_forward: Vec3,
    /// Listener up direction in world space.
    pub listener_up: Vec3,
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
    btm_diffraction_pipeline: PipelineId,
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
    head_radius_meters: f32,
    max_ild_db: f32,
    _pad0: u32,
}

unsafe impl bytemuck::Zeroable for ListenerTracePushConstants {}
unsafe impl bytemuck::Pod for ListenerTracePushConstants {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct BtmDiffractionPushConstants {
    source_position: [f32; 3],
    source_amplitude: f32,
    listener_position: [f32; 3],
    maximum_path_length: f32,
    listener_right: [f32; 3],
    speed_of_sound: f32,
    sample_rate: u32,
    sample_count: u32,
    output_channels: u32,
    edge_count: u32,
    segments_per_edge: u32,
    head_radius_meters: f32,
    max_ild_db: f32,
    _pad0: u32,
}

unsafe impl bytemuck::Zeroable for BtmDiffractionPushConstants {}
unsafe impl bytemuck::Pod for BtmDiffractionPushConstants {}

impl AcousticPipeline {
    /// Creates the listener-originated RT pipeline and the empty-scene sentinel BLAS.
    pub fn new(gpu: &VkBackend, config: EngineConfig) -> SoundResult<Self> {
        validate_config(config)?;

        let raygen = load_shader(gpu, ShaderStage::RayGeneration, "listener_backward.rgen.glsl")?;
        let continuation_miss = load_shader(gpu, ShaderStage::RayMiss, "continuation.rmiss.glsl")?;
        let visibility_miss = load_shader(gpu, ShaderStage::RayMiss, "visibility.rmiss.glsl")?;
        let surface_closest_hit = load_shader(gpu, ShaderStage::RayClosestHit, "listener_backward_surface.rchit.glsl")?;
        let visibility_closest_hit = load_shader(gpu, ShaderStage::RayClosestHit, "visibility.rchit.glsl")?;
        let btm_diffraction = load_shader(gpu, ShaderStage::Compute, "btm_diffraction.comp.glsl")?;

        let listener_pipeline = gpu.rt().create_pipeline(RtPipelineSpec {
            shaders: vec![
                RtShaderStageSpec::new(raygen),
                RtShaderStageSpec::new(continuation_miss),
                RtShaderStageSpec::new(visibility_miss),
                RtShaderStageSpec::new(surface_closest_hit),
                RtShaderStageSpec::new(visibility_closest_hit),
            ],
            // Surface continuation uses record zero and visibility adds one.
            groups: vec![
                RtShaderGroupSpec::Raygen { shader: 0 },
                RtShaderGroupSpec::Miss { shader: 1 },
                RtShaderGroupSpec::Miss { shader: 2 },
                RtShaderGroupSpec::TrianglesHit { closest_hit_shader: 3 },
                RtShaderGroupSpec::TrianglesHit { closest_hit_shader: 4 },
            ],
            // The compact descriptor set contains only traversal state, the
            // final IR, and immutable scene lookup tables. There is no
            // per-ray hit or contribution storage.
            descriptor_bindings: vec![
                RtDescriptorBindingSpec::acceleration_structure(0),
                RtDescriptorBindingSpec::storage_buffer(1),
                RtDescriptorBindingSpec::storage_buffer(2),
                RtDescriptorBindingSpec::storage_buffer(3),
                // Retained while the ray-generation module shares the edge
                // record declaration with legacy dead-code paths.
                RtDescriptorBindingSpec::storage_buffer(4),
            ],
            push_constant_ranges: vec![RtPushConstantSpec::new(
                0,
                size_of::<ListenerTracePushConstants>() as u32,
            )],
            max_ray_recursion_depth: 1,
        })?;
        let btm_diffraction_pipeline = gpu.compute().create_pipeline(ComputePipelineSpec {
            shader_id: btm_diffraction,
            descriptor_bindings: vec![
                DescriptorBindingSpec::storage_buffer(0),
                DescriptorBindingSpec::storage_buffer(1),
            ],
            push_constant_ranges: vec![PushConstantSpec::new(
                0,
                size_of::<BtmDiffractionPushConstants>() as u32,
            )],
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
            btm_diffraction_pipeline,
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
        // Diffraction is integrated deterministically after tracing. Virtual
        // proxy geometry must not alter ordinary surface continuation paths.
        let tlas_id = self.build_query_tlas(gpu, gpu_scene.instances())?;
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
            listener_right: query
                .listener_forward
                .normalize()
                .cross(query.listener_up.normalize())
                .normalize_or_zero()
                .to_array(),
            maximum_path_length: self.config.acoustics.maximum_path_length,
            listener_ray_count: self.config.acoustics.listener_ray_count,
            max_bounces: self.config.acoustics.max_bounces,
            sample_rate: self.config.sound.sample_rate,
            sample_count: self.config.sound.ir_num_samples,
            output_channels: output_channels_to_shader(self.config.sound.output_channels),
            sampling_seed: self.config.acoustics.sampling_seed,
            minimum_throughput: self.config.acoustics.minimum_throughput,
            head_radius_meters: self.config.binaural.head_radius_meters,
            max_ild_db: self.config.binaural.max_ild_db,
            _pad0: 0,
        };
        let edge_count = gpu_scene.diffraction_edge_count();
        let btm_constants = BtmDiffractionPushConstants {
            source_position: query.source_position.to_array(),
            source_amplitude: query.source_amplitude,
            listener_position: query.listener_position.to_array(),
            maximum_path_length: self.config.acoustics.maximum_path_length,
            listener_right: query
                .listener_forward
                .normalize()
                .cross(query.listener_up.normalize())
                .normalize_or_zero()
                .to_array(),
            speed_of_sound: SPEED_OF_SOUND_METERS_PER_SECOND,
            sample_rate: self.config.sound.sample_rate,
            sample_count: self.config.sound.ir_num_samples,
            output_channels: output_channels_to_shader(self.config.sound.output_channels),
            edge_count,
            segments_per_edge: BTM_SEGMENTS_PER_EDGE,
            head_radius_meters: self.config.binaural.head_radius_meters,
            max_ild_db: self.config.binaural.max_ild_db,
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
            if edge_count > 0 {
                gpu.compute().record_buffer_barrier(
                    command_buffer,
                    &BufferBarrierSpec::ray_tracing_shader_write_to_compute_shader_read_write(ir_buffer.buffer),
                );
                gpu.compute().record_dispatch(
                    command_buffer,
                    &DispatchSpec {
                        pipeline_id: self.btm_diffraction_pipeline,
                        descriptor_writes: vec![
                            DescriptorWrite::storage_buffer(0, ir_buffer.buffer),
                            DescriptorWrite::storage_buffer(1, gpu_scene.diffraction_edge_buffer().buffer),
                        ],
                        push_constants: bytemuck::bytes_of(&btm_constants).to_vec(),
                        push_constant_offset: 0,
                        groups: [BTM_SEGMENTS_PER_EDGE.div_ceil(BTM_WORKGROUP_SIZE), edge_count, 1],
                    },
                )?;
                gpu.compute().record_buffer_barrier(
                    command_buffer,
                    &BufferBarrierSpec::compute_shader_write_to_compute_or_transfer_read(ir_buffer.buffer),
                );
            } else {
                // Ray-generation float atomics are shader writes. Make them
                // visible to IR download, caching, or convolution.
                gpu.compute().record_buffer_barrier(
                    command_buffer,
                    &BufferBarrierSpec::ray_tracing_shader_write_to_compute_or_transfer_read(ir_buffer.buffer),
                );
            }
            Ok(())
        })?;
        // The submission above is synchronous, so no descriptor or command
        // buffer can still reference this per-query TLAS. Releasing it here
        // prevents moving sources/listeners from retaining one acceleration
        // structure and backing allocation per acoustic update.
        gpu.rt().destroy_tlas(tlas_id)?;

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
    ) -> SoundResult<crate::gpu::rt::TlasId> {
        let mut instances = Vec::with_capacity(scene_instances.len() + 1);
        instances.extend_from_slice(scene_instances);
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
    if !config.binaural.head_radius_meters.is_finite() || config.binaural.head_radius_meters <= 0.0 {
        return Err(SoundError::invalid_argument(
            "binaural head radius must be finite and positive",
        ));
    }
    if !config.binaural.max_ild_db.is_finite() || config.binaural.max_ild_db < 0.0 {
        return Err(SoundError::invalid_argument(
            "binaural maximum ILD must be finite and non-negative",
        ));
    }
    Ok(())
}

fn validate_query(query: AcousticQuery) -> SoundResult<()> {
    if !query.source_position.is_finite()
        || !query.listener_position.is_finite()
        || !query.listener_forward.is_finite()
        || !query.listener_up.is_finite()
    {
        return Err(SoundError::invalid_argument("acoustic query vectors must be finite"));
    }
    let forward = query.listener_forward.normalize_or_zero();
    let up = query.listener_up.normalize_or_zero();
    if forward == Vec3::ZERO || up == Vec3::ZERO || forward.cross(up).length_squared() <= 1.0e-8 {
        return Err(SoundError::invalid_argument(
            "listener forward and up vectors must be non-zero and non-collinear",
        ));
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
    if !sample_position.is_finite()
        || !amplitude.is_finite()
        || sample_position <= -1.0
        || sample_position >= samples.len() as f32
    {
        return;
    }
    let base = sample_position.floor() as isize;
    let fraction = sample_position - base as f32;
    if base >= 0
        && let Some(sample) = samples.get_mut(base as usize)
    {
        *sample += amplitude * (1.0 - fraction);
    }
    let next = base + 1;
    if next >= 0
        && let Some(sample) = samples.get_mut(next as usize)
    {
        *sample += amplitude * fraction;
    }
}

#[cfg(test)]
fn analytic_binaural_estimate(
    lateral: f32,
    head_radius_meters: f32,
    max_ild_db: f32,
    sample_rate: u32,
) -> ([f32; 2], [f32; 2]) {
    let lateral = lateral.clamp(-1.0, 1.0);
    let absolute_lateral = lateral.abs();
    let itd_seconds = lateral.signum() * head_radius_meters / SPEED_OF_SOUND_METERS_PER_SECOND
        * (absolute_lateral.asin() + absolute_lateral);
    let half_itd_samples = 0.5 * itd_seconds * sample_rate as f32;
    let far_to_near = 10.0_f32.powf(-max_ild_db * absolute_lateral / 20.0);
    let near_factor = (1.0 + far_to_near * far_to_near).sqrt().recip();
    let far_factor = far_to_near * near_factor;
    let gains = if lateral < 0.0 {
        [near_factor, far_factor]
    } else if lateral > 0.0 {
        [far_factor, near_factor]
    } else {
        [near_factor, near_factor]
    };
    ([half_itd_samples, -half_itd_samples], gains)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> EngineConfig {
        EngineConfig {
            sound: crate::core::config::SoundConfig {
                output_channels: OutputChannels::Binaural,
                sample_rate: 48_000,
                ir_num_samples: 16_384,
            },
            acoustics: crate::core::config::AcousticsConfig {
                listener_ray_count: 1,
                max_bounces: 0,
                maximum_path_length: 343.0,
                minimum_throughput: 0.0,
                sampling_seed: 0,
            },
            cache: crate::core::config::IrCacheConfig::default(),
            binaural: crate::core::config::BinauralConfig::default(),
            auralization: crate::core::config::AuralizationConfig { block_size: 1_024 },
        }
    }

    fn test_query() -> AcousticQuery {
        AcousticQuery {
            query_id: 1,
            source_position: Vec3::Z,
            listener_position: Vec3::ZERO,
            listener_forward: -Vec3::Z,
            listener_up: Vec3::Y,
            source_amplitude: 1.0,
        }
    }

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
        assert_eq!(size_of::<ListenerTracePushConstants>(), 88);
        assert_eq!(std::mem::offset_of!(ListenerTracePushConstants, minimum_throughput), 72);
        assert_eq!(std::mem::offset_of!(ListenerTracePushConstants, head_radius_meters), 76);
        assert_eq!(std::mem::offset_of!(ListenerTracePushConstants, max_ild_db), 80);
    }

    #[test]
    fn centered_binaural_arrivals_have_equal_levels_and_no_itd() {
        let right = Vec3::X;
        for direction in [-Vec3::Z, Vec3::Z, Vec3::Y, Vec3::ZERO] {
            let lateral = direction.normalize_or_zero().dot(right);
            let (offsets, gains) = analytic_binaural_estimate(lateral, 0.0875, 12.0, 48_000);
            assert_eq!(offsets, [0.0, 0.0]);
            assert!((gains[0] - gains[1]).abs() < 1.0e-7);
        }
    }

    #[test]
    fn left_and_right_binaural_arrivals_are_mirrored() {
        let (left_offsets, left_gains) = analytic_binaural_estimate(-0.75, 0.0875, 12.0, 48_000);
        let (right_offsets, right_gains) = analytic_binaural_estimate(0.75, 0.0875, 12.0, 48_000);
        assert_eq!(left_offsets, [right_offsets[1], right_offsets[0]]);
        assert_eq!(left_gains, [right_gains[1], right_gains[0]]);
    }

    #[test]
    fn maximum_binaural_itd_matches_spherical_head_formula() {
        let radius = 0.0875;
        let sample_rate = 48_000;
        let (offsets, _) = analytic_binaural_estimate(1.0, radius, 12.0, sample_rate);
        let expected =
            radius / SPEED_OF_SOUND_METERS_PER_SECOND * (std::f32::consts::FRAC_PI_2 + 1.0) * sample_rate as f32;
        assert!(((offsets[0] - offsets[1]) - expected).abs() < 1.0e-5);
    }

    #[test]
    fn binaural_ild_gains_preserve_unit_energy() {
        for lateral in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            let (_, gains) = analytic_binaural_estimate(lateral, 0.0875, 12.0, 48_000);
            assert!((gains[0] * gains[0] + gains[1] * gains[1] - 1.0).abs() < 1.0e-6);
        }
    }

    #[test]
    fn binaural_fractional_deposition_preserves_each_ear_pressure() {
        let (offsets, gains) = analytic_binaural_estimate(0.6, 0.0875, 12.0, 48_000);
        let center = 10.25;
        let amplitude = -2.0;
        let mut left = [0.0; 32];
        let mut right = [0.0; 32];
        deposit_fractional(&mut left, center + offsets[0], amplitude * gains[0]);
        deposit_fractional(&mut right, center + offsets[1], amplitude * gains[1]);
        assert!((left.iter().sum::<f32>() - amplitude * gains[0]).abs() < 1.0e-6);
        assert!((right.iter().sum::<f32>() - amplitude * gains[1]).abs() < 1.0e-6);
    }

    #[test]
    fn fractional_deposition_clips_only_the_out_of_bounds_portion() {
        let mut samples = [0.0; 2];
        deposit_fractional(&mut samples, -0.25, -2.0);
        assert_eq!(samples, [-1.5, 0.0]);

        samples.fill(0.0);
        deposit_fractional(&mut samples, 1.25, 2.0);
        assert_eq!(samples, [0.0, 1.5]);
    }

    #[test]
    fn rejects_invalid_binaural_configuration() {
        let mut config = test_config();
        config.binaural.head_radius_meters = 0.0;
        assert!(validate_config(config).is_err());

        config = test_config();
        config.binaural.head_radius_meters = f32::NAN;
        assert!(validate_config(config).is_err());

        config = test_config();
        config.binaural.max_ild_db = -1.0;
        assert!(validate_config(config).is_err());

        config = test_config();
        config.binaural.max_ild_db = f32::INFINITY;
        assert!(validate_config(config).is_err());
    }

    #[test]
    fn rejects_invalid_acoustic_query_orientation() {
        let mut query = test_query();
        query.listener_forward = Vec3::ZERO;
        assert!(validate_query(query).is_err());

        query = test_query();
        query.listener_up = query.listener_forward;
        assert!(validate_query(query).is_err());

        query = test_query();
        query.listener_up = Vec3::splat(f32::NAN);
        assert!(validate_query(query).is_err());
    }

    #[test]
    fn visibility_hit_group_follows_surface_continuation() {
        assert_eq!(SURFACE_VISIBILITY_HIT_GROUP_OFFSET, 1);
    }
}
