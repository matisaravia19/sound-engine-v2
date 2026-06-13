use crate::acoustics::{ContributionRecord, IrBuilder, IrConfig, IrSnapshot, MAX_CONTRIBUTIONS_PER_QUERY};
use crate::error::{SoundError, SoundResult};
use crate::gpu::backend::VkBackend;
use crate::gpu::compute::BufferBarrierSpec;
use crate::gpu::memory::BufferHandle;
use crate::gpu::rt::{
    RtDescriptorBindingSpec, RtDescriptorWrite, RtPipelineId, RtPipelineSpec, RtPushConstantSpec, RtShaderGroupSpec,
    RtShaderStageSpec, RtTraceSpec,
};
use crate::gpu::shader::ShaderStage;
use crate::scene::SceneManager;
use ash::vk;
use std::mem::size_of;
use std::path::PathBuf;

const SHADER_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/acoustics/shaders");

/// Runtime configuration for acoustic IR generation.
#[derive(Debug, Clone, Copy)]
pub struct AcousticConfig {
    /// Sample rate used when binning path arrivals into IR samples.
    pub sample_rate: u32,
    /// Number of mono samples in each generated impulse response.
    pub ir_len_samples: u32,
    /// Ray budget reserved for future stochastic reflection tracing.
    pub rays_per_query: u32,
    /// Maximum path depth reserved for future reflection tracing.
    pub max_bounces: u32,
}

/// Source/listener pair that should produce one impulse response.
#[derive(Debug, Clone, Copy)]
pub struct AcousticQuery {
    /// Application-defined query identifier copied into the returned IR snapshot.
    pub query_id: u32,
    /// Source position in world meters.
    pub source_position: [f32; 3],
    /// Listener position in world meters.
    pub listener_position: [f32; 3],
    /// Linear source gain applied before distance attenuation.
    pub gain: f32,
}

/// Acoustic pipeline that traces path contributions and builds mono IR snapshots.
///
/// The pipeline owns long-lived RT resources and a contribution buffer. Calls
/// are synchronous today: GPU ray tracing finishes, contributions are downloaded,
/// and CPU IR construction completes before `build_ir` returns.
pub struct AcousticPipeline {
    cfg: AcousticConfig,
    contribution_pipeline: RtPipelineId,
    contribution_buffer: BufferHandle,
    ir_builder: IrBuilder,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct VisibilityPushConstants {
    source: [f32; 4],
    listener: [f32; 4],
}

unsafe impl bytemuck::Zeroable for VisibilityPushConstants {}
unsafe impl bytemuck::Pod for VisibilityPushConstants {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ContributionHeader {
    count: u32,
    _pad: [u32; 3],
}

unsafe impl bytemuck::Zeroable for ContributionHeader {}
unsafe impl bytemuck::Pod for ContributionHeader {}

impl AcousticPipeline {
    /// Creates the RT contribution pipeline, GPU contribution buffer, and IR builder.
    pub fn new(gpu: &VkBackend, cfg: AcousticConfig) -> SoundResult<Self> {
        let raygen = gpu
            .shaders()
            .load_glsl_file(ShaderStage::RayGeneration, shader_path("direct_visibility.rgen.glsl"))?;
        let miss = gpu
            .shaders()
            .load_glsl_file(ShaderStage::RayMiss, shader_path("direct_visibility.rmiss.glsl"))?;
        let closest_hit = gpu
            .shaders()
            .load_glsl_file(ShaderStage::RayClosestHit, shader_path("direct_visibility.rchit.glsl"))?;
        let contribution_pipeline = gpu.rt().create_pipeline(
            gpu.memory(),
            RtPipelineSpec {
                shaders: vec![
                    RtShaderStageSpec::new(raygen),
                    RtShaderStageSpec::new(miss),
                    RtShaderStageSpec::new(closest_hit),
                ],
                groups: vec![
                    RtShaderGroupSpec::Raygen { shader: 0 },
                    RtShaderGroupSpec::Miss { shader: 1 },
                    RtShaderGroupSpec::TrianglesHit { closest_hit_shader: 2 },
                ],
                descriptor_bindings: vec![
                    RtDescriptorBindingSpec::acceleration_structure(0),
                    RtDescriptorBindingSpec::StorageBuffer {
                        binding: 1,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::RAYGEN_KHR,
                    },
                ],
                push_constant_ranges: vec![RtPushConstantSpec::new(0, size_of::<VisibilityPushConstants>() as u32)],
                max_ray_recursion_depth: 1,
            },
        )?;
        let contribution_buffer = gpu.memory().create_storage_buffer(contribution_buffer_size() as u64)?;
        let ir_builder = IrBuilder::new(IrConfig {
            sample_rate: cfg.sample_rate,
            ir_len_samples: cfg.ir_len_samples,
        })?;

        Ok(Self {
            cfg,
            contribution_pipeline,
            contribution_buffer,
            ir_builder,
        })
    }

    /// Builds one mono IR for a scene/source/listener query.
    pub fn build_ir(
        &mut self,
        gpu: &VkBackend,
        scene: &mut SceneManager,
        query: AcousticQuery,
    ) -> SoundResult<IrSnapshot> {
        let gpu_scene = scene.sync_gpu_if_needed(gpu)?;
        let tlas_id = gpu_scene.tlas_id().ok_or_else(|| {
            SoundError::invalid_state("Cannot run acoustic IR generation without at least one active scene object")
        })?;

        let _ray_budget = self.cfg.rays_per_query;
        let _max_bounces = self.cfg.max_bounces;
        let visibility_constants = VisibilityPushConstants {
            source: [
                query.source_position[0],
                query.source_position[1],
                query.source_position[2],
                query.gain,
            ],
            listener: [
                query.listener_position[0],
                query.listener_position[1],
                query.listener_position[2],
                SPEED_OF_SOUND_METERS_PER_SECOND,
            ],
        };

        gpu.compute().submit_compute_and_wait(|command_buffer| {
            gpu.rt().record_trace(
                command_buffer,
                &RtTraceSpec {
                    pipeline_id: self.contribution_pipeline,
                    descriptor_writes: vec![
                        RtDescriptorWrite::acceleration_structure(0, tlas_id),
                        RtDescriptorWrite::storage_buffer(1, self.contribution_buffer.buffer),
                    ],
                    push_constants: bytemuck::bytes_of(&visibility_constants).to_vec(),
                    push_constant_offset: 0,
                    dimensions: [1, 1, 1],
                    barrier_after_trace: false,
                },
            )?;

            Ok(())
        })?;

        let contributions = self.download_contributions(gpu)?;
        Ok(self
            .ir_builder
            .build_from_contributions(gpu_scene.version(), query.query_id, &contributions))
    }

    /// Builds mono IR snapshots for a batch of source/listener queries.
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
}

fn shader_path(file: &str) -> PathBuf {
    PathBuf::from(SHADER_DIR).join(file)
}

const SPEED_OF_SOUND_METERS_PER_SECOND: f32 = 343.0;

impl AcousticPipeline {
    fn download_contributions(&self, gpu: &VkBackend) -> SoundResult<Vec<ContributionRecord>> {
        let mut raw = vec![0_u8; contribution_buffer_size()];
        gpu.memory()
            .download_bytes(&self.contribution_buffer, raw.len(), &mut raw)?;

        let header_size = size_of::<ContributionHeader>();
        let header = bytemuck::from_bytes::<ContributionHeader>(&raw[..header_size]);
        let count = (header.count as usize).min(MAX_CONTRIBUTIONS_PER_QUERY);
        let records_bytes =
            &raw[header_size..header_size + MAX_CONTRIBUTIONS_PER_QUERY * size_of::<ContributionRecord>()];
        let records = bytemuck::cast_slice::<u8, ContributionRecord>(records_bytes);
        Ok(records[..count].to_vec())
    }
}

fn contribution_buffer_size() -> usize {
    size_of::<ContributionHeader>() + MAX_CONTRIBUTIONS_PER_QUERY * size_of::<ContributionRecord>()
}
