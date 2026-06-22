use crate::acoustics::{IrSample, IrSnapshot};
use crate::core::config::OutputChannels;
use crate::core::error::{SoundError, SoundResult};
use crate::gpu::backend::VkBackend;
use crate::gpu::memory::BufferHandle;
use crate::gpu::rt::{
    AabbBlasBuildSpec, BlasId, RtAabbBuffers, RtAabbSpec, RtDescriptorBindingSpec, RtDescriptorWrite, RtInstanceSpec,
    RtPipelineId, RtPipelineSpec, RtPushConstantSpec, RtShaderGroupSpec, RtShaderStageSpec, RtTraceSpec, TlasBuildSpec,
};
use crate::gpu::shader::ShaderStage;
use crate::scene::SceneManager;
use ash::vk;
use glam::{Mat4, Vec3};
use std::mem::size_of;
use std::path::PathBuf;

const SHADER_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/acoustics/shaders");
const IR_FIXED_POINT_SCALE: f32 = 1_048_576.0;

/// Runtime configuration for acoustic IR generation.
#[derive(Debug, Clone, Copy)]
pub struct AcousticConfig {
    /// Sample rate used when binning path arrivals into IR samples.
    pub sample_rate: u32,
    /// Number of samples in each generated impulse response.
    pub num_samples: u32,
    /// Output layout used when constructing stereo IR channel gains.
    pub output_channels: OutputChannels,
    /// Listener capture sphere radius in world meters, centered on each query listener position.
    pub listener_radius: f32,
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
    pub source_position: Vec3,
    /// Listener position in world meters.
    pub listener_position: Vec3,
    /// Listener right-ear direction in world space, used for directional stereo binning.
    pub listener_right: Vec3,
    /// Acoustic source energy split across the traced ray directions.
    pub source_energy: f32,
}

/// Acoustic pipeline that traces acoustic paths into stereo IR snapshots.
///
/// The pipeline owns long-lived RT resources and an IR accumulation buffer.
/// Calls are synchronous today: GPU ray tracing writes the IR, the final samples
/// are downloaded, and `build_ir` returns an immutable snapshot.
pub struct AcousticPipeline {
    cfg: AcousticConfig,
    contribution_pipeline: RtPipelineId,
    ir_buffer: BufferHandle,
    /// Procedural BLAS bounds kept alive while the analytical listener sphere is traced.
    _listener_bounds: RtAabbBuffers,
    listener_blas: BlasId,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct VisibilityPushConstants {
    source_position: [f32; 3],
    source_energy: f32,
    listener_position: [f32; 3],
    speed_of_sound: f32,
    listener_radius: f32,
    listener_volume: f32,
    _pad_listener_volume: u32,
    ray_count: u32,
    max_bounces: u32,
    _pad0: [u32; 3],
    listener_right: [f32; 3],
    sample_rate: u32,
    sample_count: u32,
    output_channels: u32,
}

unsafe impl bytemuck::Zeroable for VisibilityPushConstants {}
unsafe impl bytemuck::Pod for VisibilityPushConstants {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GpuIrSample {
    left: i32,
    right: i32,
}

unsafe impl bytemuck::Zeroable for GpuIrSample {}
unsafe impl bytemuck::Pod for GpuIrSample {}

impl AcousticPipeline {
    /// Creates the RT contribution pipeline, GPU contribution buffer, and IR builder.
    pub fn new(gpu: &VkBackend, cfg: AcousticConfig) -> SoundResult<Self> {
        validate_listener_radius(cfg.listener_radius)?;
        validate_ir_dimensions(cfg.sample_rate, cfg.num_samples)?;

        let raygen = gpu
            .shaders()
            .load_glsl_file(ShaderStage::RayGeneration, shader_path("direct_visibility.rgen.glsl"))?;
        let miss = gpu
            .shaders()
            .load_glsl_file(ShaderStage::RayMiss, shader_path("direct_visibility.rmiss.glsl"))?;
        let closest_hit = gpu
            .shaders()
            .load_glsl_file(ShaderStage::RayClosestHit, shader_path("direct_visibility.rchit.glsl"))?;
        let listener_closest_hit = gpu.shaders().load_glsl_file(
            ShaderStage::RayClosestHit,
            shader_path("direct_visibility_listener.rchit.glsl"),
        )?;
        let listener_intersection = gpu.shaders().load_glsl_file(
            ShaderStage::RayIntersection,
            shader_path("direct_visibility_listener.rint.glsl"),
        )?;
        let contribution_pipeline = gpu.rt().create_pipeline(RtPipelineSpec {
            shaders: vec![
                RtShaderStageSpec::new(raygen),
                RtShaderStageSpec::new(miss),
                RtShaderStageSpec::new(closest_hit),
                RtShaderStageSpec::new(listener_closest_hit),
                RtShaderStageSpec::new(listener_intersection),
            ],
            groups: vec![
                RtShaderGroupSpec::Raygen { shader: 0 },
                RtShaderGroupSpec::Miss { shader: 1 },
                RtShaderGroupSpec::TrianglesHit { closest_hit_shader: 2 },
                RtShaderGroupSpec::ProceduralHit {
                    closest_hit_shader: 3,
                    intersection_shader: 4,
                },
            ],
            descriptor_bindings: vec![
                RtDescriptorBindingSpec::acceleration_structure(0),
                RtDescriptorBindingSpec::storage_buffer(1),
                RtDescriptorBindingSpec::storage_buffer(2),
                RtDescriptorBindingSpec::storage_buffer(3),
            ],
            push_constant_ranges: vec![RtPushConstantSpec::new(0, size_of::<VisibilityPushConstants>() as u32)],
            max_ray_recursion_depth: cfg.max_bounces.saturating_add(1).max(1),
        })?;
        let ir_buffer = gpu
            .memory()
            .create_storage_buffer(ir_buffer_size(cfg.num_samples) as u64)?;
        let listener_bounds = create_listener_sphere_bounds(gpu, cfg.listener_radius)?;
        let listener_blas = gpu.rt().build_aabb_blas(AabbBlasBuildSpec {
            aabbs: &listener_bounds,
            flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
        })?;

        Ok(Self {
            cfg,
            contribution_pipeline,
            ir_buffer,
            _listener_bounds: listener_bounds,
            listener_blas,
        })
    }

    /// Builds one stereo IR for a scene/source/listener query.
    pub fn build_ir(
        &mut self,
        gpu: &VkBackend,
        scene: &mut SceneManager,
        query: AcousticQuery,
    ) -> SoundResult<IrSnapshot> {
        let gpu_scene = scene.sync_gpu_if_needed(gpu)?;
        let tlas_id = self.build_query_tlas(gpu, gpu_scene.instances(), query.listener_position)?;

        let _ray_budget = self.cfg.rays_per_query;
        let _max_bounces = self.cfg.max_bounces;
        let visibility_constants = VisibilityPushConstants {
            source_position: query.source_position.to_array(),
            source_energy: query.source_energy,
            listener_position: query.listener_position.to_array(),
            speed_of_sound: SPEED_OF_SOUND_METERS_PER_SECOND,
            listener_radius: self.cfg.listener_radius,
            listener_volume: listener_sphere_volume(self.cfg.listener_radius),
            _pad_listener_volume: 0,
            ray_count: self.cfg.rays_per_query.max(1),
            max_bounces: self.cfg.max_bounces,
            _pad0: [0; 3],
            listener_right: query.listener_right.to_array(),
            sample_rate: self.cfg.sample_rate,
            sample_count: self.cfg.num_samples,
            output_channels: output_channels_to_shader(self.cfg.output_channels),
        };

        gpu.memory().clear_buffer(&self.ir_buffer)?;
        gpu.compute().submit_compute_and_wait(|command_buffer| {
            gpu.rt().record_trace(
                command_buffer,
                &RtTraceSpec {
                    pipeline_id: self.contribution_pipeline,
                    descriptor_writes: vec![
                        RtDescriptorWrite::acceleration_structure(0, tlas_id),
                        RtDescriptorWrite::storage_buffer(1, self.ir_buffer.buffer),
                        RtDescriptorWrite::storage_buffer(2, gpu_scene.object_buffer().buffer),
                        RtDescriptorWrite::storage_buffer(3, gpu_scene.material_buffer().buffer),
                    ],
                    push_constants: bytemuck::bytes_of(&visibility_constants).to_vec(),
                    push_constant_offset: 0,
                    dimensions: [self.cfg.rays_per_query.max(1), 1, 1],
                    barrier_after_trace: true,
                },
            )?;

            Ok(())
        })?;

        self.download_ir(gpu, gpu_scene.version(), query.query_id)
    }

    /// Builds stereo IR snapshots for a batch of source/listener queries.
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

    fn download_ir(
        &self,
        gpu: &VkBackend,
        scene_version: crate::scene::SceneVersion,
        query_id: u32,
    ) -> SoundResult<IrSnapshot> {
        let mut gpu_samples = vec![GpuIrSample { left: 0, right: 0 }; self.cfg.num_samples as usize];
        gpu.memory()
            .download_typed(&self.ir_buffer, gpu_samples.len(), &mut gpu_samples)?;

        let mut energy = 0.0f32;
        let samples = gpu_samples
            .into_iter()
            .map(|sample| {
                let left = sample.left as f32 / IR_FIXED_POINT_SCALE;
                let right = sample.right as f32 / IR_FIXED_POINT_SCALE;
                energy += left * left + right * right;
                IrSample::new(left, right)
            })
            .collect();

        Ok(IrSnapshot {
            sample_rate: self.cfg.sample_rate,
            samples,
            energy,
            scene_version,
            query_id,
        })
    }

    fn build_query_tlas(
        &self,
        gpu: &VkBackend,
        scene_instances: &[RtInstanceSpec],
        listener_position: Vec3,
    ) -> SoundResult<crate::gpu::rt::TlasId> {
        let mut instances = Vec::with_capacity(scene_instances.len() + 1);
        instances.extend_from_slice(scene_instances);
        instances.push(RtInstanceSpec {
            blas_id: self.listener_blas,
            transform: Mat4::from_translation(listener_position),
            custom_index: LISTENER_INSTANCE_CUSTOM_INDEX,
            mask: 0xff,
            sbt_record_offset: LISTENER_HIT_GROUP_OFFSET,
        });

        gpu.rt().build_tlas(TlasBuildSpec {
            instances: &instances,
            flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
        })
    }
}

fn shader_path(file: &str) -> PathBuf {
    PathBuf::from(SHADER_DIR).join(file)
}

const SPEED_OF_SOUND_METERS_PER_SECOND: f32 = 343.0;
const LISTENER_INSTANCE_CUSTOM_INDEX: u32 = u32::MAX;
const LISTENER_HIT_GROUP_OFFSET: u32 = 1;
const SPHERE_VOLUME_FACTOR: f32 = 4.0 * std::f32::consts::PI / 3.0;

fn validate_listener_radius(listener_radius: f32) -> SoundResult<()> {
    if !listener_radius.is_finite() || listener_radius <= 0.0 {
        return Err(SoundError::invalid_argument("listener_radius must be finite and > 0"));
    }
    Ok(())
}

fn validate_ir_dimensions(sample_rate: u32, num_samples: u32) -> SoundResult<()> {
    if sample_rate == 0 {
        return Err(SoundError::invalid_argument("IR sample_rate must be > 0"));
    }
    if num_samples == 0 {
        return Err(SoundError::invalid_argument("IR length must be > 0"));
    }
    Ok(())
}

fn create_listener_sphere_bounds(gpu: &VkBackend, listener_radius: f32) -> SoundResult<RtAabbBuffers> {
    let extent = Vec3::splat(listener_radius);
    gpu.rt().upload_aabbs(&[RtAabbSpec {
        min: -extent,
        max: extent,
        opaque: true,
    }])
}

fn listener_sphere_volume(listener_radius: f32) -> f32 {
    SPHERE_VOLUME_FACTOR * listener_radius * listener_radius * listener_radius
}

fn ir_buffer_size(sample_count: u32) -> usize {
    sample_count as usize * size_of::<GpuIrSample>()
}

fn output_channels_to_shader(output_channels: OutputChannels) -> u32 {
    match output_channels {
        OutputChannels::Mono => 0,
        OutputChannels::Stereo => 1,
        OutputChannels::Binaural => 2,
    }
}
