use super::spatial::ValidatedAcousticQuery;
use crate::acoustics::{
    AcousticPipelineConfig, AcousticQuery, GpuIrSnapshot, IrSample, IrSnapshot,
};
use crate::core::config::OutputChannels;
use crate::core::error::SoundResult;
use crate::gpu::backend::VkBackend;
use crate::gpu::compute::BufferBarrierSpec;
use crate::gpu::rt::{
    OwnedRtPipeline, RtDescriptorBindingSpec, RtDescriptorWrite, RtPipelineSpec,
    RtPushConstantSpec, RtShaderGroupSpec, RtTraceSpec,
};
use crate::gpu::shader::ShaderStage;
use crate::scene::SceneManager;
use std::mem::size_of;
use std::path::PathBuf;
use std::sync::Arc;

const SHADER_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/acoustics/shaders");
const SPEED_OF_SOUND_METERS_PER_SECOND: f32 = 343.0;

/// Listener-originated acoustic path tracer that deposits pressure paths directly into a GPU IR.
///
/// The ray-generation invocation owns all path state and performs sequential
/// continuation and source-visibility traces. The RT pipeline has recursion
/// depth one because hit shaders only report compact traversal results.
/// Scenes own their traversal structures, so queries can share this pipeline
/// without retaining or rebuilding another scene's geometry.
pub struct AcousticPipeline {
    config: AcousticPipelineConfig,
    listener_pipeline: OwnedRtPipeline,
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
}

unsafe impl bytemuck::Zeroable for ListenerTracePushConstants {}
unsafe impl bytemuck::Pod for ListenerTracePushConstants {}

impl AcousticPipeline {
    /// Rejects a foreign backend even when a caller can satisfy its query from a cache.
    pub(crate) fn validate_backend(&self, gpu: &VkBackend) -> SoundResult<()> {
        self.listener_pipeline.validate_context(gpu.rt())
    }

    /// Creates the listener-originated RT pipeline; each scene owns its traversal geometry.
    pub fn new(gpu: &VkBackend, config: AcousticPipelineConfig) -> SoundResult<Self> {
        config.validate()?;

        let raygen = load_shader(
            gpu,
            ShaderStage::RayGeneration,
            "listener_backward.rgen.glsl",
        )?;
        let continuation_miss = load_shader(gpu, ShaderStage::RayMiss, "continuation.rmiss.glsl")?;
        let visibility_miss = load_shader(gpu, ShaderStage::RayMiss, "visibility.rmiss.glsl")?;
        let surface_closest_hit = load_shader(
            gpu,
            ShaderStage::RayClosestHit,
            "listener_backward_surface.rchit.glsl",
        )?;
        let visibility_closest_hit =
            load_shader(gpu, ShaderStage::RayClosestHit, "visibility.rchit.glsl")?;
        let diffraction_closest_hit = load_shader(
            gpu,
            ShaderStage::RayClosestHit,
            "listener_backward_diffraction.rchit.glsl",
        )?;
        let diffraction_intersection =
            load_shader(gpu, ShaderStage::RayIntersection, "diffraction.rint.glsl")?;

        let listener_pipeline = gpu.rt().create_pipeline(RtPipelineSpec {
            shaders: vec![
                raygen,
                continuation_miss,
                visibility_miss,
                surface_closest_hit,
                visibility_closest_hit,
                diffraction_closest_hit,
                diffraction_intersection,
            ],
            // Surface continuation uses record zero and visibility adds one.
            groups: vec![
                RtShaderGroupSpec::Raygen {
                    shader: 0,
                },
                RtShaderGroupSpec::Miss {
                    shader: 1,
                },
                RtShaderGroupSpec::Miss {
                    shader: 2,
                },
                RtShaderGroupSpec::TrianglesHit {
                    closest_hit_shader: 3,
                },
                RtShaderGroupSpec::TrianglesHit {
                    closest_hit_shader: 4,
                },
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
        let listener_pipeline = gpu.rt_context().own_pipeline(listener_pipeline);
        Ok(Self {
            config,
            listener_pipeline,
        })
    }

    /// Builds one CPU-resident impulse response for a source/listener query.
    pub fn build_ir(
        &self,
        gpu: &VkBackend,
        scene: &mut SceneManager,
        query: AcousticQuery,
    ) -> SoundResult<IrSnapshot> {
        let gpu_ir = self.build_ir_gpu(gpu, scene, query)?;
        gpu_ir.download(gpu)
    }

    /// Builds one GPU-resident time-domain pressure IR.
    pub fn build_ir_gpu(
        &self,
        gpu: &VkBackend,
        scene: &mut SceneManager,
        query: AcousticQuery,
    ) -> SoundResult<GpuIrSnapshot> {
        self.build_validated_ir_gpu(gpu, scene, query.canonicalized()?)
    }

    /// Traces a query already validated for cache lookup, preserving its exact basis.
    pub(super) fn build_validated_ir_gpu(
        &self,
        gpu: &VkBackend,
        scene: &mut SceneManager,
        validated_query: ValidatedAcousticQuery,
    ) -> SoundResult<GpuIrSnapshot> {
        self.validate_backend(gpu)?;
        let query = validated_query.as_query();
        let ray_count = if self.config.tracing.max_bounces == 0 || query.source.amplitude == 0.0 {
            1
        } else {
            self.config.tracing.listener_ray_count
        };

        let profile = std::env::var_os("SOUND_ENGINE_TRACE_PROFILE").is_some();
        let started = std::time::Instant::now();
        let gpu_scene = scene.sync_gpu_if_needed(gpu)?;
        let scene_ms = started.elapsed().as_secs_f64() * 1000.0;
        let storage_count = self.config.sound.ir_num_samples;
        let bytes = u64::from(storage_count) * size_of::<IrSample>() as u64;
        let ir_buffer = Arc::new(gpu.memory().create_storage_buffer(bytes)?);

        let allocation_ms = started.elapsed().as_secs_f64() * 1000.0;

        let constants = ListenerTracePushConstants {
            source_position: query.source.position.to_array(),
            source_amplitude: query.source.amplitude,
            listener_position: query.listener.position.to_array(),
            speed_of_sound: SPEED_OF_SOUND_METERS_PER_SECOND,
            listener_right: query.listener.right().to_array(),
            maximum_path_length: self.config.tracing.maximum_path_length,
            listener_ray_count: self.config.tracing.listener_ray_count,
            max_bounces: self.config.tracing.max_bounces,
            sample_rate: self.config.sound.sample_rate,
            sample_count: self.config.sound.ir_num_samples,
            output_channels: output_channels_to_shader(self.config.sound.output_channels),
            sampling_seed: self.config.tracing.sampling_seed,
            minimum_throughput: self.config.tracing.minimum_throughput,
            head_radius_meters: self.config.binaural.head_radius_meters,
            max_ild_db: self.config.binaural.max_ild_db,
        };
        let mut record_ms = 0.0;
        // Scene traversal data, pipeline and IR storage are retained until the wait below.
        let submission = unsafe {
            gpu.compute().submit_compute(|command_buffer| {
                let record_started = profile.then(std::time::Instant::now);
                gpu.memory()
                    .record_clear_buffer(command_buffer, &ir_buffer)?;
                // IR accumulation uses atomic read/modify/write after the transfer fill.
                gpu.compute().record_buffer_barrier(
                    command_buffer,
                    &BufferBarrierSpec::transfer_write_to_ray_tracing_shader_read_write(
                        ir_buffer.buffer,
                    ),
                );
                gpu.rt().record_trace(
                    command_buffer,
                    &RtTraceSpec {
                        pipeline_id: self.listener_pipeline.id,
                        descriptor_writes: vec![
                            RtDescriptorWrite::acceleration_structure(0, gpu_scene.tlas_id()),
                            RtDescriptorWrite::storage_buffer(1, ir_buffer.buffer),
                            RtDescriptorWrite::storage_buffer(2, gpu_scene.object_buffer().buffer),
                            RtDescriptorWrite::storage_buffer(
                                3,
                                gpu_scene.material_buffer().buffer,
                            ),
                            RtDescriptorWrite::storage_buffer(
                                4,
                                gpu_scene.diffraction_edge_buffer().buffer,
                            ),
                        ],
                        push_constants: bytemuck::bytes_of(&constants).to_vec(),
                        push_constant_offset: 0,
                        // Invocation zero owns direct visibility. No other lane
                        // contributes when vertex paths are disabled or silent.
                        dimensions: [ray_count, 1, 1],
                    },
                )?;
                // Ray-generation float atomics are shader writes. Make them visible
                // to IR download, caching, or convolution.
                gpu.compute().record_buffer_barrier(
                    command_buffer,
                    &BufferBarrierSpec::ray_tracing_shader_write_to_compute_or_transfer_read(
                        ir_buffer.buffer,
                    ),
                );
                if let Some(record_started) = record_started {
                    record_ms = record_started.elapsed().as_secs_f64() * 1000.0;
                }
                Ok(())
            })?
        };
        let submitted_ms = if profile {
            started.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
        };
        submission.wait()?;
        let trace_ms = started.elapsed().as_secs_f64() * 1000.0;
        if profile {
            eprintln!(
                "Trace stages: scene={scene_ms:.3}, allocation={:.3}, rays={:.3}, record={record_ms:.3}, submit={:.3}, wait={:.3}, count={ray_count}, query={}",
                allocation_ms - scene_ms,
                trace_ms - allocation_ms,
                submitted_ms - allocation_ms - record_ms,
                trace_ms - submitted_ms,
                query.query_id,
            );
        }

        GpuIrSnapshot::from_validated_query(
            ir_buffer,
            self.config.sound.sample_rate,
            storage_count,
            gpu_scene.version(),
            validated_query,
        )
    }
}

fn load_shader(
    gpu: &VkBackend,
    stage: ShaderStage,
    file: &str,
) -> SoundResult<crate::gpu::shader::ShaderId> {
    gpu.shaders().load_glsl_file(stage, shader_path(file))
}

fn shader_path(file: &str) -> PathBuf {
    PathBuf::from(SHADER_DIR).join(file)
}

fn output_channels_to_shader(output_channels: OutputChannels) -> u32 {
    match output_channels {
        OutputChannels::Mono => 0,
        OutputChannels::Stereo => 1,
        OutputChannels::Binaural => 2,
    }
}

#[cfg(test)]
#[path = "../../tests/unit/acoustics/pipeline.rs"]
mod tests;
