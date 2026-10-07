use crate::acoustics::{GpuIrSnapshot, IrSample, IrSnapshot};
use crate::core::config::{EngineConfig, OutputChannels};
use crate::core::error::{SoundError, SoundResult};
use crate::gpu::backend::VkBackend;
use crate::gpu::compute::BufferBarrierSpec;
use crate::gpu::rt::{
    BlasBuildSpec, BlasId, OwnedBlas, OwnedRtPipeline, OwnedTlas, RtDescriptorBindingSpec,
    RtDescriptorWrite, RtInstanceSpec, RtMeshBuffers, RtMeshSpec, RtPipelineSpec,
    RtPushConstantSpec, RtShaderGroupSpec, RtShaderStageSpec, RtTraceSpec, TlasBuildSpec, TlasId,
};
use crate::gpu::shader::ShaderStage;
use crate::scene::{GpuSceneResources, SceneManager};
use ash::vk;
use glam::{Mat4, Vec3};
use std::mem::size_of;
use std::path::PathBuf;
use std::sync::Arc;

const SHADER_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/acoustics/shaders");
const SPEED_OF_SOUND_METERS_PER_SECOND: f32 = 343.0;
const DIFFRACTION_GEOMETRY_MASK: u8 = 0x02;
const DIFFRACTION_INSTANCE_CUSTOM_INDEX: u32 = 0x00ff_fffe;
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
    listener_pipeline: OwnedRtPipeline,
    // One query TLAS for the latest geometry; its registry record retains every BLAS.
    query_tlas: Option<CachedQueryTlas>,
    /// Triangle storage kept alive for the mask-zero empty-scene sentinel BLAS.
    _empty_scene_mesh: RtMeshBuffers,
    empty_scene_blas: OwnedBlas,
}

/// Stable AS IDs distinguish separate scene managers even at the same scene version.
struct CachedQueryTlas {
    scene_tlas: Option<TlasId>,
    diffraction_blas: Option<BlasId>,
    tlas: OwnedTlas,
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
        gpu.memory()
            .validate_owner(&self._empty_scene_mesh.vertex_buffer)
    }

    /// Creates the listener-originated RT pipeline and the empty-scene sentinel BLAS.
    pub fn new(gpu: &VkBackend, config: EngineConfig) -> SoundResult<Self> {
        validate_config(config)?;

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
                RtShaderStageSpec::new(raygen),
                RtShaderStageSpec::new(continuation_miss),
                RtShaderStageSpec::new(visibility_miss),
                RtShaderStageSpec::new(surface_closest_hit),
                RtShaderStageSpec::new(visibility_closest_hit),
                RtShaderStageSpec::new(diffraction_closest_hit),
                RtShaderStageSpec::new(diffraction_intersection),
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
        // upload_mesh validates the immutable triangle and supplies AS input usage.
        let empty_scene_blas = unsafe {
            gpu.rt().build_blas(BlasBuildSpec {
                mesh: &empty_scene_mesh,
                flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
            })?
        };
        Ok(Self {
            config,
            listener_pipeline,
            query_tlas: None,
            _empty_scene_mesh: empty_scene_mesh,
            empty_scene_blas: gpu.rt_context().own_blas(empty_scene_blas),
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
        self.validate_backend(gpu)?;
        validate_query(query)?;
        let ray_count = if self.config.acoustics.max_bounces == 0 || query.source_amplitude == 0.0 {
            1
        } else {
            self.config.acoustics.listener_ray_count
        };

        let profile = std::env::var_os("SOUND_ENGINE_TRACE_PROFILE").is_some();
        let started = std::time::Instant::now();
        let gpu_scene = scene.sync_gpu_if_needed(gpu)?;
        let tlas_id = self.query_tlas(gpu, gpu_scene)?;
        let scene_ms = started.elapsed().as_secs_f64() * 1000.0;
        let storage_count = self.config.sound.ir_num_samples;
        let bytes = ir_buffer_size(storage_count) as u64;
        let ir_buffer = Arc::new(gpu.memory().create_storage_buffer(bytes)?);

        let allocation_ms = started.elapsed().as_secs_f64() * 1000.0;

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
        };
        let mut record_ms = 0.0;
        // Scene, query TLAS, pipeline and IR storage are retained until the wait below.
        let token = unsafe {
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
                            RtDescriptorWrite::acceleration_structure(0, tlas_id),
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
        gpu.compute().wait_for(token)?;
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

        Ok(GpuIrSnapshot {
            buffer: Some(ir_buffer),
            byte_size: ir_buffer_size(storage_count),
            sample_rate: self.config.sound.sample_rate,
            sample_count: storage_count,
            energy: None,
            scene_version: gpu_scene.version(),
            query_id: query.query_id,
        })
    }

    /// Downloads a GPU-resident pressure IR into a CPU snapshot.
    pub fn download_gpu_ir(
        &self,
        gpu: &VkBackend,
        snapshot: &GpuIrSnapshot,
    ) -> SoundResult<IrSnapshot> {
        let ir_buffer = snapshot
            .buffer
            .as_ref()
            .ok_or_else(|| SoundError::invalid_argument("GPU IR snapshot has no buffer"))?;
        if snapshot.sample_count as u64 * size_of::<IrSample>() as u64 > ir_buffer.size {
            return Err(SoundError::invalid_argument(
                "IR sample count exceeds its GPU buffer",
            ));
        }
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

    fn query_tlas(&mut self, gpu: &VkBackend, scene: &GpuSceneResources) -> SoundResult<TlasId> {
        let scene_tlas = scene.tlas_id();
        let diffraction_blas = scene.diffraction_blas_id();
        if let Some(cached) = &self.query_tlas
            && cached.scene_tlas == scene_tlas
            && cached.diffraction_blas == diffraction_blas
        {
            return Ok(cached.tlas.id);
        }
        let id = self.build_query_tlas(gpu, scene.instances(), diffraction_blas)?;
        self.query_tlas = Some(CachedQueryTlas {
            scene_tlas,
            diffraction_blas,
            tlas: gpu.rt_context().own_tlas(id),
        });
        Ok(id)
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
            // The separate mask keeps virtual strips out of visibility rays;
            // SBT offset two selects the procedural intersection/hit pair.
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
                blas_id: self.empty_scene_blas.id,
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

fn validate_config(config: EngineConfig) -> SoundResult<()> {
    if config.sound.sample_rate == 0 || config.sound.ir_num_samples == 0 {
        return Err(SoundError::invalid_argument(
            "IR sample rate and sample count must be non-zero",
        ));
    }
    if config.acoustics.listener_ray_count == 0 {
        return Err(SoundError::invalid_argument(
            "listener_ray_count must be non-zero",
        ));
    }
    if !config.acoustics.maximum_path_length.is_finite()
        || config.acoustics.maximum_path_length <= 0.0
    {
        return Err(SoundError::invalid_argument(
            "maximum_path_length must be finite and positive",
        ));
    }
    if !config.acoustics.minimum_throughput.is_finite() || config.acoustics.minimum_throughput < 0.0
    {
        return Err(SoundError::invalid_argument(
            "minimum_throughput must be finite and non-negative",
        ));
    }
    if !config.binaural.head_radius_meters.is_finite() || config.binaural.head_radius_meters <= 0.0
    {
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
        return Err(SoundError::invalid_argument(
            "acoustic query vectors must be finite",
        ));
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
#[path = "pipeline_tests.rs"]
mod tests;
