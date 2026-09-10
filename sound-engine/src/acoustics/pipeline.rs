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
    /// Listener forward direction in world space.
    pub listener_forward: Vec3,
    /// Listener up direction in world space.
    pub listener_up: Vec3,
    /// Reference source pressure amplitude.
    pub source_amplitude: f32,
    /// Engine-local source radiation profile and physical source orientation.
    pub source_radiation: super::SourceRadiation,
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
    cone_width: f32,
    maximum_frequency_hz: f32,
    /// Stores broadband surface pressure in a separate time-domain suffix.
    time_domain_surfaces: bool,
    /// Number of complex pressure samples carried by each band-model ray.
    band_count: u32,
    /// Reusable band accumulation storage, allocated during setup. Queries are
    /// synchronous; reuse is allowed only when no raw snapshot retains it.
    band_accumulator: Option<Arc<crate::gpu::memory::BufferHandle>>,
    /// Enables the disjoint source-to-surface-to-edge family in the two-edge shader.
    source_surface_paths: bool,
    /// Enables the paired one-surface/two-edge research path families.
    surface_two_edge_paths: bool,
    /// Immutable engine-owned source tables; index zero is the unity sentinel.
    source_profiles: Vec<Arc<crate::gpu::memory::BufferHandle>>,
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
    cone_width: f32,
    maximum_frequency_hz: f32,
    band_count: u32,
    /// Global launch offset keeps directions, seeds and normalization invariant
    /// when one query is traced in disjoint batches across audio blocks.
    ray_offset: u32,
    /// Bit 0 enables source-surface-edge; bit 1 enables both surface/two-edge orders.
    source_surface_paths: u32,
    /// Uniform branch selecting the immutable directional source table.
    source_profile_enabled: u32,
    /// Explicit padding keeps the quaternion at the std430 vec4 offset.
    source_pad: u32,
    /// Unit quaternion rotating world directions into the source frame.
    source_inverse_rotation: [f32; 4],
}

unsafe impl bytemuck::Zeroable for ListenerTracePushConstants {}
unsafe impl bytemuck::Pod for ListenerTracePushConstants {}

impl AcousticPipeline {
    /// Creates the listener-originated RT pipeline and the empty-scene sentinel BLAS.
    pub fn new(gpu: &VkBackend, config: EngineConfig) -> SoundResult<Self> {
        Self::with_shader(gpu, config, "listener_backward.rgen.glsl", 0.0)
    }

    fn with_shader(gpu: &VkBackend, config: EngineConfig, shader: &str, cone_width: f32) -> SoundResult<Self> {
        validate_config(config)?;

        let raygen = load_shader(gpu, ShaderStage::RayGeneration, shader)?;
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
            // Surface continuation uses record zero and visibility adds one.
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
                RtDescriptorBindingSpec::storage_buffer(5),
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
        let unity_profile = Arc::new(gpu.memory().create_storage_buffer(16)?);
        gpu.memory().upload_typed(&unity_profile, &[0u32; 4])?;
        Ok(Self {
            config,
            listener_pipeline,
            _empty_scene_mesh: empty_scene_mesh,
            empty_scene_blas,
            cone_width,
            maximum_frequency_hz: 0.0,
            time_domain_surfaces: false,
            band_count: 0,
            band_accumulator: None,
            source_surface_paths: false,
            surface_two_edge_paths: false,
            source_profiles: vec![unity_profile],
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
        self.build_ir_range_gpu(gpu, scene, query, 0, self.ray_count(query), None)
    }

    /// Effective launch count, including the sole direct-visibility invocation.
    fn ray_count(&self, query: AcousticQuery) -> u32 {
        if self.config.acoustics.max_bounces == 0 || query.source_amplitude == 0.0 {
            1
        } else {
            self.config.acoustics.listener_ray_count
        }
    }

    /// Accumulates a disjoint range of a fixed query. The previous pressure is
    /// exclusively retained by the caller; every submission completes here.
    fn build_ir_range_gpu(
        &mut self,
        gpu: &VkBackend,
        scene: &mut SceneManager,
        query: AcousticQuery,
        ray_offset: u32,
        ray_count: u32,
        previous: Option<GpuIrSnapshot>,
    ) -> SoundResult<GpuIrSnapshot> {
        validate_query(query)?;
        let source_profile = self
            .source_profiles
            .get(query.source_radiation.profile as usize)
            .ok_or_else(|| SoundError::invalid_argument("source directivity is not registered in this engine"))?
            .clone();
        if ray_count == 0
            || ray_offset
                .checked_add(ray_count)
                .is_none_or(|end| end > self.ray_count(query))
        {
            return Err(SoundError::invalid_argument("invalid acoustic ray range"));
        }
        let profile = std::env::var_os("SOUND_ENGINE_TRACE_PROFILE").is_some();
        let started = std::time::Instant::now();
        let gpu_scene = scene.sync_gpu_if_needed(gpu)?;
        let tlas_id = self.build_query_tlas(gpu, gpu_scene.instances(), gpu_scene.diffraction_blas_id())?;
        let scene_ms = started.elapsed().as_secs_f64() * 1000.0;
        // A spectral bin stores one complex pair per ear; time-domain IRs
        // retain the existing one-pair-per-frame layout.
        let storage_count = if self.band_count > 0 {
            (self.band_count * self.config.sound.output_channels.count() as u32 + 1)
                * 2
                * (self.config.sound.ir_num_samples - 1)
        } else {
            self.config.sound.ir_num_samples
                * if self.maximum_frequency_hz > 0.0 {
                    self.config.sound.output_channels.count() as u32
                } else {
                    1
                }
                + if self.time_domain_surfaces {
                    2 * (self.config.sound.ir_num_samples - 1)
                } else {
                    0
                }
        };
        let bytes = ir_buffer_size(storage_count) as u64;
        let ir_buffer = if let Some(previous) = previous {
            if previous.byte_size != bytes as usize || previous.scene_version != gpu_scene.version() {
                return Err(SoundError::invalid_state(
                    "partial ray pressure does not match the current scene/storage",
                ));
            }
            previous
                .buffer
                .ok_or_else(|| SoundError::invalid_state("partial ray pressure is missing"))?
        } else {
            match self.band_accumulator.as_ref() {
                Some(buffer) if Arc::strong_count(buffer) == 1 && buffer.size == bytes => buffer.clone(),
                _ => Arc::new(gpu.memory().create_storage_buffer(bytes)?),
            }
        };
        // Every query starts with zero pressure. Band scratch may be reused
        // after synchronous synthesis; escaped raw snapshots never alias a writer.
        if ray_offset == 0 {
            gpu.memory().clear_buffer(&ir_buffer)?;
        }
        let clear_ms = started.elapsed().as_secs_f64() * 1000.0;

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
            cone_width: self.cone_width,
            maximum_frequency_hz: self.maximum_frequency_hz,
            band_count: self.band_count,
            ray_offset,
            source_surface_paths: u32::from(self.source_surface_paths) | (u32::from(self.surface_two_edge_paths) << 1),
            source_profile_enabled: u32::from(query.source_radiation.profile != 0),
            source_pad: 0,
            source_inverse_rotation: query.source_radiation.orientation.normalize().conjugate().to_array(),
        };
        let mut record_ms = 0.0;
        let token = gpu.compute().submit_compute(|command_buffer| {
            let record_started = profile.then(std::time::Instant::now);
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
                        RtDescriptorWrite::storage_buffer(5, source_profile.buffer),
                    ],
                    push_constants: bytemuck::bytes_of(&constants).to_vec(),
                    push_constant_offset: 0,
                    // Invocation zero owns direct visibility. No other lane
                    // contributes when vertex paths are disabled or silent.
                    dimensions: [ray_count, 1, 1],
                    barrier_after_trace: false,
                },
            )?;
            // Ray-generation float atomics are shader writes. Make them visible
            // to IR download, caching, or convolution.
            gpu.compute().record_buffer_barrier(
                command_buffer,
                &BufferBarrierSpec::ray_tracing_shader_write_to_compute_or_transfer_read(ir_buffer.buffer),
            );
            if let Some(record_started) = record_started {
                record_ms = record_started.elapsed().as_secs_f64() * 1000.0;
            }
            Ok(())
        })?;
        let submitted_ms = if profile {
            started.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
        };
        gpu.compute().wait_for(token)?;
        let trace_ms = started.elapsed().as_secs_f64() * 1000.0;
        // The submission above is synchronous, so no descriptor or command
        // buffer can still reference this per-query TLAS. Releasing it here
        // prevents moving sources/listeners from retaining one acceleration
        // structure and backing allocation per acoustic update.
        gpu.rt().destroy_tlas(tlas_id)?;
        if profile {
            eprintln!(
                "Trace stages: scene={scene_ms:.3}, clear={:.3}, rays={:.3}, release={:.3}, record={record_ms:.3}, submit={:.3}, wait={:.3}, offset={ray_offset}, count={ray_count}, query={}",
                clear_ms - scene_ms,
                trace_ms - clear_ms,
                started.elapsed().as_secs_f64() * 1000.0 - trace_ms,
                submitted_ms - clear_ms - record_ms,
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

/// Experimental spatial UTD tracer with configurable bounded edge orders.
///
/// Each ray owns its local complex pressure; only completed, visible paths write
/// the spectrum. Stereo gains and binaural delays act on each ray's arrival.
/// Frequencies above the selected bandwidth and DC are omitted. Reflection
/// after diffraction remains incomplete; the band variant optionally adds
/// consecutive two-edge paths. It does not implement all mixed interaction orders.
pub struct ExperimentalUtdPipeline {
    pipeline: AcousticPipeline,
    inverse: Arc<dyn realfft::ComplexToReal<f32>>,
    spectrum: Vec<realfft::num_complex::Complex32>,
    scratch: Vec<realfft::num_complex::Complex32>,
    /// Optional GPU reconstruction of interpolated ray-carried complex bands.
    bands: Option<super::utd_bands::BandSynthesis>,
    /// Adaptive launch granularity; total query rays and sample indices stay fixed.
    ray_budget: Option<super::ray_budget::RayBatchBudget>,
}

impl ExperimentalUtdPipeline {
    /// Creates a spectral tracer. `ir_num_samples` is an even FFT length and
    /// `cone_width` is the regularized Keller-cone half-width in axial cosine.
    /// GPU resources follow the same ownership and synchronous submission rules
    /// as `AcousticPipeline`; assess width/ray convergence for the target scene.
    pub fn new(gpu: &VkBackend, config: EngineConfig, cone_width: f32) -> SoundResult<Self> {
        Self::with_bandwidth(gpu, config, cone_width, 2000.0)
    }

    /// Creates the experimental tracer with a specified upper frequency below
    /// Nyquist. Wider bandwidth increases per-ray coefficient/accumulation work;
    /// it does not imply measured validation outside the benchmark's frequency range.
    pub fn with_bandwidth(
        gpu: &VkBackend,
        config: EngineConfig,
        cone_width: f32,
        maximum_frequency_hz: f32,
    ) -> SoundResult<Self> {
        Self::create(gpu, config, cone_width, maximum_frequency_hz, false)
    }

    /// Keeps frequency-independent direct/reflected ray pressure in the time
    /// domain, avoiding a full spectral loop at each ordinary surface hit.
    pub fn with_time_domain_surfaces(
        gpu: &VkBackend,
        config: EngineConfig,
        cone_width: f32,
        maximum_frequency_hz: f32,
    ) -> SoundResult<Self> {
        Self::create(gpu, config, cone_width, maximum_frequency_hz, true)
    }

    /// Uses logarithmically spaced complex pressure bands on each diffracted ray.
    /// GPU reconstruction interpolates those coefficients without creating paths.
    pub fn with_bands(
        gpu: &VkBackend,
        config: EngineConfig,
        cone_width: f32,
        maximum_frequency_hz: f32,
        band_count: u32,
    ) -> SoundResult<Self> {
        Self::with_band_orders(gpu, config, cone_width, maximum_frequency_hz, band_count, 1)
    }

    /// Selects one-edge transport or experimental consecutive two-edge transport.
    /// The latter uses ray-discovered vertices and a coupled stationary-path
    /// sampling Jacobian; mixed reflection/diffraction orders remain incomplete.
    pub fn with_band_orders(
        gpu: &VkBackend,
        config: EngineConfig,
        cone_width: f32,
        maximum_frequency_hz: f32,
        band_count: u32,
        orders: u32,
    ) -> SoundResult<Self> {
        if !(1..=2).contains(&orders) {
            return Err(SoundError::invalid_argument("UTD band orders must be 1 or 2"));
        }
        if !(4..=64).contains(&band_count) || maximum_frequency_hz <= 31.25 {
            return Err(SoundError::invalid_argument(
                "UTD bands require 4..64 bands and bandwidth > 31.25 Hz",
            ));
        }
        let mut result = Self::create(gpu, config, cone_width, maximum_frequency_hz, true)?;
        let shader = if orders == 2 {
            "listener_utd_two_edges.rgen.glsl"
        } else {
            "listener_utd_bands.rgen.glsl"
        };
        let mut pipeline = AcousticPipeline::with_shader(gpu, result.pipeline.config, shader, cone_width)?;
        pipeline.maximum_frequency_hz = maximum_frequency_hz;
        pipeline.time_domain_surfaces = true;
        pipeline.band_count = band_count;
        // The optional family passes an isolated pressure check but worsens
        // current BRAS box error. Keep the measured playback candidate unchanged.
        pipeline.source_surface_paths = false;
        let accumulator_bytes = (u64::from(band_count) * config.sound.output_channels.count() as u64 + 1)
            * u64::from(config.sound.ir_num_samples)
            * size_of::<IrSample>() as u64;
        pipeline.band_accumulator = Some(Arc::new(gpu.memory().create_storage_buffer(accumulator_bytes)?));
        result.pipeline = pipeline;
        result.ray_budget = Some(super::ray_budget::RayBatchBudget::new(
            config.acoustics.listener_ray_count,
            config.sound.sample_rate,
            config.auralization.block_size,
        ));
        result.bands = Some(super::utd_bands::BandSynthesis::new(
            gpu,
            config.sound.ir_num_samples,
            config.sound.sample_rate,
            config.sound.output_channels.count() as u32,
            band_count,
            maximum_frequency_hz,
        )?);
        Ok(result)
    }

    /// Enables or disables the source-launched surface/edge family for controlled
    /// two-edge-model comparisons. This uncached pipeline applies the choice to
    /// subsequent queries; other shader variants do not launch that family.
    /// Finish or discard pending staged queries before changing this option.
    pub fn set_source_surface_paths(&mut self, enabled: bool) {
        self.pipeline.source_surface_paths = enabled;
    }

    /// Enables the disjoint R-D-D and D-D-R research families for subsequent
    /// uncached queries. Only the two-edge shader with at least three interactions
    /// traces them. Planar specular reflection defines the reference limit.
    /// Finish or discard pending staged queries before changing this option.
    pub fn set_surface_two_edge_paths(&mut self, enabled: bool) {
        self.pipeline.surface_two_edge_paths = enabled;
    }

    /// Uploads an immutable source calibration to this band tracer. Registration
    /// performs allocation/resampling and should precede realtime playback.
    /// The returned id remains valid for this pipeline's lifetime; ids are never
    /// reused, so pending ray batches cannot observe a replaced table.
    pub fn register_source_directivity(
        &mut self,
        gpu: &VkBackend,
        profile: &super::SourceDirectivity,
    ) -> SoundResult<super::SourceDirectivityId> {
        let p = &mut self.pipeline;
        if p.band_count == 0 {
            return Err(SoundError::invalid_argument(
                "source directivity requires a UTD band pipeline",
            ));
        }
        let samples = 2 * (p.config.sound.ir_num_samples - 1);
        let minimum = p.config.sound.sample_rate as f32 / samples.next_power_of_two() as f32;
        let nodes: Vec<_> = (0..p.band_count)
            .map(|i| minimum * (p.maximum_frequency_hz / minimum).powf(i as f32 / (p.band_count - 1) as f32))
            .collect();
        let pressure = profile.resample(&nodes);
        let id = u32::try_from(p.source_profiles.len())
            .map_err(|_| SoundError::invalid_state("source profile id overflow"))?;
        let mut bytes = bytemuck::cast_slice(&[profile.phi_count, profile.theta_count, p.band_count, 0]).to_vec();
        bytes.extend_from_slice(bytemuck::cast_slice(&pressure));
        let buffer = Arc::new(gpu.memory().create_storage_buffer(bytes.len() as u64)?);
        gpu.memory().upload_bytes(&buffer, &bytes)?;
        p.source_profiles.push(buffer);
        Ok(id)
    }

    /// Checks immutable registry membership before a tracked source is mutated.
    pub(crate) fn has_source_directivity(&self, id: super::SourceDirectivityId) -> bool {
        (id as usize) < self.pipeline.source_profiles.len()
    }

    fn create(
        gpu: &VkBackend,
        mut config: EngineConfig,
        cone_width: f32,
        maximum_frequency_hz: f32,
        time_domain_surfaces: bool,
    ) -> SoundResult<Self> {
        if config.sound.ir_num_samples < 4
            || config.sound.ir_num_samples % 2 != 0
            || !cone_width.is_finite()
            || !(0.0..=0.25).contains(&cone_width)
            || cone_width == 0.0
            || !maximum_frequency_hz.is_finite()
            || maximum_frequency_hz <= 0.0
            || maximum_frequency_hz >= 0.5 * config.sound.sample_rate as f32
        {
            return Err(SoundError::invalid_argument(
                "UTD requires an even FFT length >= 4, 0 < cone width <= 0.25, and a positive bandwidth below Nyquist",
            ));
        }
        let inverse = realfft::RealFftPlanner::<f32>::new().plan_fft_inverse(config.sound.ir_num_samples as usize);
        let spectrum = inverse.make_input_vec();
        let scratch = inverse.make_scratch_vec();
        config.sound.ir_num_samples = config.sound.ir_num_samples / 2 + 1;
        let shader = if time_domain_surfaces {
            "listener_utd_playground.rgen.glsl"
        } else {
            "listener_utd.rgen.glsl"
        };
        let mut pipeline = AcousticPipeline::with_shader(gpu, config, shader, cone_width)?;
        pipeline.maximum_frequency_hz = maximum_frequency_hz;
        pipeline.time_domain_surfaces = time_domain_surfaces;
        Ok(Self {
            pipeline,
            inverse,
            spectrum,
            scratch,
            bands: None,
            ray_budget: None,
        })
    }

    /// Returns the nonnegative-frequency complex transfer bins as [real, imaginary].
    /// The first and last entries represent DC and Nyquist; no source calibration,
    /// time alignment, inverse transform, or measurement fitting is performed here.
    pub fn build_transfer(
        &mut self,
        gpu: &VkBackend,
        scene: &mut SceneManager,
        query: AcousticQuery,
    ) -> SoundResult<Vec<[f32; 2]>> {
        if self.pipeline.config.sound.output_channels != OutputChannels::Mono || self.pipeline.time_domain_surfaces {
            return Err(SoundError::invalid_argument(
                "build_transfer requires mono; use build_ir_gpu for spatial output",
            ));
        }
        let data = self.pipeline.build_ir(gpu, scene, query)?;
        Ok(data.samples.iter().map(|s| [s.left, s.right]).collect())
    }

    /// Begins a band query with a bounded ray batch. The optional offset identifies
    /// the next range; absence denotes a complete time-domain snapshot.
    pub(super) fn begin_ir_gpu(
        &mut self,
        gpu: &VkBackend,
        scene: &mut SceneManager,
        query: AcousticQuery,
    ) -> SoundResult<(GpuIrSnapshot, Option<u32>)> {
        if self.bands.is_some() {
            let count = self.ray_budget.as_ref().unwrap().count(self.pipeline.ray_count(query));
            let started = std::time::Instant::now();
            let raw = self.pipeline.build_ir_range_gpu(gpu, scene, query, 0, count, None)?;
            self.ray_budget
                .as_mut()
                .unwrap()
                .observe(count, started.elapsed().as_secs_f64() * 1000.0);
            Ok((raw, Some(count)))
        } else {
            self.build_ir_gpu(gpu, scene, query).map(|ir| (ir, None))
        }
    }

    /// Advances the remaining ray range or reconstructs an already complete
    /// accumulator. Neither phase observes a newer endpoint pose.
    pub(super) fn advance_ir_gpu(
        &mut self,
        gpu: &VkBackend,
        scene: &mut SceneManager,
        query: AcousticQuery,
        raw: GpuIrSnapshot,
        next_ray: u32,
    ) -> SoundResult<(GpuIrSnapshot, Option<u32>)> {
        let total = self.pipeline.ray_count(query);
        if next_ray < total {
            let count = self.ray_budget.as_ref().unwrap().count(total - next_ray);
            let started = std::time::Instant::now();
            let raw = self
                .pipeline
                .build_ir_range_gpu(gpu, scene, query, next_ray, count, Some(raw))?;
            self.ray_budget
                .as_mut()
                .unwrap()
                .observe(count, started.elapsed().as_secs_f64() * 1000.0);
            return Ok((raw, Some(next_ray + count)));
        }
        self.bands
            .as_mut()
            .ok_or_else(|| SoundError::invalid_state("band reconstruction is not configured"))?
            .synthesize(gpu, &raw)
            .map(|ir| (ir, None))
    }

    /// Builds a time-domain GPU snapshot compatible with the cache and convolver.
    /// Each ear is transformed independently; synthesis only changes the signal
    /// representation of traced pressure. Upload completes before ownership is shared.
    pub fn build_ir_gpu(
        &mut self,
        gpu: &VkBackend,
        scene: &mut SceneManager,
        query: AcousticQuery,
    ) -> SoundResult<GpuIrSnapshot> {
        if let Some(bands) = &mut self.bands {
            let raw = self.pipeline.build_ir_gpu(gpu, scene, query)?;
            return bands.synthesize(gpu, &raw);
        }
        let raw = self.pipeline.build_ir(gpu, scene, query)?;
        let bins = self.spectrum.len();
        let channels = self.pipeline.config.sound.output_channels.count();
        let mut samples = vec![IrSample::zero(); self.inverse.len()];
        for channel in 0..channels {
            let transfer: Vec<_> = raw.samples[channel * bins..(channel + 1) * bins]
                .iter()
                .map(|s| [s.left, s.right])
                .collect();
            let values = self.synthesize_ir(&transfer)?;
            for (sample, value) in samples.iter_mut().zip(values) {
                if channel == 0 {
                    sample.left = value;
                } else {
                    sample.right = value;
                }
            }
        }
        if self.pipeline.time_domain_surfaces {
            for (sample, surface) in samples.iter_mut().zip(&raw.samples[channels * bins..]) {
                sample.left += surface.left;
                sample.right += surface.right;
            }
        }
        let byte_size = samples.len() * size_of::<IrSample>();
        let buffer = Arc::new(gpu.memory().create_storage_buffer(byte_size as u64)?);
        gpu.memory().upload_typed(&buffer, &samples)?;
        Ok(GpuIrSnapshot {
            buffer: Some(buffer),
            byte_size,
            sample_rate: raw.sample_rate,
            sample_count: samples.len() as u32,
            energy: Some(samples.iter().map(|s| s.left * s.left + s.right * s.right).sum()),
            scene_version: raw.scene_version,
            query_id: raw.query_id,
        })
    }

    /// Downloads synthesized stereo frames using the shared snapshot layout.
    pub fn download_gpu_ir(&self, gpu: &VkBackend, snapshot: &GpuIrSnapshot) -> SoundResult<IrSnapshot> {
        self.pipeline.download_gpu_ir(gpu, snapshot)
    }

    /// Synthesizes mono pressure samples from this tracer's complex transfer bins.
    /// The cached inverse plan and scratch storage are exclusively borrowed during
    /// the call. This linear transform adds no propagation paths or calibration.
    /// The returned IR has the configured FFT length and includes band-limit ringing.
    pub fn synthesize_ir(&mut self, transfer: &[[f32; 2]]) -> SoundResult<Vec<f32>> {
        if transfer.len() != self.spectrum.len() || transfer.iter().flatten().any(|v| !v.is_finite()) {
            return Err(SoundError::invalid_argument(
                "UTD transfer bins must match the FFT length and be finite",
            ));
        }
        for (target, value) in self.spectrum.iter_mut().zip(transfer) {
            *target = realfft::num_complex::Complex32::new(value[0], value[1]);
        }
        let mut output = self.inverse.make_output_vec();
        self.inverse
            .process_with_scratch(&mut self.spectrum, &mut output, &mut self.scratch)
            .map_err(|e| SoundError::invalid_argument(format!("UTD inverse FFT: {e}")))?;
        let normalization = 1.0 / output.len() as f32;
        output.iter_mut().for_each(|v| *v *= normalization);
        Ok(output)
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
    query.source_radiation.validate()?;
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
fn direct_lanczos_weights(fraction: f32) -> [f32; 16] {
    const RADIUS: i32 = 8;
    let mut weights = [0.0; 16];
    for (index, offset) in (-RADIUS + 1..=RADIUS).enumerate() {
        let distance = offset as f32 - fraction;
        let sinc = if distance.abs() < 1.0e-6 {
            1.0
        } else {
            (std::f32::consts::PI * distance).sin() / (std::f32::consts::PI * distance)
        };
        let window_argument = distance / RADIUS as f32;
        let window = if distance.abs() < 1.0e-6 {
            1.0
        } else {
            (std::f32::consts::PI * window_argument).sin() / (std::f32::consts::PI * window_argument)
        };
        weights[index] = sinc * window;
    }
    let sum = weights.iter().sum::<f32>();
    for weight in &mut weights {
        *weight /= sum;
    }
    weights
}

#[cfg(test)]
fn specular_exponent_from_roughness(roughness: f32) -> f32 {
    let alpha = roughness.clamp(0.0, 1.0).max(0.0001);
    (2.0 / (alpha * alpha) - 2.0).clamp(1.0, 4096.0)
}

#[cfg(test)]
fn folded_power_cosine_pdf(direction: Vec3, axis: Vec3, normal: Vec3, exponent: f32) -> f32 {
    power_cosine_pdf(direction.dot(axis).max(0.0), exponent)
        + power_cosine_pdf(direction.reflect(normal).dot(axis).max(0.0), exponent)
}

#[cfg(test)]
fn diffraction_radial_kernel(distance_to_edge: f32, radius: f32) -> f32 {
    if distance_to_edge < 0.0 {
        return 0.0;
    }
    2.0 * (1.0 - distance_to_edge / radius).clamp(0.0, 1.0) / radius
}

#[cfg(test)]
fn diffraction_shadow_interval(incident_angle: f64, air_angle: f64) -> Option<(f64, f64)> {
    use std::f64::consts::PI;
    if !(0.0..=air_angle).contains(&incident_angle) {
        None
    } else if incident_angle < air_angle - PI {
        Some((0.0, air_angle - PI - incident_angle))
    } else if incident_angle > PI {
        Some((PI - incident_angle, 0.0))
    } else {
        None
    }
}

#[cfg(test)]
fn diffraction_continuation_pressure_gain(base_strength: f32) -> f32 {
    base_strength.clamp(0.0, 1.0).sqrt()
}

#[cfg(test)]
fn material_energies(absorption: f32, scattering: f32, transmission: f32) -> [f32; 4] {
    let absorption = absorption.clamp(0.0, 1.0);
    let transmission = transmission.clamp(0.0, 1.0).min(1.0 - absorption);
    let reflection = (1.0 - absorption - transmission).max(0.0);
    let rough = reflection * scattering.clamp(0.0, 1.0);
    [reflection - rough, rough, transmission, absorption]
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
            source_radiation: Default::default(),
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
    fn folded_surface_lobe_preserves_probability_at_grazing_angles() {
        let normal = Vec3::Y;
        let axis = Vec3::new(1.0, 0.05, 0.0).normalize();
        let exponent = 32.0;
        let polar_steps = 512;
        let azimuth_steps = 1_024;
        let mut integral = 0.0;
        for polar_index in 0..polar_steps {
            let cosine = (polar_index as f32 + 0.5) / polar_steps as f32;
            let sine = (1.0 - cosine * cosine).sqrt();
            for azimuth_index in 0..azimuth_steps {
                let azimuth = std::f32::consts::TAU * (azimuth_index as f32 + 0.5) / azimuth_steps as f32;
                let direction = Vec3::new(sine * azimuth.cos(), cosine, sine * azimuth.sin());
                integral += folded_power_cosine_pdf(direction, axis, normal, exponent);
            }
        }
        integral *= std::f32::consts::TAU / (polar_steps * azimuth_steps) as f32;
        assert!((integral - 1.0).abs() < 0.002, "folded lobe integral was {integral}");
    }

    #[test]
    fn diffraction_strip_kernel_is_finite_normalized_and_bounded() {
        for radius in [0.05, 0.2, 1.0, 4.0] {
            let steps = 100_000;
            let width = radius / steps as f32;
            let integral = (0..steps)
                .map(|index| {
                    let distance = (index as f32 + 0.5) * width;
                    diffraction_radial_kernel(distance, radius) * width
                })
                .sum::<f32>();
            assert!((integral - 1.0).abs() < 1.0e-4);
            assert_eq!(diffraction_radial_kernel(0.0, radius), 2.0 / radius);
            assert_eq!(diffraction_radial_kernel(-0.5 * radius, radius), 0.0);
            assert_eq!(diffraction_radial_kernel(radius, radius), 0.0);
            assert_eq!(diffraction_radial_kernel(2.0 * radius, radius), 0.0);
        }
    }

    #[test]
    fn diffraction_shadow_domain_covers_air_including_deep_shadow() {
        use std::f64::consts::{PI, TAU};
        let mut deep_shadow = false;
        for air_angle in [PI, 1.1 * PI, 1.5 * PI, 1.9 * PI, TAU] {
            for incidence in 0..64 {
                let incident = (incidence as f64 + 0.5) * air_angle / 64.0;
                let Some((start, end)) = diffraction_shadow_interval(incident, air_angle) else {
                    assert!(incident >= air_angle - PI && incident <= PI);
                    continue;
                };
                for step in 0..1024 {
                    let angle = start + (step as f64 + 0.5) / 1024.0 * (end-start);
                    let outgoing = (incident + PI + angle).rem_euclid(TAU);
                    assert!(outgoing <= air_angle + 1e-12);
                    assert!((outgoing-incident).abs() >= PI-1e-12);
                    deep_shadow |= angle.abs() > PI/2.0;
                }
            }
        }
        assert!(deep_shadow);
    }

    #[test]
    fn projected_strip_density_is_radius_independent_for_uniform_launches() {
        // Independent area-to-solid-angle integration. Relocating proxy samples
        // must recover the edge density for both approaching and receding strips.
        for normal_distance in [0.1_f64, 1.0, 3.0] {
            for lateral in [-1.0_f64, 0.0, 1.0] {
                for radius in [0.05_f64, 0.2, 0.9] {
                    let edge_distance = normal_distance.hypot(lateral);
                    let expected = normal_distance / edge_distance.powi(3);
                    let mut measured = 0.0;
                    for step in 0..4096 {
                        let r = (step as f64 + 0.5) * radius / 4096.0;
                        let proxy_distance = normal_distance.hypot(lateral + r);
                        let density = normal_distance / proxy_distance.powi(3);
                        let weight = (proxy_distance / edge_distance).powi(3);
                        measured += density * weight * 2.0 * (1.0 - r / radius) / 4096.0;
                    }
                    assert!((measured / expected - 1.0).abs() < 1e-10);
                }
            }
        }
    }

    #[test]
    fn diffraction_continuation_never_amplifies_and_is_radius_independent() {
        for strength in [0.0, 0.01, 0.25, 0.65, 1.0] {
            let gain = diffraction_continuation_pressure_gain(strength);
            assert!(gain.is_finite() && (0.0..=1.0).contains(&gain));
            assert!((gain * gain - strength).abs() < 1.0e-6);

            let mut multi_edge_gain = 1.0;
            for _ in 0..16 {
                multi_edge_gain *= gain;
                assert!(multi_edge_gain <= 1.0);
            }
        }
    }

    #[test]
    fn material_energy_partition_neither_loses_nor_creates_energy() {
        for absorption_step in 0..=20 {
            for scattering_step in 0..=20 {
                for transmission_step in 0..=20 {
                    let energies = material_energies(
                        absorption_step as f32 / 20.0,
                        scattering_step as f32 / 20.0,
                        transmission_step as f32 / 20.0,
                    );
                    assert!(energies.iter().all(|energy| energy.is_finite() && *energy >= 0.0));
                    assert!((energies.iter().sum::<f32>() - 1.0).abs() < 1.0e-6);
                }
            }
        }
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
    fn direct_lanczos_delay_preserves_pressure_and_bounds_energy_ripple() {
        let mut minimum_energy = f32::INFINITY;
        let mut maximum_energy = f32::NEG_INFINITY;
        for step in 0..=100 {
            let weights = direct_lanczos_weights(step as f32 / 100.0);
            assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1.0e-5);
            let energy = weights.iter().map(|weight| weight * weight).sum::<f32>();
            minimum_energy = minimum_energy.min(energy);
            maximum_energy = maximum_energy.max(energy);
        }
        let ripple_db = 10.0 * (maximum_energy / minimum_energy).log10();
        assert!(
            ripple_db < 0.4,
            "direct fractional-delay energy ripple was {ripple_db} dB"
        );
    }

    #[test]
    fn specular_exponent_uses_rms_slope_relation() {
        assert_eq!(specular_exponent_from_roughness(0.0), 4096.0);
        assert!((specular_exponent_from_roughness(0.15) - 86.888_89).abs() < 1.0e-4);
        assert_eq!(specular_exponent_from_roughness(1.0), 1.0);
        assert!(specular_exponent_from_roughness(0.1) > specular_exponent_from_roughness(0.2));
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
        assert_eq!(size_of::<ListenerTracePushConstants>(), 128);
        assert_eq!(
            std::mem::offset_of!(ListenerTracePushConstants, source_profile_enabled),
            104
        );
        assert_eq!(
            std::mem::offset_of!(ListenerTracePushConstants, source_inverse_rotation),
            112
        );
        assert_eq!(std::mem::offset_of!(ListenerTracePushConstants, band_count), 92);
        assert_eq!(
            std::mem::offset_of!(ListenerTracePushConstants, maximum_frequency_hz),
            88
        );
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
