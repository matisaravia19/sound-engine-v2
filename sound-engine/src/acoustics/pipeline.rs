use crate::acoustics::{AcousticHit, AcousticHitSnapshot, GpuAcousticHitSnapshot, GpuIrSnapshot, IrSample, IrSnapshot};
use crate::core::config::{EngineConfig, OutputChannels};
use crate::core::error::{SoundError, SoundResult};
use crate::gpu::backend::VkBackend;
use crate::gpu::compute::{
    BufferBarrierSpec, ComputePipelineSpec, DescriptorBindingSpec, DescriptorWrite, DispatchSpec, PipelineId,
    PushConstantSpec,
};
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
use std::sync::Arc;

const SHADER_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/acoustics/shaders");
const HIT_BUFFER_HEADER_SIZE: usize = size_of::<GpuHitBufferHeader>();
const COMPUTE_GROUP_SIZE: u32 = 256;

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

/// Acoustic pipeline that traces acoustic paths into listener hits and bins them into IR snapshots.
///
/// The pipeline owns long-lived RT resources and a hit accumulation buffer.
/// Calls are synchronous today: GPU ray tracing writes listener hits, GPU
/// compute bins those hits into an IR, and CPU readback is only used by
/// compatibility/debug snapshot APIs.
pub struct AcousticPipeline {
    config: EngineConfig,
    contribution_pipeline: RtPipelineId,
    hit_capacity: u32,
    hits_to_ir_pipeline: PipelineId,
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
    hit_capacity: u32,
}

unsafe impl bytemuck::Zeroable for VisibilityPushConstants {}
unsafe impl bytemuck::Pod for VisibilityPushConstants {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GpuHitBufferHeader {
    count: u32,
    overflow_count: u32,
    _pad0: [u32; 2],
}

unsafe impl bytemuck::Zeroable for GpuHitBufferHeader {}
unsafe impl bytemuck::Pod for GpuHitBufferHeader {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GpuAcousticHit {
    /// Octahedral direction packed with GLSL `packSnorm2x16`.
    packed_incoming_direction: u32,
    /// Octahedral listener-sphere hit direction packed with GLSL `packSnorm2x16`.
    packed_listener_hit_direction: u32,
    arrival_time_seconds: f32,
    energy: f32,
}

unsafe impl bytemuck::Zeroable for GpuAcousticHit {}
unsafe impl bytemuck::Pod for GpuAcousticHit {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct HitsToIrPushConstants {
    listener_right: [f32; 3],
    sample_rate: u32,
    sample_count: u32,
    output_channels: u32,
    hit_capacity: u32,
    _pad0: u32,
}

unsafe impl bytemuck::Zeroable for HitsToIrPushConstants {}
unsafe impl bytemuck::Pod for HitsToIrPushConstants {}

impl AcousticPipeline {
    /// Creates the RT contribution pipeline, GPU contribution buffer, and IR builder.
    pub fn new(gpu: &VkBackend, config: EngineConfig) -> SoundResult<Self> {
        validate_listener_radius(config.acoustics.listener_radius)?;
        validate_ir_dimensions(config.sound.sample_rate, config.sound.ir_num_samples)?;

        let raygen = gpu
            .shaders()
            .load_glsl_file(ShaderStage::RayGeneration, shader_path("acoustics.rgen.glsl"))?;
        let miss = gpu
            .shaders()
            .load_glsl_file(ShaderStage::RayMiss, shader_path("acoustics.rmiss.glsl"))?;
        let closest_hit = gpu
            .shaders()
            .load_glsl_file(ShaderStage::RayClosestHit, shader_path("solid.rchit.glsl"))?;
        let listener_closest_hit = gpu
            .shaders()
            .load_glsl_file(ShaderStage::RayClosestHit, shader_path("listener.rchit.glsl"))?;
        let listener_intersection = gpu
            .shaders()
            .load_glsl_file(ShaderStage::RayIntersection, shader_path("listener.rint.glsl"))?;
        let diffraction_closest_hit = gpu
            .shaders()
            .load_glsl_file(ShaderStage::RayClosestHit, shader_path("diffraction.rchit.glsl"))?;
        let diffraction_intersection = gpu
            .shaders()
            .load_glsl_file(ShaderStage::RayIntersection, shader_path("diffraction.rint.glsl"))?;
        let contribution_pipeline = gpu.rt().create_pipeline(RtPipelineSpec {
            shaders: vec![
                RtShaderStageSpec::new(raygen),
                RtShaderStageSpec::new(miss),
                RtShaderStageSpec::new(closest_hit),
                RtShaderStageSpec::new(listener_closest_hit),
                RtShaderStageSpec::new(listener_intersection),
                RtShaderStageSpec::new(diffraction_closest_hit),
                RtShaderStageSpec::new(diffraction_intersection),
            ],
            groups: vec![
                RtShaderGroupSpec::Raygen { shader: 0 },
                RtShaderGroupSpec::Miss { shader: 1 },
                RtShaderGroupSpec::TrianglesHit { closest_hit_shader: 2 },
                RtShaderGroupSpec::ProceduralHit {
                    closest_hit_shader: 3,
                    intersection_shader: 4,
                },
                RtShaderGroupSpec::ProceduralHit {
                    closest_hit_shader: 5,
                    intersection_shader: 6,
                },
            ],
            descriptor_bindings: vec![
                RtDescriptorBindingSpec::acceleration_structure(0),
                RtDescriptorBindingSpec::storage_buffer(1),
                RtDescriptorBindingSpec::storage_buffer(2),
                RtDescriptorBindingSpec::storage_buffer(3),
                RtDescriptorBindingSpec::storage_buffer(4),
            ],
            push_constant_ranges: vec![RtPushConstantSpec::new(0, size_of::<VisibilityPushConstants>() as u32)],
            max_ray_recursion_depth: config.acoustics.max_bounces.saturating_add(1).max(1),
        })?;
        let hit_capacity = hit_buffer_capacity(config.acoustics.rays_per_query, config.acoustics.max_bounces);
        let hits_to_ir_pipeline = create_hits_to_ir_pipeline(gpu)?;
        let listener_bounds = create_listener_sphere_bounds(gpu, config.acoustics.listener_radius)?;
        let listener_blas = gpu.rt().build_aabb_blas(AabbBlasBuildSpec {
            aabbs: &listener_bounds,
            flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
        })?;

        Ok(Self {
            config,
            contribution_pipeline,
            hit_capacity,
            hits_to_ir_pipeline,
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
        let gpu_ir = self.build_ir_gpu(gpu, scene, query)?;
        self.download_gpu_ir(gpu, &gpu_ir)
    }

    /// Builds one GPU-resident time-domain IR for a scene/source/listener query.
    pub fn build_ir_gpu(
        &mut self,
        gpu: &VkBackend,
        scene: &mut SceneManager,
        query: AcousticQuery,
    ) -> SoundResult<GpuIrSnapshot> {
        let hits = self.trace_hits_gpu(gpu, scene, query)?;
        self.build_ir_from_gpu_hits(gpu, &hits, query.listener_right, query.query_id)
    }

    /// Traces one source/listener location query into direction-independent listener hits.
    pub fn trace_hits(
        &mut self,
        gpu: &VkBackend,
        scene: &mut SceneManager,
        query: AcousticQuery,
    ) -> SoundResult<AcousticHitSnapshot> {
        let hits = self.trace_hits_gpu(gpu, scene, query)?;
        self.download_hits(gpu, &hits)
    }

    /// Traces one source/listener location query into a GPU-resident hit buffer.
    pub fn trace_hits_gpu(
        &mut self,
        gpu: &VkBackend,
        scene: &mut SceneManager,
        query: AcousticQuery,
    ) -> SoundResult<GpuAcousticHitSnapshot> {
        let gpu_scene = scene.sync_gpu_if_needed(gpu)?;
        let tlas_id = self.build_query_tlas(
            gpu,
            gpu_scene.instances(),
            gpu_scene.diffraction_blas_id(),
            query.listener_position,
        )?;

        let _ray_budget = self.config.acoustics.rays_per_query;
        let _max_bounces = self.config.acoustics.max_bounces;
        let visibility_constants = VisibilityPushConstants {
            source_position: query.source_position.to_array(),
            source_energy: query.source_energy,
            listener_position: query.listener_position.to_array(),
            speed_of_sound: SPEED_OF_SOUND_METERS_PER_SECOND,
            listener_radius: self.config.acoustics.listener_radius,
            listener_volume: listener_sphere_volume(self.config.acoustics.listener_radius),
            _pad_listener_volume: 0,
            ray_count: self.config.acoustics.rays_per_query.max(1),
            max_bounces: self.config.acoustics.max_bounces,
            _pad0: [0; 3],
            listener_right: query.listener_right.to_array(),
            sample_rate: self.config.sound.sample_rate,
            sample_count: self.config.sound.ir_num_samples,
            output_channels: output_channels_to_shader(self.config.sound.output_channels),
            hit_capacity: self.hit_capacity,
        };
        let hit_buffer = Arc::new(
            gpu.memory()
                .create_storage_buffer(hit_buffer_size(self.hit_capacity) as u64)?,
        );

        gpu.memory().clear_buffer(&hit_buffer)?;
        gpu.compute().submit_compute_and_wait(|command_buffer| {
            gpu.rt().record_trace(
                command_buffer,
                &RtTraceSpec {
                    pipeline_id: self.contribution_pipeline,
                    descriptor_writes: vec![
                        RtDescriptorWrite::acceleration_structure(0, tlas_id),
                        RtDescriptorWrite::storage_buffer(1, hit_buffer.buffer),
                        RtDescriptorWrite::storage_buffer(2, gpu_scene.object_buffer().buffer),
                        RtDescriptorWrite::storage_buffer(3, gpu_scene.material_buffer().buffer),
                        RtDescriptorWrite::storage_buffer(4, gpu_scene.diffraction_edge_buffer().buffer),
                    ],
                    push_constants: bytemuck::bytes_of(&visibility_constants).to_vec(),
                    push_constant_offset: 0,
                    dimensions: [self.config.acoustics.rays_per_query.max(1), 1, 1],
                    barrier_after_trace: false,
                },
            )?;
            gpu.compute().record_buffer_barrier(
                command_buffer,
                &BufferBarrierSpec::ray_tracing_shader_write_to_compute_or_transfer_read(hit_buffer.buffer),
            );

            Ok(())
        })?;

        Ok(GpuAcousticHitSnapshot {
            buffer: Some(hit_buffer),
            hit_capacity: self.hit_capacity,
            byte_size: hit_buffer_size(self.hit_capacity),
            scene_version: gpu_scene.version(),
            query_id: query.query_id,
            source_position: query.source_position,
            source_energy: query.source_energy,
            listener_position: query.listener_position,
        })
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

    /// Builds one IR from previously traced listener hits and the current listener orientation.
    pub fn build_ir_from_hits(&self, hits: &AcousticHitSnapshot, listener_right: Vec3, query_id: u32) -> IrSnapshot {
        build_ir_from_hits(
            hits,
            listener_right,
            query_id,
            self.config.sound.sample_rate,
            self.config.sound.ir_num_samples,
            self.config.sound.output_channels,
        )
    }

    /// Builds a GPU-resident IR from cached GPU hit data and the requested listener orientation.
    pub fn build_ir_from_gpu_hits(
        &self,
        gpu: &VkBackend,
        hits: &GpuAcousticHitSnapshot,
        listener_right: Vec3,
        query_id: u32,
    ) -> SoundResult<GpuIrSnapshot> {
        let hit_buffer = hits
            .buffer
            .as_ref()
            .ok_or_else(|| SoundError::invalid_argument("GPU hit snapshot has no buffer"))?;
        let ir_buffer = Arc::new(
            gpu.memory()
                .create_storage_buffer(ir_buffer_size(self.config.sound.ir_num_samples) as u64)?,
        );
        gpu.memory().clear_buffer(&ir_buffer)?;

        let hits_to_ir_constants = HitsToIrPushConstants {
            listener_right: listener_right.to_array(),
            sample_rate: self.config.sound.sample_rate,
            sample_count: self.config.sound.ir_num_samples,
            output_channels: output_channels_to_shader(self.config.sound.output_channels),
            hit_capacity: hits.hit_capacity,
            _pad0: 0,
        };

        gpu.compute().submit_compute_and_wait(|command_buffer| {
            gpu.compute().record_dispatch(
                command_buffer,
                &DispatchSpec {
                    pipeline_id: self.hits_to_ir_pipeline,
                    descriptor_writes: vec![
                        DescriptorWrite::storage_buffer(0, hit_buffer.buffer),
                        DescriptorWrite::storage_buffer(1, ir_buffer.buffer),
                    ],
                    push_constants: bytemuck::bytes_of(&hits_to_ir_constants).to_vec(),
                    push_constant_offset: 0,
                    groups: [hits.hit_capacity.max(1).div_ceil(COMPUTE_GROUP_SIZE), 1, 1],
                },
            )?;
            gpu.compute().record_buffer_barrier(
                command_buffer,
                &BufferBarrierSpec::compute_shader_write_to_transfer_read(ir_buffer.buffer),
            );
            Ok(())
        })?;

        Ok(GpuIrSnapshot {
            buffer: Some(ir_buffer),
            byte_size: ir_buffer_size(self.config.sound.ir_num_samples),
            sample_rate: self.config.sound.sample_rate,
            sample_count: self.config.sound.ir_num_samples,
            energy: None,
            scene_version: hits.scene_version,
            query_id,
        })
    }

    fn download_hits(&self, gpu: &VkBackend, snapshot: &GpuAcousticHitSnapshot) -> SoundResult<AcousticHitSnapshot> {
        let hit_buffer = snapshot
            .buffer
            .as_ref()
            .ok_or_else(|| SoundError::invalid_argument("GPU hit snapshot has no buffer"))?;
        let mut bytes = vec![0u8; snapshot.byte_size];
        gpu.memory().download_bytes(hit_buffer, bytes.len(), &mut bytes)?;

        let header = bytemuck::pod_read_unaligned::<GpuHitBufferHeader>(&bytes[..HIT_BUFFER_HEADER_SIZE]);
        let stored_count = header.count.min(snapshot.hit_capacity) as usize;
        let hit_bytes = &bytes[HIT_BUFFER_HEADER_SIZE
            ..HIT_BUFFER_HEADER_SIZE + snapshot.hit_capacity as usize * size_of::<GpuAcousticHit>()];
        let hits = hit_bytes
            .chunks_exact(size_of::<GpuAcousticHit>())
            .take(stored_count)
            .map(|bytes| bytemuck::pod_read_unaligned::<GpuAcousticHit>(bytes))
            .map(|hit| AcousticHit {
                incoming_direction: decode_octahedral_snorm16(hit.packed_incoming_direction),
                listener_hit_direction: decode_octahedral_snorm16(hit.packed_listener_hit_direction),
                arrival_time_seconds: hit.arrival_time_seconds,
                energy: hit.energy,
            })
            .collect();

        Ok(AcousticHitSnapshot {
            hits,
            scene_version: snapshot.scene_version,
            query_id: snapshot.query_id,
            source_position: snapshot.source_position,
            source_energy: snapshot.source_energy,
            listener_position: snapshot.listener_position,
            overflow_count: header.overflow_count,
        })
    }

    /// Downloads a GPU-resident time-domain IR into a CPU snapshot.
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
        listener_position: Vec3,
    ) -> SoundResult<crate::gpu::rt::TlasId> {
        let mut instances = Vec::with_capacity(scene_instances.len() + 2);
        instances.extend_from_slice(scene_instances);
        if let Some(blas_id) = diffraction_blas_id {
            instances.push(RtInstanceSpec {
                blas_id,
                transform: Mat4::IDENTITY,
                custom_index: DIFFRACTION_INSTANCE_CUSTOM_INDEX,
                mask: 0xff,
                sbt_record_offset: DIFFRACTION_HIT_GROUP_OFFSET,
            });
        }
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
const DIFFRACTION_INSTANCE_CUSTOM_INDEX: u32 = u32::MAX - 1;
const LISTENER_HIT_GROUP_OFFSET: u32 = 1;
const DIFFRACTION_HIT_GROUP_OFFSET: u32 = 2;
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

fn hit_buffer_capacity(rays_per_query: u32, max_bounces: u32) -> u32 {
    let multiplier = max_bounces.saturating_add(1).saturating_mul(2).max(1);
    rays_per_query
        .max(1)
        .saturating_mul(multiplier)
        .max(rays_per_query.max(1))
}

fn hit_buffer_size(hit_capacity: u32) -> usize {
    HIT_BUFFER_HEADER_SIZE + hit_capacity as usize * size_of::<GpuAcousticHit>()
}

fn ir_buffer_size(sample_count: u32) -> usize {
    sample_count as usize * size_of::<IrSample>()
}

fn create_hits_to_ir_pipeline(gpu: &VkBackend) -> SoundResult<PipelineId> {
    let shader = gpu
        .shaders()
        .load_glsl_file(ShaderStage::Compute, shader_path("hits_to_ir.comp.glsl"))?;
    gpu.compute().create_pipeline(ComputePipelineSpec {
        shader_id: shader,
        descriptor_bindings: vec![
            DescriptorBindingSpec::storage_buffer(0),
            DescriptorBindingSpec::storage_buffer(1),
        ],
        push_constant_ranges: vec![PushConstantSpec::new(0, size_of::<HitsToIrPushConstants>() as u32)],
    })
}

fn decode_octahedral_snorm16(packed: u32) -> Vec3 {
    // Matches GLSL packSnorm2x16(octEncode(direction)): low 16 bits are X,
    // high 16 bits are Y, and the octahedral fold reconstructs Z.
    let x = decode_snorm16((packed & 0xffff) as u16);
    let y = decode_snorm16((packed >> 16) as u16);
    let mut direction = Vec3::new(x, y, 1.0 - x.abs() - y.abs());

    if direction.z < 0.0 {
        let folded_x = (1.0 - direction.y.abs()) * sign_not_zero(direction.x);
        let folded_y = (1.0 - direction.x.abs()) * sign_not_zero(direction.y);
        direction.x = folded_x;
        direction.y = folded_y;
    }

    direction.normalize_or_zero()
}

fn decode_snorm16(value: u16) -> f32 {
    let signed = value as i16;
    ((signed as f32) / 32767.0).max(-1.0)
}

fn sign_not_zero(value: f32) -> f32 {
    if value >= 0.0 { 1.0 } else { -1.0 }
}

fn output_channels_to_shader(output_channels: OutputChannels) -> u32 {
    match output_channels {
        OutputChannels::Mono => 0,
        OutputChannels::Stereo => 1,
        OutputChannels::Binaural => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_hit_record_stays_compact() {
        assert_eq!(size_of::<GpuAcousticHit>(), 16);
    }

    #[test]
    fn decodes_octahedral_axis_directions() {
        assert!(decode_octahedral_snorm16(0).abs_diff_eq(Vec3::Z, 0.0001));
        assert!(decode_octahedral_snorm16(0x0000_7fff).abs_diff_eq(Vec3::X, 0.0001));
        assert!(decode_octahedral_snorm16(0x7fff_0000).abs_diff_eq(Vec3::Y, 0.0001));
    }

    #[test]
    fn bins_mono_hits_into_left_channel() {
        let hits = hit_snapshot(vec![AcousticHit {
            incoming_direction: Vec3::X,
            listener_hit_direction: Vec3::Y,
            arrival_time_seconds: 0.001,
            energy: 0.5,
        }]);

        let ir = build_ir_from_hits(&hits, Vec3::X, 3, 1_000, 4, OutputChannels::Mono);

        assert_eq!(ir.query_id, 3);
        assert_eq!(ir.samples[1].left, 0.5);
        assert_eq!(ir.samples[1].right, 0.0);
        assert_eq!(ir.energy, 0.25);
    }

    #[test]
    fn bins_stereo_hits_with_equal_power_pan() {
        let hits = hit_snapshot(vec![
            AcousticHit {
                incoming_direction: Vec3::X,
                listener_hit_direction: Vec3::Y,
                arrival_time_seconds: 0.0,
                energy: 1.0,
            },
            AcousticHit {
                incoming_direction: -Vec3::X,
                listener_hit_direction: -Vec3::Y,
                arrival_time_seconds: 0.0,
                energy: 1.0,
            },
        ]);

        let ir = build_ir_from_hits(&hits, Vec3::X, 0, 1_000, 1, OutputChannels::Stereo);

        assert_eq!(ir.samples[0].left, 1.0);
        assert_eq!(ir.samples[0].right, 1.0);
        assert_eq!(ir.energy, 2.0);
    }

    #[test]
    fn ignores_out_of_range_hits() {
        let hits = hit_snapshot(vec![AcousticHit {
            incoming_direction: Vec3::X,
            listener_hit_direction: Vec3::Y,
            arrival_time_seconds: 10.0,
            energy: 1.0,
        }]);

        let ir = build_ir_from_hits(&hits, Vec3::X, 0, 1_000, 4, OutputChannels::Stereo);

        assert!(
            ir.samples
                .iter()
                .all(|sample| sample.left == 0.0 && sample.right == 0.0)
        );
        assert_eq!(ir.energy, 0.0);
    }

    fn hit_snapshot(hits: Vec<AcousticHit>) -> AcousticHitSnapshot {
        AcousticHitSnapshot {
            hits,
            scene_version: 7,
            query_id: 0,
            source_position: Vec3::ZERO,
            source_energy: 1.0,
            listener_position: Vec3::Y,
            overflow_count: 0,
        }
    }
}

fn build_ir_from_hits(
    hits: &AcousticHitSnapshot,
    listener_right: Vec3,
    query_id: u32,
    sample_rate: u32,
    sample_count: u32,
    output_channels: OutputChannels,
) -> IrSnapshot {
    let mut samples = vec![IrSample::zero(); sample_count as usize];
    let normalized_right = listener_right.normalize_or_zero();

    for hit in &hits.hits {
        if !hit.arrival_time_seconds.is_finite() || hit.arrival_time_seconds < 0.0 || !hit.energy.is_finite() {
            continue;
        }
        let sample_index = (hit.arrival_time_seconds * sample_rate as f32).round() as usize;
        let Some(sample) = samples.get_mut(sample_index) else {
            continue;
        };

        match output_channels {
            OutputChannels::Mono => {
                sample.left += hit.energy;
            }
            OutputChannels::Stereo | OutputChannels::Binaural => {
                let pan = hit
                    .incoming_direction
                    .normalize_or_zero()
                    .dot(normalized_right)
                    .clamp(-1.0, 1.0);
                let left_factor = (0.5 * (1.0 - pan)).sqrt();
                let right_factor = (0.5 * (1.0 + pan)).sqrt();
                sample.left += hit.energy * left_factor;
                sample.right += hit.energy * right_factor;
            }
        }
    }

    let energy = samples
        .iter()
        .map(|sample| sample.left * sample.left + sample.right * sample.right)
        .sum();

    IrSnapshot {
        sample_rate,
        samples,
        energy,
        scene_version: hits.scene_version,
        query_id,
    }
}
