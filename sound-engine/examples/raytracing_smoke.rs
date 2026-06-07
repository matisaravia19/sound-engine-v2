use ash::vk;
use sound_engine::gpu::GpuError;
use sound_engine::gpu::backend::VkBackend;
use sound_engine::gpu::rt::{
    BlasBuildSpec, RtDescriptorBindingSpec, RtDescriptorWrite, RtInstanceSpec, RtMeshSpec, RtPipelineSpec,
    RtPushConstantSpec, RtShaderGroupSpec, RtShaderStageSpec, RtSubmitExt, RtTraceSpec, TlasBuildSpec,
};
use sound_engine::gpu::shader::{ShaderId, ShaderLibrary, ShaderStage};
use std::mem::size_of;
use std::path::PathBuf;

const SHADER_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/shaders");
const HIT_VALUE: u32 = 1;
const MISS_VALUE: u32 = 2;

fn main() -> Result<(), GpuError> {
    let gpu = VkBackend::new()?;
    let result = run_smoke_trace(&gpu)?;

    println!("raytracing smoke result: hit={}, miss={}", result.hit, result.miss);
    if result.hit != HIT_VALUE || result.miss != MISS_VALUE {
        return Err(std::io::Error::other(format!(
            "Unexpected smoke trace result: expected hit={HIT_VALUE}, miss={MISS_VALUE}"
        ))
        .into());
    }

    Ok(())
}

struct SmokeTraceResult {
    hit: u32,
    miss: u32,
}

fn run_smoke_trace(gpu: &VkBackend) -> Result<SmokeTraceResult, GpuError> {
    let vertices: [[f32; 3]; 3] = [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]];
    let mesh = gpu.rt().upload_mesh(
        gpu.memory(),
        RtMeshSpec {
            vertices: &vertices,
            indices: None,
            opaque: true,
        },
    )?;

    let blas_id = gpu.rt().build_blas(
        gpu.memory(),
        BlasBuildSpec {
            mesh: &mesh,
            flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
        },
    )?;
    let tlas_id = gpu.rt().build_tlas(
        gpu.memory(),
        TlasBuildSpec {
            instances: &[RtInstanceSpec {
                blas_id,
                transform: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                custom_index: 0,
                mask: 0xff,
                sbt_record_offset: 0,
            }],
            flags: vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
        },
    )?;

    let result_buffer = gpu.memory().create_host_device_address_buffer(
        (2 * size_of::<u32>()) as vk::DeviceSize,
        vk::BufferUsageFlags::STORAGE_BUFFER,
    )?;
    gpu.memory()
        .write_mapped_bytes(&result_buffer, bytemuck::cast_slice(&[0_u32; 2]))?;

    let shader_ids = load_smoke_shaders(gpu.shaders())?;
    let pipeline_id = gpu.rt().create_pipeline(
        gpu.memory(),
        RtPipelineSpec {
            shaders: vec![
                RtShaderStageSpec::new(shader_ids.raygen),
                RtShaderStageSpec::new(shader_ids.miss),
                RtShaderStageSpec::new(shader_ids.closest_hit),
            ],
            groups: vec![
                RtShaderGroupSpec::Raygen { general_shader: 0 },
                RtShaderGroupSpec::Miss { general_shader: 1 },
                RtShaderGroupSpec::TrianglesHit { closest_hit_shader: 2 },
            ],
            descriptor_bindings: vec![
                RtDescriptorBindingSpec::AccelerationStructure {
                    binding: 0,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::RAYGEN_KHR,
                },
                RtDescriptorBindingSpec::StorageBuffer {
                    binding: 1,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::RAYGEN_KHR,
                },
            ],
            push_constant_ranges: Vec::<RtPushConstantSpec>::new(),
            max_ray_recursion_depth: 1,
        },
    )?;

    gpu.rt().submit_rt_and_wait(|command_buffer| {
        gpu.rt().record_trace(
            command_buffer,
            &RtTraceSpec {
                pipeline_id,
                descriptor_writes: vec![
                    RtDescriptorWrite::acceleration_structure(0, tlas_id),
                    RtDescriptorWrite::storage_buffer(1, result_buffer.buffer),
                ],
                push_constants: Vec::new(),
                push_constant_offset: 0,
                dimensions: [2, 1, 1],
                barrier_after_trace: true,
            },
        )
    })?;

    let mut result = [0_u32; 2];
    gpu.memory().read_mapped_bytes(
        &result_buffer,
        2 * size_of::<u32>(),
        bytemuck::cast_slice_mut(&mut result),
    )?;

    Ok(SmokeTraceResult {
        hit: result[0],
        miss: result[1],
    })
}

#[derive(Clone, Copy)]
struct SmokeShaderIds {
    raygen: ShaderId,
    miss: ShaderId,
    closest_hit: ShaderId,
}

fn load_smoke_shaders(shaders: &ShaderLibrary) -> Result<SmokeShaderIds, GpuError> {
    Ok(SmokeShaderIds {
        raygen: shaders.load_glsl_file(ShaderStage::RayGeneration, shader_path("smoke.rgen.glsl"))?,
        miss: shaders.load_glsl_file(ShaderStage::RayMiss, shader_path("smoke.rmiss.glsl"))?,
        closest_hit: shaders.load_glsl_file(ShaderStage::RayClosestHit, shader_path("smoke.rchit.glsl"))?,
    })
}

fn shader_path(file: &str) -> PathBuf {
    PathBuf::from(SHADER_DIR).join(file)
}
