use glam::{Mat4, Vec3, vec3};
use sound_engine::acoustics::{AcousticPipeline, AcousticQuery};
use sound_engine::core::config::{
    AcousticsConfig, AuralizationConfig, EngineConfig, IrCacheConfig, OutputChannels, SoundConfig,
};
use sound_engine::core::debug::export_ir;
use sound_engine::core::error::{SoundError, SoundResult};
use sound_engine::gpu::backend::VkBackend;
use sound_engine::scene::{DiffractionEdge, Material, MeshAsset, SceneDescription, SceneManager, SceneObject};

const BASELINE_IR_EXPORT_PATH: &str = "target/gpu_ir_diffraction_baseline_ir.csv";
const DIFFRACTION_IR_EXPORT_PATH: &str = "target/gpu_ir_diffraction_edge_ir.csv";
const SAMPLE_RATE: u32 = 48_000;
const IR_SAMPLES: u32 = SAMPLE_RATE;

fn main() -> SoundResult<()> {
    let gpu = VkBackend::new()?;
    let source_position = vec3(-1.0, 0.0, -0.5);
    let listener_position = vec3(-1.0, 0.0, 0.5);

    let baseline = trace_scene(&gpu, diffraction_scene(false), 1, source_position, listener_position)?;
    let diffracted = trace_scene(&gpu, diffraction_scene(true), 2, source_position, listener_position)?;

    export_ir(&baseline, BASELINE_IR_EXPORT_PATH)?;
    export_ir(&diffracted, DIFFRACTION_IR_EXPORT_PATH)?;

    let baseline_nonzero = baseline
        .samples
        .iter()
        .filter(|sample| sample.left != 0.0 || sample.right != 0.0)
        .count();
    let diffracted_nonzero = diffracted
        .samples
        .iter()
        .filter(|sample| sample.left != 0.0 || sample.right != 0.0)
        .count();

    println!(
        "diffraction edge smoke: baseline_energy={}, diffraction_energy={}",
        baseline.energy, diffracted.energy
    );
    println!("  baseline nonzero samples: {baseline_nonzero}");
    println!("  diffraction nonzero samples: {diffracted_nonzero}");
    println!("  exported baseline ir: {BASELINE_IR_EXPORT_PATH}");
    println!("  exported diffraction ir: {DIFFRACTION_IR_EXPORT_PATH}");

    if diffracted.energy <= 0.0 {
        return Err(SoundError::invalid_state(
            "Expected diffraction edge to produce non-zero IR energy",
        ));
    }
    if diffracted_nonzero <= baseline_nonzero {
        return Err(SoundError::invalid_state(
            "Expected diffraction edge to add extra shadow-region arrivals",
        ));
    }

    Ok(())
}

fn trace_scene(
    gpu: &VkBackend,
    scene_description: SceneDescription,
    query_id: u32,
    source_position: Vec3,
    listener_position: Vec3,
) -> SoundResult<sound_engine::acoustics::IrSnapshot> {
    let mut scene = SceneManager::new();
    scene.load(scene_description)?;
    let mut pipeline = AcousticPipeline::new(gpu, engine_config())?;

    pipeline.build_ir(
        gpu,
        &mut scene,
        AcousticQuery {
            query_id,
            source_position,
            listener_position,
            listener_right: Vec3::X,
            source_energy: 2_000.0,
        },
    )
}

fn engine_config() -> EngineConfig {
    EngineConfig {
        sound: SoundConfig {
            output_channels: OutputChannels::Stereo,
            sample_rate: SAMPLE_RATE,
            ir_num_samples: IR_SAMPLES,
        },
        acoustics: AcousticsConfig {
            listener_radius: 0.35,
            rays_per_query: 131_072,
            max_bounces: 1,
        },
        cache: IrCacheConfig::default(),
        auralization: AuralizationConfig { block_size: 1024 },
    }
}

fn diffraction_scene(include_edge: bool) -> SceneDescription {
    SceneDescription {
        meshes: vec![MeshAsset {
            id: 1,
            vertices: baffle_vertices(),
            indices: Vec::new(),
            opaque: true,
        }],
        materials: vec![Material {
            id: 1,
            absorption_bands: vec![1.0],
            scattering: 0.0,
            transmission: None,
        }],
        objects: vec![SceneObject {
            id: 1,
            mesh_id: 1,
            material_id: 1,
            transform: Mat4::IDENTITY,
            active: true,
        }],
        diffraction_edges: include_edge
            .then_some(DiffractionEdge {
                id: 1,
                start: vec3(-0.05, -2.0, 0.0),
                end: vec3(-0.05, 2.0, 0.0),
                bisector_dir: vec3(1.0, 0.0, 0.0),
                edge_angle_radians: 1.5 * std::f32::consts::PI,
                diffraction_radius: 0.45,
                base_strength: 0.65,
            })
            .into_iter()
            .collect(),
    }
}

fn baffle_vertices() -> Vec<Vec3> {
    let x = 0.0;
    let y0 = -2.0;
    let y1 = 2.0;
    let z0 = -3.0;
    let z1 = 0.0;
    vec![
        vec3(x, y0, z0),
        vec3(x, y1, z0),
        vec3(x, y1, z1),
        vec3(x, y0, z0),
        vec3(x, y1, z1),
        vec3(x, y0, z1),
    ]
}
