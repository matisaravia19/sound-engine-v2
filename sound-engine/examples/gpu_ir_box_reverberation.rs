use glam::{Mat4, Vec3, vec3};
use sound_engine::core::config::{
    AcousticsConfig, AuralizationConfig, EngineConfig, IrCacheConfig, OutputChannels, SoundConfig,
};
use sound_engine::core::debug::export_ir;
use sound_engine::core::error::{SoundError, SoundResult};
use sound_engine::scene::{Material, MeshAsset, SceneDescription, SceneObject};
use sound_engine::{ListenerPose, PointSource, SoundEngine};

const IR_EXPORT_PATH: &str = "target/gpu_ir_box_reverberation_ir.csv";
const SAMPLE_RATE: u32 = 48_000;
const IR_SAMPLES: u32 = SAMPLE_RATE * 2;
const BLOCK_SIZE: usize = 1024;

fn main() -> SoundResult<()> {
    let source_position = vec3(-0.65, -0.25, 0.1);
    let listener_position = vec3(0.55, 0.35, -0.05);
    let listener = ListenerPose {
        position: listener_position,
        right: vec3(1.0, 0.0, 0.0),
    };
    let source = PointSource {
        position: source_position,
        energy: 10_000.0,
    };

    let mut engine = SoundEngine::new(engine_config())?;
    engine.load_scene(box_room_scene())?;
    let ir = engine.build_impulse_response(source, listener)?;

    println!(
        "gpu box reverberation: version={}, query={}, samples={}, energy={}",
        ir.scene_version,
        ir.query_id,
        ir.samples.len(),
        ir.energy
    );
    export_ir(&ir, IR_EXPORT_PATH)?;
    println!("  exported ir: {IR_EXPORT_PATH}");

    let direct_center_sample =
        ((listener_position - source_position).length() / 343.0 * SAMPLE_RATE as f32).round() as usize;
    let late_start = direct_center_sample + 100;
    let mut nonzero = 0;
    let mut late_nonzero = 0;
    let mut first_late = None;
    let mut strongest_late = (0_usize, 0.0_f32, 0.0_f32);

    for (sample_idx, sample) in ir.samples.iter().enumerate() {
        if sample.left == 0.0 && sample.right == 0.0 {
            continue;
        }

        nonzero += 1;
        if sample_idx >= late_start {
            late_nonzero += 1;
            first_late.get_or_insert(sample_idx);

            let amplitude = sample.left.abs().max(sample.right.abs());
            if amplitude > strongest_late.1.abs().max(strongest_late.2.abs()) {
                strongest_late = (sample_idx, sample.left, sample.right);
            }
        }
    }

    println!("  nonzero samples: {nonzero}");
    println!("  late nonzero samples: {late_nonzero}");
    if let Some(sample_idx) = first_late {
        println!(
            "  first late sample {}: time={}s",
            sample_idx,
            sample_idx as f32 / SAMPLE_RATE as f32
        );
    }
    println!(
        "  strongest late sample {}: left={}, right={}",
        strongest_late.0, strongest_late.1, strongest_late.2
    );

    if nonzero == 0 {
        return Err(SoundError::invalid_state("Expected direct listener hits"));
    }
    if late_nonzero == 0 {
        return Err(SoundError::invalid_state("Expected reverberant listener hits"));
    }

    Ok(())
}

fn engine_config() -> EngineConfig {
    EngineConfig {
        sound: SoundConfig {
            output_channels: OutputChannels::Stereo,
            sample_rate: SAMPLE_RATE,
            ir_num_samples: IR_SAMPLES,
        },
        acoustics: AcousticsConfig {
            listener_half_extent: vec3(0.25, 0.25, 0.25),
            rays_per_query: 1_000_000,
            max_bounces: 16,
        },
        cache: IrCacheConfig::default(),
        auralization: AuralizationConfig { block_size: BLOCK_SIZE },
    }
}

fn box_room_scene() -> SceneDescription {
    SceneDescription {
        meshes: vec![MeshAsset {
            id: 1,
            vertices: box_room_vertices(vec3(10.0, 2.4, 20.0)),
            indices: Vec::new(),
            opaque: true,
        }],
        materials: vec![Material {
            id: 1,
            absorption_bands: vec![0.02],
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
    }
}

fn box_room_vertices(half_extent: Vec3) -> Vec<Vec3> {
    let min = -half_extent;
    let max = half_extent;
    let corners = [
        vec3(min.x, min.y, min.z),
        vec3(max.x, min.y, min.z),
        vec3(max.x, max.y, min.z),
        vec3(min.x, max.y, min.z),
        vec3(min.x, min.y, max.z),
        vec3(max.x, min.y, max.z),
        vec3(max.x, max.y, max.z),
        vec3(min.x, max.y, max.z),
    ];

    let faces = [
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [3, 7, 6, 2],
        [0, 4, 7, 3],
        [1, 2, 6, 5],
    ];

    let mut vertices = Vec::with_capacity(faces.len() * 6);
    for [a, b, c, d] in faces {
        vertices.extend_from_slice(&[corners[a], corners[b], corners[c], corners[a], corners[c], corners[d]]);
    }
    vertices
}
