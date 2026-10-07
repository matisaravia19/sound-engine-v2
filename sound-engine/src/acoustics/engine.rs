use crate::acoustics::{
    AcousticPipeline, AcousticQuery, GpuIrSnapshot, IrCache, IrCacheQuery, IrSnapshot,
};
use crate::core::config::EngineConfig;
use crate::core::error::{SoundError, SoundResult};
use crate::engine::{ListenerPose, PointSource};
use crate::gpu::backend::VkBackend;
use crate::scene::{SceneDescription, SceneManager, SceneUpdates, SceneVersion};
use glam::Vec3;

/// Scene-backed acoustic IR engine with synchronous cache-backed queries.
///
/// The engine owns scene state, the GPU acoustic pipeline, and the IR cache.
/// Cache hits share GPU snapshots, while misses may synchronize scene
/// acceleration data and block on GPU ray tracing before returning.
pub struct AcousticsEngine {
    /// Mutable scene state and versioning used when building acoustic queries.
    scene: SceneManager,
    /// Ray-tracing pipeline that produces impulse responses for source/listener pairs.
    pipeline: AcousticPipeline,
    /// Reuse cache for impulse responses derived from scene, source, and listener state.
    cache: IrCache,
}

impl AcousticsEngine {
    /// Creates the scalar acoustic engine with an empty scene and IR cache.
    pub fn new(gpu: &VkBackend, config: EngineConfig) -> SoundResult<Self> {
        Ok(Self {
            scene: SceneManager::new(),
            pipeline: AcousticPipeline::new(gpu, config)?,
            cache: IrCache::new(config.cache),
        })
    }

    /// Replaces the scene and clears IR cache entries tied to prior geometry.
    pub fn load_scene(&mut self, scene: SceneDescription) -> SoundResult<SceneVersion> {
        let version = self.scene.load(scene)?;
        self.cache.clear();
        Ok(version)
    }

    /// Applies scene edits and clears cached IRs when scene contents changed.
    pub fn apply_scene_updates(&mut self, updates: SceneUpdates) -> SoundResult<SceneVersion> {
        let changed = !updates.updates.is_empty();
        let version = self.scene.apply_updates(updates)?;
        if changed {
            self.cache.clear();
        }
        Ok(version)
    }

    /// Builds or reuses an acoustic impulse response for a source/listener pair.
    pub fn build_impulse_response(
        &mut self,
        gpu: &VkBackend,
        source: PointSource,
        listener: ListenerPose,
    ) -> SoundResult<IrSnapshot> {
        let snapshot = self.build_impulse_response_gpu(gpu, source, listener)?;
        self.pipeline.download_gpu_ir(gpu, &snapshot)
    }

    /// Builds or reuses a GPU-resident acoustic impulse response for a source/listener pair.
    pub fn build_impulse_response_gpu(
        &mut self,
        gpu: &VkBackend,
        source: PointSource,
        listener: ListenerPose,
    ) -> SoundResult<GpuIrSnapshot> {
        self.pipeline.validate_backend(gpu)?;
        validate_source(source)?;
        let listener = canonicalize_listener(listener)?;

        let cache_query = cache_query(self.scene.version(), source, listener);
        if let Some(snapshot) = self.cache.get_ir(cache_query) {
            return Ok(snapshot);
        }

        // Building an IR can update scene-side acceleration data, so the scene
        // manager remains mutable even though the acoustic query is read-only.
        let snapshot = self.pipeline.build_ir_gpu(
            gpu,
            &mut self.scene,
            AcousticQuery {
                query_id: cache_query.query_id,
                source_position: source.position,
                listener_position: listener.position,
                listener_forward: listener.forward,
                listener_up: listener.up,
                source_amplitude: source.amplitude,
            },
        )?;
        self.cache.insert_ir(cache_query, snapshot.clone());
        Ok(snapshot)
    }
}

/// Builds the impulse-response cache key for the current acoustic endpoints.
fn cache_query(
    scene_version: SceneVersion,
    source: PointSource,
    listener: ListenerPose,
) -> IrCacheQuery {
    IrCacheQuery {
        scene_version,
        query_id: 0,
        source_position: source.position,
        source_amplitude: source.amplitude,
        listener_position: listener.position,
        listener_forward: listener.forward,
        listener_up: listener.up,
    }
}

/// Validates finite source parameters before they reach acoustic queries.
pub(crate) fn validate_source(source: PointSource) -> SoundResult<()> {
    if !source.amplitude.is_finite() || source.amplitude < 0.0 {
        return Err(SoundError::invalid_argument(
            "source amplitude must be finite and non-negative",
        ));
    }
    if !source.position.is_finite() {
        return Err(SoundError::invalid_argument(
            "source position must be finite",
        ));
    }
    Ok(())
}

/// Validates and orthonormalizes listener parameters before acoustic queries.
pub(crate) fn canonicalize_listener(listener: ListenerPose) -> SoundResult<ListenerPose> {
    if !listener.position.is_finite() {
        return Err(SoundError::invalid_argument(
            "listener position must be finite",
        ));
    }
    if !listener.forward.is_finite() || !listener.up.is_finite() {
        return Err(SoundError::invalid_argument(
            "listener forward and up vectors must be finite",
        ));
    }
    let forward = listener.forward.normalize_or_zero();
    let approximate_up = listener.up.normalize_or_zero();
    let approximate_right = forward.cross(approximate_up);
    if forward == Vec3::ZERO || approximate_right.length_squared() <= 1.0e-8 {
        return Err(SoundError::invalid_argument(
            "listener forward and up vectors must be non-zero and non-collinear",
        ));
    }
    let right = approximate_right.normalize();
    let up = right.cross(forward);
    Ok(ListenerPose {
        position: listener.position,
        forward,
        up,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};

    #[test]
    #[ignore = "requires Vulkan; run hardware tests serially"]
    fn audit_cache_hits_reject_a_foreign_backend() {
        use crate::core::config::*;
        let gpu = VkBackend::new().unwrap();
        let foreign = VkBackend::new().unwrap();
        let mut engine = AcousticsEngine::new(
            &gpu,
            EngineConfig {
                sound: SoundConfig {
                    sample_rate: 48_000,
                    output_channels: OutputChannels::Stereo,
                    ir_num_samples: 1024,
                },
                acoustics: AcousticsConfig {
                    listener_ray_count: 1,
                    max_bounces: 0,
                    maximum_path_length: 100.0,
                    minimum_throughput: 0.0,
                    sampling_seed: 0,
                },
                cache: Default::default(),
                binaural: Default::default(),
                auralization: AuralizationConfig {
                    block_size: 64,
                },
            },
        )
        .unwrap();
        let source = PointSource::new(Vec3::X);
        let listener = ListenerPose::default();
        engine
            .build_impulse_response_gpu(&gpu, source, listener)
            .unwrap();
        assert_eq!(engine.cache.len(), 1);
        assert!(
            engine
                .build_impulse_response_gpu(&foreign, source, listener)
                .is_err()
        );
        assert!(
            engine
                .build_impulse_response_gpu(&gpu, source, listener)
                .is_ok()
        );
    }

    #[test]
    fn rejects_non_finite_acoustic_query_values() {
        let listener = ListenerPose::default();
        assert!(validate_source(PointSource::new(Vec3::new(f32::INFINITY, 0.0, 0.0))).is_err());

        let mut source = PointSource::new(Vec3::ZERO);
        source.amplitude = f32::NAN;
        assert!(validate_source(source).is_err());
        source.amplitude = -0.1;
        assert!(validate_source(source).is_err());

        assert!(
            canonicalize_listener(ListenerPose {
                position: Vec3::new(0.0, f32::NAN, 0.0),
                forward: -Vec3::Z,
                up: Vec3::Y,
            })
            .is_err()
        );
        assert!(
            canonicalize_listener(ListenerPose {
                position: Vec3::ZERO,
                forward: Vec3::new(0.0, 0.0, f32::INFINITY),
                up: Vec3::Y,
            })
            .is_err()
        );
        assert!(canonicalize_listener(listener).is_ok());
        assert!(canonicalize_listener(ListenerPose::new(Vec3::ZERO, Vec3::ZERO, Vec3::Y)).is_err());
        assert!(canonicalize_listener(ListenerPose::new(Vec3::ZERO, Vec3::Y, Vec3::Y)).is_err());

        let canonical = canonicalize_listener(ListenerPose::new(
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, -2.0),
            Vec3::new(0.0, 3.0, -0.5),
        ))
        .unwrap();
        assert!((canonical.forward.length() - 1.0).abs() < 1.0e-6);
        assert!((canonical.up.length() - 1.0).abs() < 1.0e-6);
        assert!(canonical.forward.dot(canonical.up).abs() < 1.0e-6);
    }

    #[test]
    fn builds_cache_query_from_listener_and_source() {
        let listener = ListenerPose {
            position: Vec3::new(1.0, 2.0, 3.0),
            forward: -Vec3::Z,
            up: Vec3::Y,
        };
        let source = PointSource {
            position: Vec3::new(4.0, 5.0, 6.0),
            amplitude: 0.75,
        };

        let query = cache_query(9, source, listener);

        assert_eq!(query.scene_version, 9);
        assert_eq!(query.query_id, 0);
        assert_eq!(query.source_position, source.position);
        assert_eq!(query.source_amplitude, source.amplitude);
        assert_eq!(query.listener_position, listener.position);
        assert_eq!(query.listener_forward, listener.forward);
        assert_eq!(query.listener_up, listener.up);
    }

    #[test]
    fn listener_relative_lateral_direction_is_rotation_invariant() {
        let listener = canonicalize_listener(ListenerPose::default()).unwrap();
        let arrival = Vec3::new(0.7, 0.2, -0.4).normalize();
        let lateral = listener.right().dot(arrival);

        let rotation = Quat::from_rotation_y(0.8) * Quat::from_rotation_x(-0.35);
        let rotated = canonicalize_listener(ListenerPose::new(
            rotation * listener.position,
            rotation * listener.forward,
            rotation * listener.up,
        ))
        .unwrap();
        let rotated_lateral = rotated.right().dot(rotation * arrival);

        assert!((rotated_lateral - lateral).abs() < 1.0e-6);
    }
}
