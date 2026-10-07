use super::*;
use crate::{ListenerPose, PointSource};
use glam::Vec3;

#[test]
fn layout_requires_nonzero_rate_and_samples_within_the_allocation() {
    assert!(validate_layout(8, 48_000, 1).is_ok());
    assert!(validate_layout(16, 48_000, 1).is_ok());
    assert!(validate_layout(8, 0, 1).is_err());
    assert!(validate_layout(8, 48_000, 0).is_err());
    assert!(validate_layout(7, 48_000, 1).is_err());
    assert!(validate_layout(8, 48_000, u32::MAX).is_err());
    assert!(validate_layout(u64::from(u32::MAX) * 8, 48_000, u32::MAX).is_ok());
}

#[test]
#[ignore = "requires Vulkan; run hardware tests serially"]
fn snapshots_validate_provenance_and_read_back_only_declared_samples() {
    let gpu = VkBackend::new().unwrap();
    let buffer = Arc::new(gpu.memory().create_storage_buffer(24).unwrap());
    let samples = [
        IrSample::new(1.0, -2.0),
        IrSample::new(0.5, 0.25),
        IrSample::new(99.0, 99.0),
    ];
    gpu.memory().upload_typed(&buffer, &samples).unwrap();
    let query = AcousticQuery {
        query_id: 7,
        source: PointSource::new(Vec3::X),
        listener: ListenerPose::new(Vec3::ZERO, Vec3::Z * 2.0, Vec3::Y * 3.0),
    };
    let snapshot = GpuIrSnapshot::new(buffer.clone(), 48_000, 2, 3, query).unwrap();
    assert_eq!(snapshot.byte_size(), 24);
    assert_eq!(snapshot.sample_count(), 2);
    assert_eq!(snapshot.sample_rate(), 48_000);
    assert_eq!(snapshot.query().listener.forward, Vec3::Z);
    assert_eq!(snapshot.query().listener.up, Vec3::Y);
    let downloaded = snapshot.download(&gpu).unwrap();
    assert_eq!(downloaded.samples.len(), 2);
    assert_eq!(downloaded.energy, 5.3125);
    assert_eq!(downloaded.sample_rate, 48_000);
    assert_eq!(downloaded.scene_version, 3);
    assert_eq!(downloaded.query_id, 7);

    assert!(GpuIrSnapshot::new(buffer.clone(), 0, 2, 3, query).is_err());
    assert!(GpuIrSnapshot::new(buffer.clone(), 48_000, 4, 3, query).is_err());
    let mut invalid = query;
    invalid.listener.forward = Vec3::ZERO;
    assert!(GpuIrSnapshot::new(buffer, 48_000, 2, 3, invalid).is_err());
}
