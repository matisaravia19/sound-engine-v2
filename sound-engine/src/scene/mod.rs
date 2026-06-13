//! Scene ownership and GPU synchronization.
//!
//! The scene module owns the canonical CPU-side scene and keeps a lazy GPU
//! mirror for ray tracing. Callers mutate `SceneManager`, then request GPU
//! resources through `sync_gpu_if_needed`.

mod gpu;
mod manager;
mod store;
mod transform;
mod types;

pub use gpu::GpuSceneResources;
pub use manager::SceneManager;
pub use store::SceneStore;
pub use transform::{identity_transform, translation_transform};
pub use types::{
    Material, MaterialId, MeshAsset, MeshId, ObjectId, SceneDescription, SceneObject, SceneUpdate, SceneUpdates,
    SceneVersion,
};
