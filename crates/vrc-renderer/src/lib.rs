//! `wgpu` + WGSL 3D renderer for lilToon / Poiyomi / UTS / Fur / AudioLink shaders.

use vrc_schema::{AudioLinkData, MaterialDef, VisemeWeights};

/// Uniform-compatible placeholder frame state for `vrc-renderer`.
#[derive(Debug, Clone, Default)]
pub struct RenderFrameInput {
    pub visemes: VisemeWeights,
    pub audiolink: AudioLinkData,
    pub materials: Vec<MaterialDef>,
}

/// Verifies `wgpu` and `glam` types are wired cleanly.
pub fn preferred_surface_format() -> wgpu::TextureFormat {
    let _ = glam::Mat4::IDENTITY;
    wgpu::TextureFormat::Bgra8UnormSrgb
}
