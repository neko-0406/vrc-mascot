//! `wgpu` + WGSL 3D renderer for lilToon / Poiyomi / UTS / Fur / AudioLink shaders.

pub mod audiolink;
pub mod error;
pub mod model;
pub mod offscreen;
pub mod pipeline;
pub mod renderer;
pub mod shaders;

pub use audiolink::{AudioLinkAnalyzer, AudioLinkCapture, FFT_HOP, FFT_SIZE};
pub use error::RendererError;
pub use model::{
    AvatarModel, CameraRig, MeshPrimitiveData, MeshVertex, ModelNode, MorphTargetData, SkinData,
    TextureData,
};
pub use offscreen::{HitTestPass, HitTestResult, HIT_TEST_ALPHA_THRESHOLD};
pub use pipeline::{PipelineCache, PipelineFlags, PipelineKey, RenderPassKind, DEPTH_FORMAT};
pub use renderer::MascotRenderer;
pub use shaders::{
    DrawUniforms, FrameUniforms, MaterialUniforms, FUR_WGSL, HIT_TEST_WGSL, OUTLINE_WGSL, TOON_WGSL,
};

use vrc_schema::{AudioLinkData, MaterialDef, VisemeWeights};

/// Uniform-compatible placeholder frame state for `vrc-renderer`.
#[derive(Debug, Clone, Default)]
pub struct RenderFrameInput {
    pub visemes: VisemeWeights,
    pub audiolink: AudioLinkData,
    pub materials: Vec<MaterialDef>,
}

/// Returns the preferred BGRA sRGB surface format for Win32 transparent DWM compositing.
pub fn preferred_surface_format() -> wgpu::TextureFormat {
    wgpu::TextureFormat::Bgra8UnormSrgb
}
