use bytemuck::{Pod, Zeroable};
use vrc_schema::{AlphaMode, MaterialDef, ShaderFamily};

/// Embedded WGSL source for the main Toon / PBR cel shader.
pub const TOON_WGSL: &str = include_str!("toon.wgsl");
/// Embedded WGSL source for the front-face-culled normal extrusion outline pass.
pub const OUTLINE_WGSL: &str = include_str!("outline.wgsl");
/// Embedded WGSL source for the instanced multi-layer fur shell pass.
pub const FUR_WGSL: &str = include_str!("fur.wgsl");
/// Embedded WGSL source for the offscreen alpha/node-ID hit-test pass.
pub const HIT_TEST_WGSL: &str = include_str!("hit_test.wgsl");

/// Per-frame scene and AudioLink uniform block (`@group(0) @binding(0)`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FrameUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub light_dir: [f32; 4],
    pub light_color: [f32; 4],
    pub audiolink_bands: [f32; 4],
    pub audiolink_meta: [f32; 4],
    pub head_sphere: [f32; 4],
    pub audiolink_history: [[f32; 4]; 32],
}

impl Default for FrameUniforms {
    fn default() -> Self {
        Self {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0, 0.8, 2.5, 0.0],
            light_dir: [0.45, 0.75, 0.55, 1.0],
            light_color: [1.0, 0.98, 0.96, 0.18],
            audiolink_bands: [0.0; 4],
            audiolink_meta: [0.0; 4],
            head_sphere: [0.0; 4],
            audiolink_history: [[0.0; 4]; 32],
        }
    }
}

/// Per-primitive draw uniform block (`@group(1) @binding(1)`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DrawUniforms {
    pub node_world: [[f32; 4]; 4],
    pub node_index: u32,
    pub is_skinned: u32,
    pub part_flags: u32,
    pub fur_layers: u32,
}

impl Default for DrawUniforms {
    fn default() -> Self {
        Self {
            node_world: glam::Mat4::IDENTITY.to_cols_array_2d(),
            node_index: 0,
            is_skinned: 0,
            part_flags: 0,
            fur_layers: 8,
        }
    }
}

/// Unified Toon / lilToon / Poiyomi / UTS / Fur / AudioLink material uniform block (`@group(2) @binding(0)`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MaterialUniforms {
    pub main_color: [f32; 4],
    pub shadow_color_1: [f32; 4],
    pub shadow_color_2: [f32; 4],
    pub rim_color: [f32; 4],
    pub matcap_color: [f32; 4],
    pub emission_color: [f32; 4],
    pub outline_color: [f32; 4],
    /// `[shadow_border, shadow_blur, shadow_2nd_border, shadow_2nd_blur]`
    pub shadow_params: [f32; 4],
    /// `[rim_enable, rim_border, rim_blur, rim_fresnel_power]`
    pub rim_params: [f32; 4],
    /// `[matcap_enable, matcap_blend_mode, emission_enable, alpha_cutoff]`
    pub matcap_emission: [f32; 4],
    /// `[outline_width, fur_length, fur_gravity, fur_layers]`
    pub outline_fur: [f32; 4],
    /// `[audiolink_enable, audiolink_band, audiolink_strength, shader_family]`
    pub audiolink_flags: [f32; 4],
    /// `[alpha_mode, 0.0, 0.0, 0.0]`
    pub mode_flags: [f32; 4],
}

impl Default for MaterialUniforms {
    fn default() -> Self {
        Self {
            main_color: [1.0, 1.0, 1.0, 1.0],
            shadow_color_1: [0.85, 0.85, 0.9, 1.0],
            shadow_color_2: [0.7, 0.7, 0.8, 1.0],
            rim_color: [1.0, 1.0, 1.0, 0.5],
            matcap_color: [1.0, 1.0, 1.0, 1.0],
            emission_color: [0.0, 0.0, 0.0, 1.0],
            outline_color: [0.1, 0.1, 0.1, 1.0],
            shadow_params: [0.5, 0.1, 0.2, 0.1],
            rim_params: [0.0, 0.5, 0.1, 3.0],
            matcap_emission: [0.0, 0.0, 0.0, 0.5],
            outline_fur: [0.002, 0.01, 0.25, 8.0],
            audiolink_flags: [0.0, 0.0, 1.0, 3.0],
            mode_flags: [0.0, 0.0, 0.0, 0.0],
        }
    }
}

impl MaterialUniforms {
    /// Converts a schema [`MaterialDef`] into GPU-ready uniform values.
    pub fn from_material_def(mat: &MaterialDef) -> Self {
        let p = &mat.params;
        let family = match mat.shader_family {
            ShaderFamily::LilToon => 0.0,
            ShaderFamily::Poiyomi => 1.0,
            ShaderFamily::Uts => 2.0,
            ShaderFamily::Standard => 3.0,
        };
        let alpha_mode = match mat.alpha_mode {
            AlphaMode::Opaque => 0.0,
            AlphaMode::Mask => 1.0,
            AlphaMode::Blend => 2.0,
        };
        Self {
            main_color: p.main_color,
            shadow_color_1: p.shadow_color_1,
            shadow_color_2: p.shadow_color_2,
            rim_color: p.rim_color,
            matcap_color: p.matcap_color,
            emission_color: p.emission_color,
            outline_color: p.outline_color,
            shadow_params: [
                p.shadow_border,
                p.shadow_blur,
                p.shadow_2nd_border,
                p.shadow_2nd_blur,
            ],
            rim_params: [
                if p.rim_enable { 1.0 } else { 0.0 },
                p.rim_border,
                p.rim_blur,
                p.rim_fresnel_power,
            ],
            matcap_emission: [
                if p.matcap_enable { 1.0 } else { 0.0 },
                p.matcap_blend_mode as f32,
                if p.emission_enable { 1.0 } else { 0.0 },
                p.alpha_cutoff,
            ],
            outline_fur: [
                p.outline_width,
                p.fur_length,
                p.fur_gravity,
                p.fur_layers.clamp(1, 32) as f32,
            ],
            audiolink_flags: [
                if p.audiolink_enable { 1.0 } else { 0.0 },
                p.audiolink_band as f32,
                p.audiolink_strength,
                family,
            ],
            mode_flags: [alpha_mode, 0.0, 0.0, 0.0],
        }
    }
}
