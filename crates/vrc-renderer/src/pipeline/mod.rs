use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, Not};

use vrc_schema::{AlphaMode, CullMode, MaterialDef, PhysicsShaderConfig};

use crate::model::MeshVertex;
use crate::shaders::{
    DrawUniforms, FrameUniforms, MaterialUniforms, FUR_WGSL, HIT_TEST_WGSL, OUTLINE_WGSL, TOON_WGSL,
};

/// Depth buffer format shared across main rendering and offscreen hit-testing.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Shader feature permutation flags used to key cached `wgpu::RenderPipeline` instances.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PipelineFlags(u32);

impl PipelineFlags {
    pub const NONE: Self = Self(0);
    pub const ALPHA_CLIP: Self = Self(1 << 0);
    pub const TRANSPARENT: Self = Self(1 << 1);
    pub const OUTLINE: Self = Self(1 << 2);
    pub const FUR_SHELL: Self = Self(1 << 3);
    pub const AUDIOLINK: Self = Self(1 << 4);

    /// Creates an empty flag set.
    pub const fn empty() -> Self {
        Self::NONE
    }

    /// Creates a flag set from raw bits, masking unknown bits.
    pub const fn from_bits_truncate(bits: u32) -> Self {
        let valid = Self::ALPHA_CLIP.0
            | Self::TRANSPARENT.0
            | Self::OUTLINE.0
            | Self::FUR_SHELL.0
            | Self::AUDIOLINK.0;
        Self(bits & valid)
    }

    /// Returns the raw bitmask.
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Returns `true` if all bits in `other` are set in `self`.
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Returns `true` if any bit in `other` is set in `self`.
    pub const fn intersects(self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }

    /// Inserts all bits in `other`.
    pub fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }

    /// Clears all bits in `other`.
    pub fn remove(&mut self, other: Self) {
        self.0 &= !other.0;
    }

    /// Derives base-pass [`PipelineFlags`] from a material and runtime shader toggles.
    pub fn from_material(mat: &MaterialDef, cfg: &PhysicsShaderConfig) -> Self {
        let mut flags = Self::NONE;
        match mat.alpha_mode {
            AlphaMode::Opaque => {}
            AlphaMode::Mask => flags.insert(Self::ALPHA_CLIP),
            AlphaMode::Blend => flags.insert(Self::TRANSPARENT),
        }
        if cfg.audiolink_enabled && mat.params.audiolink_enable {
            flags.insert(Self::AUDIOLINK);
        }
        flags
    }
}

impl BitOr for PipelineFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for PipelineFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitAnd for PipelineFlags {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for PipelineFlags {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl BitXor for PipelineFlags {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

impl Not for PipelineFlags {
    type Output = Self;
    fn not(self) -> Self::Output {
        Self::from_bits_truncate(!self.0)
    }
}

/// Logical rendering pass order: Opaque -> Outline -> AlphaTest/Transparent -> FurShell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderPassKind {
    Opaque,
    Outline,
    AlphaClip,
    Transparent,
    FurShell,
}

/// Unique key identifying a compiled [`wgpu::RenderPipeline`] permutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineKey {
    pub flags: PipelineFlags,
    pub cull_mode: CullMode,
}

impl PipelineKey {
    pub const fn new(flags: PipelineFlags, cull_mode: CullMode) -> Self {
        Self { flags, cull_mode }
    }
}

impl Hash for PipelineKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.flags.hash(state);
        cull_mode_tag(self.cull_mode).hash(state);
    }
}

const fn cull_mode_tag(cull: CullMode) -> u8 {
    match cull {
        CullMode::Back => 0,
        CullMode::Front => 1,
        CullMode::None => 2,
    }
}

const fn to_wgpu_cull(cull: CullMode) -> Option<wgpu::Face> {
    match cull {
        CullMode::Back => Some(wgpu::Face::Back),
        CullMode::Front => Some(wgpu::Face::Front),
        CullMode::None => None,
    }
}

/// Shared bind group layouts and pipeline layout for all avatar render passes.
#[derive(Debug)]
pub struct RenderLayouts {
    pub frame_bgl: wgpu::BindGroupLayout,
    pub draw_bgl: wgpu::BindGroupLayout,
    pub material_bgl: wgpu::BindGroupLayout,
    pub pipeline_layout: wgpu::PipelineLayout,
}

impl RenderLayouts {
    pub fn new(device: &wgpu::Device) -> Self {
        let frame_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vrc_renderer_frame_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<FrameUniforms>() as u64
                    ),
                },
                count: None,
            }],
        });

        let draw_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vrc_renderer_draw_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<DrawUniforms>() as u64,
                        ),
                    },
                    count: None,
                },
            ],
        });

        let texture_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };

        let material_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vrc_renderer_material_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<MaterialUniforms>() as u64,
                        ),
                    },
                    count: None,
                },
                texture_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                texture_entry(3),
                texture_entry(4),
                texture_entry(5),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vrc_renderer_pipeline_layout"),
            bind_group_layouts: &[Some(&frame_bgl), Some(&draw_bgl), Some(&material_bgl)],
            immediate_size: 0,
        });

        Self {
            frame_bgl,
            draw_bgl,
            material_bgl,
            pipeline_layout,
        }
    }
}

/// Lazily compiles and caches `wgpu::RenderPipeline` permutations for Toon, Outline, Fur, and HitTest passes.
#[derive(Debug)]
pub struct PipelineCache {
    layouts: RenderLayouts,
    toon_shader: wgpu::ShaderModule,
    outline_shader: wgpu::ShaderModule,
    fur_shader: wgpu::ShaderModule,
    hit_test_shader: wgpu::ShaderModule,
    pipelines: HashMap<(PipelineKey, wgpu::TextureFormat), wgpu::RenderPipeline>,
    hit_test_pipelines: HashMap<u8, wgpu::RenderPipeline>,
}

impl PipelineCache {
    /// Creates shader modules and shared bind group layouts on `device`.
    pub fn new(device: &wgpu::Device) -> Self {
        let layouts = RenderLayouts::new(device);
        let toon_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vrc_toon_shader"),
            source: wgpu::ShaderSource::Wgsl(TOON_WGSL.into()),
        });
        let outline_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vrc_outline_shader"),
            source: wgpu::ShaderSource::Wgsl(OUTLINE_WGSL.into()),
        });
        let fur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vrc_fur_shader"),
            source: wgpu::ShaderSource::Wgsl(FUR_WGSL.into()),
        });
        let hit_test_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vrc_hit_test_shader"),
            source: wgpu::ShaderSource::Wgsl(HIT_TEST_WGSL.into()),
        });

        Self {
            layouts,
            toon_shader,
            outline_shader,
            fur_shader,
            hit_test_shader,
            pipelines: HashMap::new(),
            hit_test_pipelines: HashMap::new(),
        }
    }

    pub fn layouts(&self) -> &RenderLayouts {
        &self.layouts
    }

    pub fn cached_pipeline_count(&self) -> usize {
        self.pipelines.len()
    }

    /// Returns a cached `RenderPipeline` for `(key, format)` or compiles it on demand.
    pub fn get_or_create(
        &mut self,
        device: &wgpu::Device,
        key: PipelineKey,
        format: wgpu::TextureFormat,
    ) -> &wgpu::RenderPipeline {
        if !self.pipelines.contains_key(&(key, format)) {
            let pipeline = self.compile_main_pipeline(device, key, format);
            self.pipelines.insert((key, format), pipeline);
        }
        &self.pipelines[&(key, format)]
    }

    /// Returns a cached offscreen `HitTest` pipeline for `cull_mode`.
    pub fn get_or_create_hit_test(
        &mut self,
        device: &wgpu::Device,
        cull_mode: CullMode,
    ) -> &wgpu::RenderPipeline {
        let tag = cull_mode_tag(cull_mode);
        if !self.hit_test_pipelines.contains_key(&tag) {
            let pipeline = self.compile_hit_test_pipeline(device, cull_mode);
            self.hit_test_pipelines.insert(tag, pipeline);
        }
        &self.hit_test_pipelines[&tag]
    }

    fn compile_main_pipeline(
        &self,
        device: &wgpu::Device,
        key: PipelineKey,
        format: wgpu::TextureFormat,
    ) -> wgpu::RenderPipeline {
        let is_outline = key.flags.contains(PipelineFlags::OUTLINE);
        let is_fur = key.flags.contains(PipelineFlags::FUR_SHELL);
        let is_transparent = key.flags.contains(PipelineFlags::TRANSPARENT) || is_fur;

        let shader = if is_outline {
            &self.outline_shader
        } else if is_fur {
            &self.fur_shader
        } else {
            &self.toon_shader
        };

        let cull_mode = if is_outline {
            Some(wgpu::Face::Front)
        } else {
            to_wgpu_cull(key.cull_mode)
        };

        let blend = if is_transparent {
            Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING)
        } else {
            Some(wgpu::BlendState::REPLACE)
        };

        let depth_write_enabled = !is_transparent;

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("vrc_main_pipeline"),
            layout: Some(&self.layouts.pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(MeshVertex::LAYOUT)],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(depth_write_enabled),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
    }

    fn compile_hit_test_pipeline(
        &self,
        device: &wgpu::Device,
        cull_mode: CullMode,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("vrc_hit_test_pipeline"),
            layout: Some(&self.layouts.pipeline_layout),
            vertex: wgpu::VertexState {
                module: &self.hit_test_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(MeshVertex::LAYOUT)],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: to_wgpu_cull(cull_mode),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &self.hit_test_shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
    }
}
