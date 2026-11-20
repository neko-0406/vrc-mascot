use std::collections::HashMap;

use glam::{Mat4, Vec3};
use vrc_schema::{
    AlphaMode, AudioLinkData, CullMode, MascotManifest, MaterialDef, PhysicsShaderConfig,
    QualityPreset,
};

use crate::model::{AvatarModel, CameraRig, MeshVertex, TextureData};
use crate::offscreen::{HitTestPass, HitTestResult};
use crate::pipeline::{PipelineCache, PipelineFlags, PipelineKey, DEPTH_FORMAT};
use crate::shaders::{DrawUniforms, FrameUniforms, MaterialUniforms};

const PART_FLAG_HEAD: u32 = 1 << 0;

#[derive(Debug)]
struct GpuTexture {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

#[derive(Debug)]
struct GpuSkin {
    buffer: wgpu::Buffer,
    joint_indices: Vec<usize>,
    inverse_bind_matrices: Vec<Mat4>,
}

#[derive(Debug)]
struct GpuMaterial {
    def: MaterialDef,
    _uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

#[derive(Debug)]
struct GpuPrimitive {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    material_index: Option<usize>,
    node_index: usize,
    node_name: String,
    node_path: String,
    skin_index: Option<usize>,
    local_center: Vec3,
    part_flags: u32,
    visible: bool,
    draw_uniform_buffer: wgpu::Buffer,
    draw_bind_group: wgpu::BindGroup,
}

/// Real-time `wgpu` 3D avatar renderer supporting Toon/lilToon/Poiyomi/UTS, Outline,
/// Fur shells, AudioLink, morph target blendshapes, and non-stalling offscreen hit-testing.
pub struct MascotRenderer {
    surface_format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    pipeline_cache: PipelineCache,
    hit_test_pass: HitTestPass,
    main_depth_texture: wgpu::Texture,
    main_depth_view: wgpu::TextureView,
    frame_uniform_buffer: wgpu::Buffer,
    frame_bind_group: wgpu::BindGroup,
    frame_uniforms: FrameUniforms,
    default_white_srgb: GpuTexture,
    default_white_linear: GpuTexture,
    default_sampler: wgpu::Sampler,
    identity_skin_buffer: wgpu::Buffer,
    camera: CameraRig,
    model: Option<AvatarModel>,
    manifest: Option<MascotManifest>,
    head_node_index: Option<usize>,
    head_radius: f32,
    current_world_matrices: Vec<Mat4>,
    _gpu_textures: Vec<GpuTexture>,
    gpu_skins: Vec<GpuSkin>,
    gpu_materials: Vec<GpuMaterial>,
    default_material: GpuMaterial,
    gpu_primitives: Vec<GpuPrimitive>,
    morph_scratch: Vec<MeshVertex>,
    skin_scratch: Vec<[[f32; 4]; 4]>,
    last_blendshape_weights: HashMap<String, f32>,
}

impl MascotRenderer {
    /// Initializes GPU resources, shader modules, offscreen hit-test pass, and default fallback material.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let w = width.max(1);
        let h = height.max(1);
        let pipeline_cache = PipelineCache::new(device);
        let hit_test_pass = HitTestPass::new(device, w, h);
        let (main_depth_texture, main_depth_view) = create_depth_texture(device, w, h);

        let frame_uniforms = FrameUniforms::default();
        let frame_uniform_buffer = create_buffer_init(
            device,
            "vrc_frame_uniforms",
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            bytemuck::bytes_of(&frame_uniforms),
        );
        let frame_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vrc_frame_bg"),
            layout: &pipeline_cache.layouts().frame_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame_uniform_buffer.as_entire_binding(),
            }],
        });

        let default_white_srgb = upload_texture(
            device,
            queue,
            &TextureData {
                name: "default_white_srgb".to_owned(),
                width: 1,
                height: 1,
                rgba8: vec![255, 255, 255, 255],
                srgb: true,
            },
        );
        let default_white_linear = upload_texture(
            device,
            queue,
            &TextureData {
                name: "default_white_linear".to_owned(),
                width: 1,
                height: 1,
                rgba8: vec![255, 255, 255, 255],
                srgb: false,
            },
        );
        let default_black_linear = upload_texture(
            device,
            queue,
            &TextureData {
                name: "default_black_linear".to_owned(),
                width: 1,
                height: 1,
                rgba8: vec![0, 0, 0, 255],
                srgb: false,
            },
        );

        let default_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("vrc_default_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let identity_mat = Mat4::IDENTITY.to_cols_array_2d();
        let identity_skin_buffer = create_buffer_init(
            device,
            "vrc_identity_skin",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            bytemuck::bytes_of(&identity_mat),
        );

        let default_mat_def = MaterialDef {
            gltf_material_index: 0,
            name: "DefaultToon".to_owned(),
            shader_family: vrc_schema::ShaderFamily::Standard,
            render_queue: 2000,
            alpha_mode: AlphaMode::Opaque,
            params: vrc_schema::MaterialParams::default(),
        };
        let default_material = build_gpu_material(
            device,
            &pipeline_cache.layouts().material_bgl,
            &default_mat_def,
            &default_white_srgb.view,
            &default_white_linear.view,
            &default_black_linear.view,
            &default_white_linear.view,
            &default_sampler,
        );

        Self {
            surface_format,
            width: w,
            height: h,
            pipeline_cache,
            hit_test_pass,
            main_depth_texture,
            main_depth_view,
            frame_uniform_buffer,
            frame_bind_group,
            frame_uniforms,
            default_white_srgb,
            default_white_linear,
            default_sampler,
            identity_skin_buffer,
            camera: CameraRig::default(),
            model: None,
            manifest: None,
            head_node_index: None,
            head_radius: 0.22,
            current_world_matrices: Vec::new(),
            _gpu_textures: Vec::new(),
            gpu_skins: Vec::new(),
            gpu_materials: Vec::new(),
            default_material,
            gpu_primitives: Vec::new(),
            morph_scratch: Vec::new(),
            skin_scratch: Vec::new(),
            last_blendshape_weights: HashMap::new(),
        }
    }

    pub fn camera(&self) -> &CameraRig {
        &self.camera
    }

    pub fn camera_mut(&mut self) -> &mut CameraRig {
        &mut self.camera
    }

    pub fn model(&self) -> Option<&AvatarModel> {
        self.model.as_ref()
    }

    /// Uploads model meshes, skins, textures, and manifest materials to GPU and auto-frames camera.
    pub fn set_model(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: AvatarModel,
        manifest: &MascotManifest,
    ) {
        self.camera =
            CameraRig::for_bounds(model.bounds_min, model.bounds_max, self.width, self.height);

        let rest_world = model.compute_rest_world_matrices();
        let head_node_index = manifest
            .humanoid_bones
            .get("Head")
            .and_then(|path| model.find_node_index(path))
            .or_else(|| model.find_node_index("Head"));
        let model_height = (model.bounds_max.y - model.bounds_min.y).max(0.3);
        self.head_node_index = head_node_index;
        self.head_radius = (model_height * 0.16).clamp(0.12, 0.40);
        self.current_world_matrices = rest_world.clone();
        self.last_blendshape_weights.clear();

        let gpu_textures: Vec<GpuTexture> = model
            .textures
            .iter()
            .map(|tex| upload_texture(device, queue, tex))
            .collect();

        let gpu_skins: Vec<GpuSkin> = model
            .skins
            .iter()
            .map(|skin| {
                let mut mats = Vec::with_capacity(skin.joints.len().max(1));
                for (j_idx, &node_idx) in skin.joints.iter().enumerate() {
                    let world = rest_world.get(node_idx).copied().unwrap_or(Mat4::IDENTITY);
                    let ibm = skin
                        .inverse_bind_matrices
                        .get(j_idx)
                        .copied()
                        .unwrap_or(Mat4::IDENTITY);
                    mats.push((world * ibm).to_cols_array_2d());
                }
                if mats.is_empty() {
                    mats.push(Mat4::IDENTITY.to_cols_array_2d());
                }
                let buffer = create_buffer_init(
                    device,
                    &format!("vrc_skin_{}", skin.name),
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    bytemuck::cast_slice(&mats),
                );
                GpuSkin {
                    buffer,
                    joint_indices: skin.joints.clone(),
                    inverse_bind_matrices: skin.inverse_bind_matrices.clone(),
                }
            })
            .collect();

        let max_mat_idx = manifest
            .materials
            .iter()
            .map(|m| m.gltf_material_index + 1)
            .max()
            .unwrap_or(0)
            .max(model.material_base_textures.len());

        let mut gpu_materials = Vec::with_capacity(max_mat_idx);
        for idx in 0..max_mat_idx {
            let def = manifest
                .materials
                .iter()
                .find(|m| m.gltf_material_index == idx)
                .cloned()
                .unwrap_or_else(|| MaterialDef {
                    gltf_material_index: idx,
                    name: format!("Material_{idx}"),
                    shader_family: vrc_schema::ShaderFamily::Standard,
                    render_queue: 2000,
                    alpha_mode: AlphaMode::Opaque,
                    params: vrc_schema::MaterialParams::default(),
                });

            let base_view = model
                .material_base_textures
                .get(idx)
                .and_then(|opt| *opt)
                .and_then(|tex_idx| gpu_textures.get(tex_idx))
                .map(|t| &t.view)
                .unwrap_or(&self.default_white_srgb.view);
            let matcap_view = def
                .params
                .matcap_texture_index
                .and_then(|tex_idx| gpu_textures.get(tex_idx))
                .map(|t| &t.view)
                .unwrap_or(&self.default_white_linear.view);
            let emission_view = def
                .params
                .emission_texture_index
                .and_then(|tex_idx| gpu_textures.get(tex_idx))
                .map(|t| &t.view)
                .unwrap_or(&self.default_white_srgb.view);
            let fur_mask_view = def
                .params
                .fur_mask_texture_index
                .and_then(|tex_idx| gpu_textures.get(tex_idx))
                .map(|t| &t.view)
                .unwrap_or(&self.default_white_linear.view);

            gpu_materials.push(build_gpu_material(
                device,
                &self.pipeline_cache.layouts().material_bgl,
                &def,
                base_view,
                matcap_view,
                emission_view,
                fur_mask_view,
                &self.default_sampler,
            ));
        }

        let mut gpu_primitives = Vec::with_capacity(model.primitives.len());
        for (prim_idx, prim) in model.primitives.iter().enumerate() {
            let vbytes: &[u8] = if prim.vertices.is_empty() {
                &[0u8; 4]
            } else {
                bytemuck::cast_slice(&prim.vertices)
            };
            let ibytes: &[u8] = if prim.indices.is_empty() {
                &[0u8; 4]
            } else {
                bytemuck::cast_slice(&prim.indices)
            };

            let vertex_buffer = create_buffer_init(
                device,
                &format!("vrc_prim_vb_{prim_idx}"),
                wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                vbytes,
            );
            let index_buffer = create_buffer_init(
                device,
                &format!("vrc_prim_ib_{prim_idx}"),
                wgpu::BufferUsages::INDEX,
                ibytes,
            );

            let is_head = model.is_head_or_descendant(prim.node_index, head_node_index);
            let part_flags = if is_head { PART_FLAG_HEAD } else { 0 };
            let node_world = rest_world
                .get(prim.node_index)
                .copied()
                .unwrap_or(Mat4::IDENTITY);
            let draw_uniform = DrawUniforms {
                node_world: node_world.to_cols_array_2d(),
                node_index: prim.node_index as u32,
                is_skinned: if prim.skin_index.is_some() { 1 } else { 0 },
                part_flags,
                fur_layers: 8,
            };
            let draw_uniform_buffer = create_buffer_init(
                device,
                &format!("vrc_draw_u_{prim_idx}"),
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                bytemuck::bytes_of(&draw_uniform),
            );
            let skin_buffer = prim
                .skin_index
                .and_then(|s_idx| gpu_skins.get(s_idx))
                .map(|s| &s.buffer)
                .unwrap_or(&self.identity_skin_buffer);

            let draw_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("vrc_draw_bg_{prim_idx}")),
                layout: &self.pipeline_cache.layouts().draw_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: skin_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: draw_uniform_buffer.as_entire_binding(),
                    },
                ],
            });

            gpu_primitives.push(GpuPrimitive {
                vertex_buffer,
                index_buffer,
                index_count: prim.indices.len() as u32,
                material_index: prim.material_index,
                node_index: prim.node_index,
                node_name: prim.node_name.clone(),
                node_path: model.node_path(prim.node_index),
                skin_index: prim.skin_index,
                local_center: prim.bounding_center(),
                part_flags,
                visible: true,
                draw_uniform_buffer,
                draw_bind_group,
            });
        }

        self.model = Some(model);
        self.manifest = Some(manifest.clone());
        self._gpu_textures = gpu_textures;
        self.gpu_skins = gpu_skins;
        self.gpu_materials = gpu_materials;
        self.gpu_primitives = gpu_primitives;
    }

    /// Resizes main depth target, offscreen hit-test buffer, and camera viewport aspect.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let w = width.max(1);
        let h = height.max(1);
        if w == self.width && h == self.height {
            return;
        }
        self.width = w;
        self.height = h;
        let (depth_tex, depth_view) = create_depth_texture(device, w, h);
        self.main_depth_texture = depth_tex;
        self.main_depth_view = depth_view;
        self.hit_test_pass.resize(device, w, h);
        self.camera.set_viewport(w, h);
    }

    /// Updates skeletal skinning matrices, morph target blendshapes, and node visibility.
    pub fn update_pose(
        &mut self,
        queue: &wgpu::Queue,
        world_matrices: &[Mat4],
        blendshape_weights: &HashMap<String, f32>,
        node_visibility: &HashMap<String, bool>,
    ) {
        let Some(model) = &self.model else {
            return;
        };

        if !world_matrices.is_empty() {
            if self.current_world_matrices.len() != model.nodes.len() {
                self.current_world_matrices = model.compute_rest_world_matrices();
            }
            let copy_len = world_matrices.len().min(self.current_world_matrices.len());
            self.current_world_matrices[..copy_len].copy_from_slice(&world_matrices[..copy_len]);
        } else if self.current_world_matrices.is_empty() {
            self.current_world_matrices = model.compute_rest_world_matrices();
        }

        for skin in &self.gpu_skins {
            self.skin_scratch.clear();
            for (j_idx, &node_idx) in skin.joint_indices.iter().enumerate() {
                let world = self
                    .current_world_matrices
                    .get(node_idx)
                    .copied()
                    .unwrap_or(Mat4::IDENTITY);
                let ibm = skin
                    .inverse_bind_matrices
                    .get(j_idx)
                    .copied()
                    .unwrap_or(Mat4::IDENTITY);
                self.skin_scratch.push((world * ibm).to_cols_array_2d());
            }
            if !self.skin_scratch.is_empty() {
                queue.write_buffer(&skin.buffer, 0, bytemuck::cast_slice(&self.skin_scratch));
            }
        }

        let weights_changed = *blendshape_weights != self.last_blendshape_weights;
        if weights_changed {
            self.last_blendshape_weights = blendshape_weights.clone();
        }

        for (prim_idx, gpu_prim) in self.gpu_primitives.iter_mut().enumerate() {
            gpu_prim.visible = resolve_node_visibility(
                model,
                gpu_prim.node_index,
                &gpu_prim.node_name,
                &gpu_prim.node_path,
                node_visibility,
            );

            let node_world = self
                .current_world_matrices
                .get(gpu_prim.node_index)
                .copied()
                .unwrap_or(Mat4::IDENTITY);
            let draw_uniform = DrawUniforms {
                node_world: node_world.to_cols_array_2d(),
                node_index: gpu_prim.node_index as u32,
                is_skinned: if gpu_prim.skin_index.is_some() { 1 } else { 0 },
                part_flags: gpu_prim.part_flags,
                fur_layers: 8,
            };
            queue.write_buffer(
                &gpu_prim.draw_uniform_buffer,
                0,
                bytemuck::bytes_of(&draw_uniform),
            );

            if weights_changed {
                if let Some(cpu_prim) = model.primitives.get(prim_idx) {
                    if !cpu_prim.morph_targets.is_empty() {
                        let weights: Vec<f32> = cpu_prim
                            .morph_targets
                            .iter()
                            .map(|mt| blendshape_weights.get(&mt.name).copied().unwrap_or(0.0))
                            .collect();
                        cpu_prim.evaluate_morph_targets(&weights, &mut self.morph_scratch);
                        if !self.morph_scratch.is_empty() {
                            queue.write_buffer(
                                &gpu_prim.vertex_buffer,
                                0,
                                bytemuck::cast_slice(&self.morph_scratch),
                            );
                        }
                    }
                }
            }
        }
    }

    /// Uploads current AudioLink 4-band FFT amplitudes and 32-frame history to the GPU.
    pub fn update_audiolink(&mut self, queue: &wgpu::Queue, data: &AudioLinkData) {
        self.frame_uniforms.audiolink_bands = data.bands;
        let avg_energy = (data.bands[0] + data.bands[1] + data.bands[2] + data.bands[3]) * 0.25;
        self.frame_uniforms.audiolink_meta =
            [if data.active { 1.0 } else { 0.0 }, avg_energy, 0.0, 0.0];
        self.frame_uniforms.audiolink_history = data.history;
        queue.write_buffer(
            &self.frame_uniform_buffer,
            0,
            bytemuck::bytes_of(&self.frame_uniforms),
        );
    }

    /// Encodes the offscreen hit-test pass and the 4-pass main toon render into `encoder`.
    ///
    /// Pass order:
    /// 1. Opaque Toon
    /// 2. Outline (`Front` cull, when `cfg.outline_enabled`)
    /// 3. AlphaTest / Z-sorted Transparent
    /// 4. Fur Shell instanced (when `cfg.fur_enabled`)
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        cfg: &PhysicsShaderConfig,
    ) {
        self.frame_uniforms.view_proj = self.camera.view_proj_matrix().to_cols_array_2d();
        self.frame_uniforms.camera_pos =
            [self.camera.eye.x, self.camera.eye.y, self.camera.eye.z, 0.0];

        if let Some(head_idx) = self.head_node_index {
            let head_pos = self
                .current_world_matrices
                .get(head_idx)
                .map(|m| m.transform_point3(Vec3::new(0.0, 0.10, 0.0)))
                .unwrap_or(Vec3::new(0.0, 1.15, 0.0));
            self.frame_uniforms.head_sphere =
                [head_pos.x, head_pos.y, head_pos.z, self.head_radius];
        } else {
            self.frame_uniforms.head_sphere = [0.0; 4];
        }

        queue.write_buffer(
            &self.frame_uniform_buffer,
            0,
            bytemuck::bytes_of(&self.frame_uniforms),
        );

        let quality_max_fur = match cfg.quality_preset {
            QualityPreset::Low => 4,
            QualityPreset::Medium => 8,
            QualityPreset::High => 16,
        };

        // Bucket visible primitives for the 4 ordered passes and pre-compile pipelines.
        let mut opaque_list: Vec<(usize, PipelineKey)> = Vec::new();
        let mut outline_list: Vec<(usize, PipelineKey)> = Vec::new();
        let mut alpha_clip_list: Vec<(usize, PipelineKey)> = Vec::new();
        let mut transparent_list: Vec<(usize, PipelineKey, f32)> = Vec::new();
        let mut fur_list: Vec<(usize, PipelineKey, u32)> = Vec::new();
        let mut hit_test_list: Vec<(usize, CullMode)> = Vec::new();

        for (idx, prim) in self.gpu_primitives.iter().enumerate() {
            if !prim.visible || prim.index_count == 0 {
                continue;
            }
            let mat_def = prim
                .material_index
                .and_then(|mi| self.gpu_materials.get(mi))
                .map(|m| &m.def)
                .unwrap_or(&self.default_material.def);

            let cull = mat_def.params.cull_mode;
            hit_test_list.push((idx, cull));
            let _ = self.pipeline_cache.get_or_create_hit_test(device, cull);

            let base_flags = PipelineFlags::from_material(mat_def, cfg);
            let base_key = PipelineKey::new(base_flags, cull);
            let _ = self
                .pipeline_cache
                .get_or_create(device, base_key, self.surface_format);

            match mat_def.alpha_mode {
                AlphaMode::Opaque => opaque_list.push((idx, base_key)),
                AlphaMode::Mask => alpha_clip_list.push((idx, base_key)),
                AlphaMode::Blend => {
                    let world_pos = self
                        .current_world_matrices
                        .get(prim.node_index)
                        .map(|m| m.transform_point3(prim.local_center))
                        .unwrap_or(prim.local_center);
                    let dist_sq = world_pos.distance_squared(self.camera.eye);
                    transparent_list.push((idx, base_key, dist_sq));
                }
            }

            if cfg.outline_enabled
                && mat_def.params.outline_enable
                && mat_def.alpha_mode != AlphaMode::Blend
            {
                let mut outline_flags = base_flags;
                outline_flags.insert(PipelineFlags::OUTLINE);
                let outline_key = PipelineKey::new(outline_flags, CullMode::Front);
                let _ = self
                    .pipeline_cache
                    .get_or_create(device, outline_key, self.surface_format);
                outline_list.push((idx, outline_key));
            }

            if cfg.fur_enabled && mat_def.params.fur_enable && mat_def.params.fur_layers > 0 {
                let mut fur_flags = base_flags;
                fur_flags.insert(PipelineFlags::FUR_SHELL);
                let fur_key = PipelineKey::new(fur_flags, cull);
                let _ = self
                    .pipeline_cache
                    .get_or_create(device, fur_key, self.surface_format);
                let layers = mat_def.params.fur_layers.clamp(1, quality_max_fur);
                fur_list.push((idx, fur_key, layers));
            }
        }

        // Sort transparent primitives back-to-front (largest distance to eye first).
        transparent_list.sort_by(|a, b| b.2.total_cmp(&a.2));

        // 0. Offscreen Hit-Test Pass (for per-pixel click-through & head picking)
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vrc_hit_test_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: self.hit_test_pass.color_view(),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: self.hit_test_pass.depth_view(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            pass.set_bind_group(0, &self.frame_bind_group, &[]);
            for &(prim_idx, cull) in &hit_test_list {
                let prim = &self.gpu_primitives[prim_idx];
                let mat = prim
                    .material_index
                    .and_then(|mi| self.gpu_materials.get(mi))
                    .unwrap_or(&self.default_material);
                let pipeline = self.pipeline_cache.get_or_create_hit_test(device, cull);
                pass.set_pipeline(pipeline);
                pass.set_bind_group(1, &prim.draw_bind_group, &[]);
                pass.set_bind_group(2, &mat.bind_group, &[]);
                pass.set_vertex_buffer(0, prim.vertex_buffer.slice(..));
                pass.set_index_buffer(prim.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..prim.index_count, 0, 0..1);
            }
        }
        self.hit_test_pass.encode_copy_to_staging(encoder);

        // Main 4-Pass Forward Toon Render into transparent swapchain target
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vrc_main_toon_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.main_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            pass.set_bind_group(0, &self.frame_bind_group, &[]);

            // Pass 1: Opaque Toon
            for &(prim_idx, key) in &opaque_list {
                draw_prim(
                    &mut pass,
                    device,
                    &mut self.pipeline_cache,
                    &self.gpu_primitives[prim_idx],
                    &self.gpu_materials,
                    &self.default_material,
                    key,
                    self.surface_format,
                    1,
                );
            }

            // Pass 2: Outline (Front cull)
            for &(prim_idx, key) in &outline_list {
                draw_prim(
                    &mut pass,
                    device,
                    &mut self.pipeline_cache,
                    &self.gpu_primitives[prim_idx],
                    &self.gpu_materials,
                    &self.default_material,
                    key,
                    self.surface_format,
                    1,
                );
            }

            // Pass 3: AlphaTest -> Z-sorted Transparent
            for &(prim_idx, key) in &alpha_clip_list {
                draw_prim(
                    &mut pass,
                    device,
                    &mut self.pipeline_cache,
                    &self.gpu_primitives[prim_idx],
                    &self.gpu_materials,
                    &self.default_material,
                    key,
                    self.surface_format,
                    1,
                );
            }
            for &(prim_idx, key, _) in &transparent_list {
                draw_prim(
                    &mut pass,
                    device,
                    &mut self.pipeline_cache,
                    &self.gpu_primitives[prim_idx],
                    &self.gpu_materials,
                    &self.default_material,
                    key,
                    self.surface_format,
                    1,
                );
            }

            // Pass 4: Fur Shell (instanced)
            for &(prim_idx, key, layers) in &fur_list {
                draw_prim(
                    &mut pass,
                    device,
                    &mut self.pipeline_cache,
                    &self.gpu_primitives[prim_idx],
                    &self.gpu_materials,
                    &self.default_material,
                    key,
                    self.surface_format,
                    layers,
                );
            }
        }
    }

    /// Advances non-blocking staging buffer readback for offscreen hit-testing.
    pub fn poll_hit_test_readback(&mut self, device: &wgpu::Device) {
        self.hit_test_pass.poll_readback(device);
    }

    /// Queries the cached offscreen hit-test pixel at `(x, y)` in O(1) without GPU stalls.
    pub fn hit_test(&self, x: u32, y: u32) -> HitTestResult {
        self.hit_test_pass.hit_test(x, y)
    }

    /// Projects a skeleton node (by name or path) into 2D screen coordinates.
    pub fn project_node_screen(
        &self,
        node_name: &str,
        world_matrices: &[Mat4],
    ) -> Option<[f32; 2]> {
        let model = self.model.as_ref()?;
        let node_idx = model.find_node_index(node_name)?;
        let mat = world_matrices
            .get(node_idx)
            .or_else(|| self.current_world_matrices.get(node_idx))
            .copied()
            .unwrap_or(Mat4::IDENTITY);
        let world_pos = mat.transform_point3(Vec3::ZERO);
        self.camera
            .project_world_point(world_pos, self.width, self.height)
    }

    /// Casts a ray from `(screen_x, screen_y)` onto the world plane `z = plane_z`.
    pub fn screen_ray_to_world_plane(&self, screen_x: f32, screen_y: f32, plane_z: f32) -> Vec3 {
        self.camera
            .screen_ray_to_world_plane(screen_x, screen_y, self.width, self.height, plane_z)
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_prim<'a>(
    pass: &mut wgpu::RenderPass<'a>,
    device: &wgpu::Device,
    pipeline_cache: &mut PipelineCache,
    prim: &'a GpuPrimitive,
    materials: &'a [GpuMaterial],
    default_material: &'a GpuMaterial,
    key: PipelineKey,
    format: wgpu::TextureFormat,
    instances: u32,
) {
    let mat = prim
        .material_index
        .and_then(|mi| materials.get(mi))
        .unwrap_or(default_material);
    let pipeline = pipeline_cache.get_or_create(device, key, format);
    pass.set_pipeline(pipeline);
    pass.set_bind_group(1, &prim.draw_bind_group, &[]);
    pass.set_bind_group(2, &mat.bind_group, &[]);
    pass.set_vertex_buffer(0, prim.vertex_buffer.slice(..));
    pass.set_index_buffer(prim.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
    pass.draw_indexed(0..prim.index_count, 0, 0..instances);
}

fn resolve_node_visibility(
    model: &AvatarModel,
    node_index: usize,
    node_name: &str,
    node_path: &str,
    visibility: &HashMap<String, bool>,
) -> bool {
    if let Some(&v) = visibility.get(node_path) {
        if !v {
            return false;
        }
    }
    if let Some(&v) = visibility.get(node_name) {
        if !v {
            return false;
        }
    }
    let mut cur = Some(node_index);
    let mut guard = 0;
    while let Some(idx) = cur {
        if guard > model.nodes.len() {
            break;
        }
        guard += 1;
        if let Some(node) = model.nodes.get(idx) {
            if let Some(&v) = visibility.get(&node.name) {
                if !v {
                    return false;
                }
            }
            cur = node.parent;
        } else {
            break;
        }
    }
    true
}

fn create_depth_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("vrc_main_depth"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn create_buffer_init(
    device: &wgpu::Device,
    label: &str,
    usage: wgpu::BufferUsages,
    data: &[u8],
) -> wgpu::Buffer {
    let aligned_size = (data.len().max(4) as u64).div_ceil(4) * 4;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: aligned_size,
        usage,
        mapped_at_creation: true,
    });
    {
        let mut range = buffer
            .slice(..)
            .get_mapped_range_mut()
            .expect("buffer created with mapped_at_creation must map");
        range.slice(..data.len()).copy_from_slice(data);
    }
    buffer.unmap();
    buffer
}

fn upload_texture(device: &wgpu::Device, queue: &wgpu::Queue, tex: &TextureData) -> GpuTexture {
    let format = if tex.srgb {
        wgpu::TextureFormat::Rgba8UnormSrgb
    } else {
        wgpu::TextureFormat::Rgba8Unorm
    };
    let w = tex.width.max(1);
    let h = tex.height.max(1);
    let expected_len = (w as usize) * (h as usize) * 4;
    let mut pixels;
    let data_slice = if tex.rgba8.len() == expected_len {
        tex.rgba8.as_slice()
    } else {
        pixels = vec![255u8; expected_len];
        let copy_len = tex.rgba8.len().min(expected_len);
        pixels[..copy_len].copy_from_slice(&tex.rgba8[..copy_len]);
        pixels.as_slice()
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(&tex.name),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        data_slice,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w * 4),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    GpuTexture {
        _texture: texture,
        view,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_gpu_material(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    def: &MaterialDef,
    base_view: &wgpu::TextureView,
    matcap_view: &wgpu::TextureView,
    emission_view: &wgpu::TextureView,
    fur_mask_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> GpuMaterial {
    let uniforms = MaterialUniforms::from_material_def(def);
    let uniform_buffer = create_buffer_init(
        device,
        &format!("vrc_mat_u_{}", def.name),
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        bytemuck::bytes_of(&uniforms),
    );
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("vrc_mat_bg_{}", def.name)),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(base_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(matcap_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(emission_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(fur_mask_view),
            },
        ],
    });

    GpuMaterial {
        def: def.clone(),
        _uniform_buffer: uniform_buffer,
        bind_group,
    }
}
