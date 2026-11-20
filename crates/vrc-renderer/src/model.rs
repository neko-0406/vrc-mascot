use std::collections::BTreeMap;
use std::f32::consts::PI;
use std::path::Path;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
use vrc_schema::{
    AlphaMode, BlendShapeBinding, ColliderDef, ColliderShape, CullMode, MascotManifest,
    MaterialDef, MaterialParams, PhysBoneDef, PhysBoneLimitType, ShaderFamily, VisemeMap,
    SCHEMA_VERSION,
};

use crate::error::RendererError;

/// Interleaved GPU vertex format matching WGSL `VertexInput` locations 0..6.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 4],
    pub texcoord_0: [f32; 2],
    pub color_0: [f32; 4],
    pub joints_0: [u16; 4],
    pub weights_0: [f32; 4],
}

impl Default for MeshVertex {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
            texcoord_0: [0.0, 0.0],
            color_0: [1.0, 1.0, 1.0, 1.0],
            joints_0: [0; 4],
            weights_0: [0.0; 4],
        }
    }
}

impl MeshVertex {
    const ATTRS: [wgpu::VertexAttribute; 7] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x4,
        3 => Float32x2,
        4 => Float32x4,
        5 => Uint16x4,
        6 => Float32x4,
    ];

    /// `wgpu::VertexBufferLayout` for [`MeshVertex`].
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &Self::ATTRS,
    };
}

/// Scene-graph / skeleton node in an [`AvatarModel`].
#[derive(Debug, Clone)]
pub struct ModelNode {
    pub name: String,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub rest_translation: Vec3,
    pub rest_rotation: Quat,
    pub rest_scale: Vec3,
}

impl ModelNode {
    /// Local rest TRS matrix.
    pub fn rest_local_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(
            self.rest_scale,
            self.rest_rotation,
            self.rest_translation,
        )
    }
}

/// BlendShape / glTF morph target deltas for a mesh primitive.
#[derive(Debug, Clone)]
pub struct MorphTargetData {
    pub name: String,
    pub position_deltas: Vec<[f32; 3]>,
    pub normal_deltas: Vec<[f32; 3]>,
}

/// Drawable mesh primitive with CPU-side base vertices, indices, and morph target deltas.
#[derive(Debug, Clone)]
pub struct MeshPrimitiveData {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    pub morph_targets: Vec<MorphTargetData>,
    pub material_index: Option<usize>,
    pub node_index: usize,
    pub node_name: String,
    pub skin_index: Option<usize>,
}

impl MeshPrimitiveData {
    /// Average position of vertices in local space (used for back-to-front transparent sorting).
    pub fn bounding_center(&self) -> Vec3 {
        if self.vertices.is_empty() {
            return Vec3::ZERO;
        }
        let mut min_v = Vec3::splat(f32::MAX);
        let mut max_v = Vec3::splat(f32::MIN);
        for v in &self.vertices {
            let p = Vec3::from_array(v.position);
            min_v = min_v.min(p);
            max_v = max_v.max(p);
        }
        (min_v + max_v) * 0.5
    }

    /// Evaluates morph targets with `weights` into `out` (reusing allocation).
    pub fn evaluate_morph_targets(&self, weights: &[f32], out: &mut Vec<MeshVertex>) {
        out.clear();
        out.extend_from_slice(&self.vertices);

        for (target_idx, target) in self.morph_targets.iter().enumerate() {
            let w = weights.get(target_idx).copied().unwrap_or(0.0);
            if w.abs() <= 1e-5 {
                continue;
            }
            for (v_idx, v) in out.iter_mut().enumerate() {
                if let Some(dp) = target.position_deltas.get(v_idx) {
                    v.position[0] += dp[0] * w;
                    v.position[1] += dp[1] * w;
                    v.position[2] += dp[2] * w;
                }
                if let Some(dn) = target.normal_deltas.get(v_idx) {
                    v.normal[0] += dn[0] * w;
                    v.normal[1] += dn[1] * w;
                    v.normal[2] += dn[2] * w;
                }
            }
        }

        if !self.morph_targets.is_empty() {
            for v in out.iter_mut() {
                let n = Vec3::from_array(v.normal);
                let len_sq = n.length_squared();
                if len_sq > 1e-8 {
                    v.normal = (n / len_sq.sqrt()).to_array();
                }
            }
        }
    }
}

/// Skeletal skin binding joints to inverse bind matrices.
#[derive(Debug, Clone)]
pub struct SkinData {
    pub name: String,
    pub joints: Vec<usize>,
    pub inverse_bind_matrices: Vec<Mat4>,
}

/// Decoded RGBA8 texture image ready for GPU upload.
#[derive(Debug, Clone)]
pub struct TextureData {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
    pub srgb: bool,
}

/// Loaded or procedurally generated 3D avatar model ready for [`crate::MascotRenderer`].
#[derive(Debug, Clone)]
pub struct AvatarModel {
    pub name: String,
    pub nodes: Vec<ModelNode>,
    pub primitives: Vec<MeshPrimitiveData>,
    pub skins: Vec<SkinData>,
    pub textures: Vec<TextureData>,
    pub material_base_textures: Vec<Option<usize>>,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
}

impl AvatarModel {
    /// Loads `mascot_manifest.json` and its paired `avatar.glb` file.
    pub fn load_from_manifest(
        manifest_path: &Path,
    ) -> Result<(AvatarModel, MascotManifest), RendererError> {
        let manifest = MascotManifest::load(manifest_path)?;
        let glb_rel = Path::new(&manifest.glb_file);
        let glb_path = if glb_rel.is_absolute() {
            glb_rel.to_path_buf()
        } else {
            manifest_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(glb_rel)
        };
        if !glb_path.exists() {
            return Err(RendererError::MissingAsset(format!(
                "GLB file not found: {}",
                glb_path.display()
            )));
        }

        let (document, buffers, images) = gltf::import(&glb_path)?;
        let target_names = extract_gltf_target_names(&glb_path);
        let model_name = if manifest.avatar_name.trim().is_empty() {
            glb_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("Avatar")
                .to_owned()
        } else {
            manifest.avatar_name.clone()
        };

        let mut nodes = vec![
            ModelNode {
                name: String::new(),
                parent: None,
                children: Vec::new(),
                rest_translation: Vec3::ZERO,
                rest_rotation: Quat::IDENTITY,
                rest_scale: Vec3::ONE,
            };
            document.nodes().len()
        ];

        for node in document.nodes() {
            let idx = node.index();
            let (t, r, s) = node.transform().decomposed();
            let children: Vec<usize> = node.children().map(|c| c.index()).collect();
            nodes[idx].name = node
                .name()
                .map(str::to_owned)
                .unwrap_or_else(|| format!("Node_{idx}"));
            nodes[idx].rest_translation = Vec3::from_array(t);
            nodes[idx].rest_rotation = Quat::from_xyzw(r[0], r[1], r[2], r[3]).normalize();
            nodes[idx].rest_scale = Vec3::from_array(s);
            nodes[idx].children = children.clone();
            for child_idx in children {
                if let Some(child) = nodes.get_mut(child_idx) {
                    child.parent = Some(idx);
                }
            }
        }

        // Track which image indices are used as sRGB color textures.
        let mut srgb_images = vec![false; images.len()];
        let mut material_base_textures = Vec::with_capacity(document.materials().len());
        for mat in document.materials() {
            let base_img = mat
                .pbr_metallic_roughness()
                .base_color_texture()
                .map(|info| info.texture().source().index());
            if let Some(img_idx) = base_img {
                if let Some(flag) = srgb_images.get_mut(img_idx) {
                    *flag = true;
                }
            }
            if let Some(em_info) = mat.emissive_texture() {
                if let Some(flag) = srgb_images.get_mut(em_info.texture().source().index()) {
                    *flag = true;
                }
            }
            material_base_textures.push(base_img);
        }

        let textures = images
            .iter()
            .enumerate()
            .map(|(idx, img)| TextureData {
                name: format!("Texture_{idx}"),
                width: img.width.max(1),
                height: img.height.max(1),
                rgba8: convert_gltf_image_to_rgba8(img),
                srgb: srgb_images.get(idx).copied().unwrap_or(false),
            })
            .collect();

        let mut skins = Vec::with_capacity(document.skins().len());
        for skin in document.skins() {
            let joints: Vec<usize> = skin.joints().map(|j| j.index()).collect();
            let reader = skin.reader(|buffer| buffers.get(buffer.index()).map(|d| d.0.as_slice()));
            let mut ibms: Vec<Mat4> = reader
                .read_inverse_bind_matrices()
                .map(|iter| iter.map(|m| Mat4::from_cols_array_2d(&m)).collect())
                .unwrap_or_default();
            if ibms.len() < joints.len() {
                ibms.resize(joints.len(), Mat4::IDENTITY);
            }
            skins.push(SkinData {
                name: skin
                    .name()
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("Skin_{}", skin.index())),
                joints,
                inverse_bind_matrices: ibms,
            });
        }

        let mut primitives = Vec::new();
        for node in document.nodes() {
            let Some(mesh) = node.mesh() else {
                continue;
            };
            let mesh_index = mesh.index();
            let node_index = node.index();
            let node_name = nodes[node_index].name.clone();
            let skin_index = node.skin().map(|s| s.index());

            for prim in mesh.primitives() {
                let prim_index = prim.index();
                let reader =
                    prim.reader(|buffer| buffers.get(buffer.index()).map(|d| d.0.as_slice()));
                let Some(positions) = reader.read_positions() else {
                    continue;
                };
                let pos_vec: Vec<[f32; 3]> = positions.collect();
                if pos_vec.is_empty() {
                    continue;
                }
                let v_count = pos_vec.len();
                let normals: Vec<[f32; 3]> = reader
                    .read_normals()
                    .map(|n| n.collect())
                    .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; v_count]);
                let tangents: Vec<[f32; 4]> = reader
                    .read_tangents()
                    .map(|t| t.collect())
                    .unwrap_or_else(|| vec![[1.0, 0.0, 0.0, 1.0]; v_count]);
                let uvs: Vec<[f32; 2]> = reader
                    .read_tex_coords(0)
                    .map(|tc| tc.into_f32().collect())
                    .unwrap_or_else(|| vec![[0.0, 0.0]; v_count]);
                let colors: Vec<[f32; 4]> = reader
                    .read_colors(0)
                    .map(|c| c.into_rgba_f32().collect())
                    .unwrap_or_else(|| vec![[1.0, 1.0, 1.0, 1.0]; v_count]);
                let joints: Vec<[u16; 4]> = reader
                    .read_joints(0)
                    .map(|j| j.into_u16().collect())
                    .unwrap_or_else(|| vec![[0, 0, 0, 0]; v_count]);
                let weights: Vec<[f32; 4]> = reader
                    .read_weights(0)
                    .map(|w| w.into_f32().collect())
                    .unwrap_or_else(|| {
                        if skin_index.is_some() {
                            vec![[1.0, 0.0, 0.0, 0.0]; v_count]
                        } else {
                            vec![[0.0, 0.0, 0.0, 0.0]; v_count]
                        }
                    });

                let mut vertices = Vec::with_capacity(v_count);
                for (i, &position) in pos_vec.iter().enumerate() {
                    vertices.push(MeshVertex {
                        position,
                        normal: normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]),
                        tangent: tangents.get(i).copied().unwrap_or([1.0, 0.0, 0.0, 1.0]),
                        texcoord_0: uvs.get(i).copied().unwrap_or([0.0, 0.0]),
                        color_0: colors.get(i).copied().unwrap_or([1.0, 1.0, 1.0, 1.0]),
                        joints_0: joints.get(i).copied().unwrap_or([0; 4]),
                        weights_0: weights.get(i).copied().unwrap_or([0.0; 4]),
                    });
                }

                let indices: Vec<u32> = reader
                    .read_indices()
                    .map(|idx| idx.into_u32().collect())
                    .unwrap_or_else(|| (0..v_count as u32).collect());

                let mut morph_targets = Vec::new();
                for (t_idx, (pos_deltas, norm_deltas, _tan_deltas)) in
                    reader.read_morph_targets().enumerate()
                {
                    let name = target_names
                        .resolve(mesh_index, prim_index, t_idx)
                        .unwrap_or_else(|| format!("target_{t_idx}"));
                    morph_targets.push(MorphTargetData {
                        name,
                        position_deltas: pos_deltas
                            .map(|it| it.collect())
                            .unwrap_or_else(|| vec![[0.0; 3]; v_count]),
                        normal_deltas: norm_deltas
                            .map(|it| it.collect())
                            .unwrap_or_else(|| vec![[0.0; 3]; v_count]),
                    });
                }

                primitives.push(MeshPrimitiveData {
                    vertices,
                    indices,
                    morph_targets,
                    material_index: prim.material().index(),
                    node_index,
                    node_name: node_name.clone(),
                    skin_index,
                });
            }
        }

        if primitives.is_empty() {
            return Err(RendererError::InvalidModel(format!(
                "GLB '{}' contains no mesh primitives",
                glb_path.display()
            )));
        }

        let mut model = AvatarModel {
            name: model_name,
            nodes,
            primitives,
            skins,
            textures,
            material_base_textures,
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::ZERO,
        };
        model.recompute_bounds();
        Ok((model, manifest))
    }

    /// Builds a stylized, rigged 3D fallback mascot with humanoid bones, PhysBone ears/hair,
    /// Viseme & Blink morph targets, and Toon/Outline/AudioLink materials.
    pub fn procedural_fallback() -> (AvatarModel, MascotManifest) {
        let mut nodes = Vec::new();
        let mut add_node = |name: &str, parent: Option<usize>, pos: Vec3, rot: Quat| -> usize {
            let idx = nodes.len();
            nodes.push(ModelNode {
                name: name.to_owned(),
                parent,
                children: Vec::new(),
                rest_translation: pos,
                rest_rotation: rot,
                rest_scale: Vec3::ONE,
            });
            if let Some(p) = parent {
                nodes[p].children.push(idx);
            }
            idx
        };

        let root = add_node("Root", None, Vec3::ZERO, Quat::IDENTITY);
        let armature = add_node("Armature", Some(root), Vec3::ZERO, Quat::IDENTITY);
        let hips = add_node(
            "Hips",
            Some(armature),
            Vec3::new(0.0, 0.56, 0.0),
            Quat::IDENTITY,
        );
        let spine = add_node(
            "Spine",
            Some(hips),
            Vec3::new(0.0, 0.14, 0.0),
            Quat::IDENTITY,
        );
        let chest = add_node(
            "Chest",
            Some(spine),
            Vec3::new(0.0, 0.15, 0.0),
            Quat::IDENTITY,
        );
        let neck = add_node(
            "Neck",
            Some(chest),
            Vec3::new(0.0, 0.13, 0.0),
            Quat::IDENTITY,
        );
        let head = add_node(
            "Head",
            Some(neck),
            Vec3::new(0.0, 0.07, 0.0),
            Quat::IDENTITY,
        );

        let left_eye = add_node(
            "LeftEye",
            Some(head),
            Vec3::new(-0.07, 0.12, 0.145),
            Quat::IDENTITY,
        );
        let right_eye = add_node(
            "RightEye",
            Some(head),
            Vec3::new(0.07, 0.12, 0.145),
            Quat::IDENTITY,
        );

        let ear_l = add_node(
            "Ear_L",
            Some(head),
            Vec3::new(-0.115, 0.24, 0.01),
            Quat::from_rotation_z(0.28),
        );
        let ear_l_tip = add_node(
            "Ear_L_Tip",
            Some(ear_l),
            Vec3::new(0.0, 0.15, 0.0),
            Quat::IDENTITY,
        );

        let ear_r = add_node(
            "Ear_R",
            Some(head),
            Vec3::new(0.115, 0.24, 0.01),
            Quat::from_rotation_z(-0.28),
        );
        let ear_r_tip = add_node(
            "Ear_R_Tip",
            Some(ear_r),
            Vec3::new(0.0, 0.15, 0.0),
            Quat::IDENTITY,
        );

        let hair_back = add_node(
            "Hair_Back",
            Some(head),
            Vec3::new(0.0, 0.15, -0.11),
            Quat::from_rotation_x(0.16),
        );
        let hair_back_tip = add_node(
            "Hair_Back_Tip",
            Some(hair_back),
            Vec3::new(0.0, -0.22, -0.04),
            Quat::IDENTITY,
        );

        let l_upper_arm = add_node(
            "LeftUpperArm",
            Some(chest),
            Vec3::new(-0.145, 0.09, 0.0),
            Quat::from_rotation_z(0.25),
        );
        let l_lower_arm = add_node(
            "LeftLowerArm",
            Some(l_upper_arm),
            Vec3::new(0.0, -0.17, 0.02),
            Quat::IDENTITY,
        );
        let r_upper_arm = add_node(
            "RightUpperArm",
            Some(chest),
            Vec3::new(0.145, 0.09, 0.0),
            Quat::from_rotation_z(-0.25),
        );
        let r_lower_arm = add_node(
            "RightLowerArm",
            Some(r_upper_arm),
            Vec3::new(0.0, -0.17, 0.02),
            Quat::IDENTITY,
        );

        let l_upper_leg = add_node(
            "LeftUpperLeg",
            Some(hips),
            Vec3::new(-0.075, -0.04, 0.0),
            Quat::IDENTITY,
        );
        let l_lower_leg = add_node(
            "LeftLowerLeg",
            Some(l_upper_leg),
            Vec3::new(0.0, -0.24, 0.0),
            Quat::IDENTITY,
        );
        let r_upper_leg = add_node(
            "RightUpperLeg",
            Some(hips),
            Vec3::new(0.075, -0.04, 0.0),
            Quat::IDENTITY,
        );
        let r_lower_leg = add_node(
            "RightLowerLeg",
            Some(r_upper_leg),
            Vec3::new(0.0, -0.24, 0.0),
            Quat::IDENTITY,
        );
        let body_node = add_node("Body", Some(root), Vec3::ZERO, Quat::IDENTITY);

        let joints = vec![
            hips,
            spine,
            chest,
            neck,
            head,
            left_eye,
            right_eye,
            ear_l,
            ear_l_tip,
            ear_r,
            ear_r_tip,
            hair_back,
            hair_back_tip,
            l_upper_arm,
            l_lower_arm,
            r_upper_arm,
            r_lower_arm,
            l_upper_leg,
            l_lower_leg,
            r_upper_leg,
            r_lower_leg,
        ];

        let rest_world = compute_world_matrices_from_nodes(&nodes);
        let inverse_bind_matrices: Vec<Mat4> =
            joints.iter().map(|&j| rest_world[j].inverse()).collect();

        // Map node index -> skin joint index
        let joint_slot = |node_idx: usize| -> u16 {
            joints.iter().position(|&j| j == node_idx).unwrap_or(0) as u16
        };

        // Helper to build skinned mesh geometry in world/model space around rest joint position.
        let mut skin_verts = Vec::new();
        let mut skin_indices = Vec::new();
        // Head sphere
        let head_center = rest_world[head].transform_point3(Vec3::new(0.0, 0.13, 0.0));
        append_uv_sphere(
            &mut skin_verts,
            &mut skin_indices,
            head_center,
            Vec3::new(0.175, 0.165, 0.165),
            20,
            14,
            joint_slot(head),
            [1.0, 1.0, 1.0, 1.0],
        );
        // Neck & hands
        let neck_pos = rest_world[neck].transform_point3(Vec3::new(0.0, 0.03, 0.0));
        append_uv_sphere(
            &mut skin_verts,
            &mut skin_indices,
            neck_pos,
            Vec3::new(0.048, 0.06, 0.048),
            12,
            8,
            joint_slot(neck),
            [1.0, 1.0, 1.0, 1.0],
        );
        for &arm_tip in &[l_lower_arm, r_lower_arm] {
            let hand_pos = rest_world[arm_tip].transform_point3(Vec3::new(0.0, -0.15, 0.0));
            append_uv_sphere(
                &mut skin_verts,
                &mut skin_indices,
                hand_pos,
                Vec3::new(0.042, 0.048, 0.042),
                12,
                8,
                joint_slot(arm_tip),
                [1.0, 1.0, 1.0, 1.0],
            );
        }

        // Build morph targets on the skin/face mesh for visemes & expressions.
        let mouth_center = head_center + Vec3::new(0.0, -0.055, 0.152);
        let eye_l_pos = rest_world[left_eye].transform_point3(Vec3::ZERO);
        let eye_r_pos = rest_world[right_eye].transform_point3(Vec3::ZERO);

        let make_mouth_target = |name: &str, dx_scale: f32, dy_scale: f32, dz_scale: f32| {
            let mut pos_deltas = vec![[0.0; 3]; skin_verts.len()];
            let norm_deltas = vec![[0.0; 3]; skin_verts.len()];
            for (i, v) in skin_verts.iter().enumerate() {
                let p = Vec3::from_array(v.position);
                let dist = p.distance(mouth_center);
                if dist < 0.055 {
                    let falloff = (1.0 - dist / 0.055).powi(2);
                    let sign_y = if p.y >= mouth_center.y { 0.4 } else { -1.0 };
                    let sign_x = (p.x - mouth_center.x).signum();
                    pos_deltas[i] = [
                        sign_x * dx_scale * falloff,
                        sign_y * dy_scale * falloff,
                        dz_scale * falloff,
                    ];
                }
            }
            MorphTargetData {
                name: name.to_owned(),
                position_deltas: pos_deltas,
                normal_deltas: norm_deltas,
            }
        };

        let make_blink_target = || {
            let mut pos_deltas = vec![[0.0; 3]; skin_verts.len()];
            let norm_deltas = vec![[0.0; 3]; skin_verts.len()];
            for (i, v) in skin_verts.iter().enumerate() {
                let p = Vec3::from_array(v.position);
                let dl = p.distance(eye_l_pos);
                let dr = p.distance(eye_r_pos);
                let d = dl.min(dr);
                if d < 0.055 {
                    let falloff = 1.0 - d / 0.055;
                    pos_deltas[i] = [0.0, -0.015 * falloff, 0.012 * falloff];
                }
            }
            MorphTargetData {
                name: "Blink".to_owned(),
                position_deltas: pos_deltas,
                normal_deltas: norm_deltas,
            }
        };

        let face_morph_targets = vec![
            make_mouth_target("vrc.v_aa", 0.004, 0.026, -0.008),
            make_mouth_target("vrc.v_ih", 0.018, 0.008, -0.004),
            make_mouth_target("vrc.v_ou", -0.008, 0.018, 0.012),
            make_mouth_target("vrc.v_ee", 0.014, 0.014, -0.004),
            make_mouth_target("vrc.v_oh", -0.004, 0.022, 0.009),
            make_blink_target(),
            make_mouth_target("Smile", 0.020, 0.010, 0.006),
        ];

        // Outfit mesh (torso hoodie, waist, skirt, sleeves, legs, shoes)
        let mut outfit_verts = Vec::new();
        let mut outfit_indices = Vec::new();
        let chest_pos = rest_world[chest].transform_point3(Vec3::new(0.0, 0.02, 0.0));
        append_uv_sphere(
            &mut outfit_verts,
            &mut outfit_indices,
            chest_pos,
            Vec3::new(0.135, 0.145, 0.095),
            16,
            12,
            joint_slot(chest),
            [1.0, 1.0, 1.0, 1.0],
        );
        let hips_pos = rest_world[hips].transform_point3(Vec3::new(0.0, -0.01, 0.0));
        append_cone_frustum(
            &mut outfit_verts,
            &mut outfit_indices,
            chest_pos,
            hips_pos + Vec3::new(0.0, 0.06, 0.0),
            0.12,
            0.105,
            16,
            joint_slot(spine),
            [1.0, 1.0, 1.0, 1.0],
        );
        append_cone_frustum(
            &mut outfit_verts,
            &mut outfit_indices,
            hips_pos + Vec3::new(0.0, 0.12, 0.0),
            hips_pos + Vec3::new(0.0, -0.14, 0.0),
            0.105,
            0.185,
            18,
            joint_slot(hips),
            [1.0, 1.0, 1.0, 1.0],
        );
        for &(upper, lower) in &[(l_upper_arm, l_lower_arm), (r_upper_arm, r_lower_arm)] {
            let p0 = rest_world[upper].transform_point3(Vec3::ZERO);
            let p1 = rest_world[lower].transform_point3(Vec3::ZERO);
            let p2 = rest_world[lower].transform_point3(Vec3::new(0.0, -0.14, 0.0));
            append_cone_frustum(
                &mut outfit_verts,
                &mut outfit_indices,
                p0,
                p1,
                0.048,
                0.044,
                12,
                joint_slot(upper),
                [1.0, 1.0, 1.0, 1.0],
            );
            append_cone_frustum(
                &mut outfit_verts,
                &mut outfit_indices,
                p1,
                p2,
                0.044,
                0.052,
                12,
                joint_slot(lower),
                [1.0, 1.0, 1.0, 1.0],
            );
        }
        for &(upper, lower) in &[(l_upper_leg, l_lower_leg), (r_upper_leg, r_lower_leg)] {
            let p0 = rest_world[upper].transform_point3(Vec3::ZERO);
            let p1 = rest_world[lower].transform_point3(Vec3::ZERO);
            let p2 = rest_world[lower].transform_point3(Vec3::new(0.0, -0.24, 0.0));
            append_cone_frustum(
                &mut outfit_verts,
                &mut outfit_indices,
                p0,
                p1,
                0.055,
                0.046,
                12,
                joint_slot(upper),
                [1.0, 1.0, 1.0, 1.0],
            );
            append_cone_frustum(
                &mut outfit_verts,
                &mut outfit_indices,
                p1,
                p2,
                0.045,
                0.042,
                12,
                joint_slot(lower),
                [1.0, 1.0, 1.0, 1.0],
            );
            append_uv_sphere(
                &mut outfit_verts,
                &mut outfit_indices,
                p2 + Vec3::new(0.0, 0.02, 0.025),
                Vec3::new(0.048, 0.036, 0.072),
                12,
                8,
                joint_slot(lower),
                [0.85, 0.85, 0.9, 1.0],
            );
        }

        // Hair & PhysBone Kemonomimi ears mesh
        let mut hair_verts = Vec::new();
        let mut hair_indices = Vec::new();
        append_uv_sphere(
            &mut hair_verts,
            &mut hair_indices,
            head_center + Vec3::new(0.0, 0.035, -0.025),
            Vec3::new(0.188, 0.175, 0.175),
            18,
            12,
            joint_slot(head),
            [1.0, 1.0, 1.0, 1.0],
        );
        for &(ear_root, ear_tip) in &[(ear_l, ear_l_tip), (ear_r, ear_r_tip)] {
            let p0 = rest_world[ear_root].transform_point3(Vec3::ZERO);
            let p1 = rest_world[ear_tip].transform_point3(Vec3::ZERO);
            append_cone_frustum(
                &mut hair_verts,
                &mut hair_indices,
                p0,
                p1,
                0.055,
                0.008,
                12,
                joint_slot(ear_root),
                [1.0, 1.0, 1.0, 1.0],
            );
        }
        let hair_p0 = rest_world[hair_back].transform_point3(Vec3::ZERO);
        let hair_p1 = rest_world[hair_back_tip].transform_point3(Vec3::ZERO);
        append_cone_frustum(
            &mut hair_verts,
            &mut hair_indices,
            hair_p0,
            hair_p1,
            0.13,
            0.08,
            14,
            joint_slot(hair_back),
            [1.0, 1.0, 1.0, 1.0],
        );

        // Eyes & blush accents
        let mut eye_verts = Vec::new();
        let mut eye_indices = Vec::new();
        for &eye_node in &[left_eye, right_eye] {
            let ep = rest_world[eye_node].transform_point3(Vec3::ZERO);
            append_uv_sphere(
                &mut eye_verts,
                &mut eye_indices,
                ep,
                Vec3::new(0.028, 0.036, 0.018),
                12,
                10,
                joint_slot(eye_node),
                [1.0, 1.0, 1.0, 1.0],
            );
        }
        // Small mouth accent dot
        append_uv_sphere(
            &mut eye_verts,
            &mut eye_indices,
            mouth_center + Vec3::new(0.0, 0.0, 0.005),
            Vec3::new(0.016, 0.008, 0.008),
            10,
            6,
            joint_slot(head),
            [0.95, 0.45, 0.52, 1.0],
        );

        let primitives = vec![
            MeshPrimitiveData {
                vertices: skin_verts,
                indices: skin_indices,
                morph_targets: face_morph_targets,
                material_index: Some(0),
                node_index: body_node,
                node_name: "Body".to_owned(),
                skin_index: Some(0),
            },
            MeshPrimitiveData {
                vertices: outfit_verts,
                indices: outfit_indices,
                morph_targets: Vec::new(),
                material_index: Some(1),
                node_index: body_node,
                node_name: "Outfit".to_owned(),
                skin_index: Some(0),
            },
            MeshPrimitiveData {
                vertices: hair_verts,
                indices: hair_indices,
                morph_targets: Vec::new(),
                material_index: Some(2),
                node_index: body_node,
                node_name: "Hair".to_owned(),
                skin_index: Some(0),
            },
            MeshPrimitiveData {
                vertices: eye_verts,
                indices: eye_indices,
                morph_targets: Vec::new(),
                material_index: Some(3),
                node_index: body_node,
                node_name: "Eyes".to_owned(),
                skin_index: Some(0),
            },
        ];

        let mut model = AvatarModel {
            name: "ProceduralMascot".to_owned(),
            nodes,
            primitives,
            skins: vec![SkinData {
                name: "ArmatureSkin".to_owned(),
                joints,
                inverse_bind_matrices,
            }],
            textures: Vec::new(),
            material_base_textures: Vec::new(),
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::ZERO,
        };
        model.recompute_bounds();

        let mut humanoid_bones = BTreeMap::new();
        for &(humanoid, path) in &[
            ("Hips", "Armature/Hips"),
            ("Spine", "Armature/Hips/Spine"),
            ("Chest", "Armature/Hips/Spine/Chest"),
            ("Neck", "Armature/Hips/Spine/Chest/Neck"),
            ("Head", "Armature/Hips/Spine/Chest/Neck/Head"),
            ("LeftEye", "Armature/Hips/Spine/Chest/Neck/Head/LeftEye"),
            ("RightEye", "Armature/Hips/Spine/Chest/Neck/Head/RightEye"),
            ("LeftUpperArm", "Armature/Hips/Spine/Chest/LeftUpperArm"),
            (
                "LeftLowerArm",
                "Armature/Hips/Spine/Chest/LeftUpperArm/LeftLowerArm",
            ),
            ("RightUpperArm", "Armature/Hips/Spine/Chest/RightUpperArm"),
            (
                "RightLowerArm",
                "Armature/Hips/Spine/Chest/RightUpperArm/RightLowerArm",
            ),
            ("LeftUpperLeg", "Armature/Hips/LeftUpperLeg"),
            ("LeftLowerLeg", "Armature/Hips/LeftUpperLeg/LeftLowerLeg"),
            ("RightUpperLeg", "Armature/Hips/RightUpperLeg"),
            ("RightLowerLeg", "Armature/Hips/RightUpperLeg/RightLowerLeg"),
        ] {
            humanoid_bones.insert(humanoid.to_owned(), path.to_owned());
        }

        let manifest = MascotManifest {
            schema_version: SCHEMA_VERSION.to_owned(),
            avatar_name: "ProceduralMascot".to_owned(),
            glb_file: "procedural_fallback.glb".to_owned(),
            humanoid_bones,
            viseme_map: VisemeMap {
                sil: None,
                aa: Some(BlendShapeBinding {
                    mesh: "Body".to_owned(),
                    blendshape: "vrc.v_aa".to_owned(),
                }),
                ih: Some(BlendShapeBinding {
                    mesh: "Body".to_owned(),
                    blendshape: "vrc.v_ih".to_owned(),
                }),
                ou: Some(BlendShapeBinding {
                    mesh: "Body".to_owned(),
                    blendshape: "vrc.v_ou".to_owned(),
                }),
                ee: Some(BlendShapeBinding {
                    mesh: "Body".to_owned(),
                    blendshape: "vrc.v_ee".to_owned(),
                }),
                oh: Some(BlendShapeBinding {
                    mesh: "Body".to_owned(),
                    blendshape: "vrc.v_oh".to_owned(),
                }),
            },
            phys_bones: vec![
                PhysBoneDef {
                    root_node: "Armature/Hips/Spine/Chest/Neck/Head/Ear_L".to_owned(),
                    pull: 0.25,
                    spring: 0.35,
                    stiffness: 0.15,
                    gravity: 0.04,
                    radius: 0.03,
                    limit_type: PhysBoneLimitType::Angle,
                    max_angle_x: 35.0,
                    collider_refs: vec!["col_head".to_owned()],
                    ..Default::default()
                },
                PhysBoneDef {
                    root_node: "Armature/Hips/Spine/Chest/Neck/Head/Ear_R".to_owned(),
                    pull: 0.25,
                    spring: 0.35,
                    stiffness: 0.15,
                    gravity: 0.04,
                    radius: 0.03,
                    limit_type: PhysBoneLimitType::Angle,
                    max_angle_x: 35.0,
                    collider_refs: vec!["col_head".to_owned()],
                    ..Default::default()
                },
                PhysBoneDef {
                    root_node: "Armature/Hips/Spine/Chest/Neck/Head/Hair_Back".to_owned(),
                    pull: 0.2,
                    spring: 0.25,
                    stiffness: 0.1,
                    gravity: 0.08,
                    radius: 0.04,
                    limit_type: PhysBoneLimitType::Angle,
                    max_angle_x: 45.0,
                    collider_refs: vec!["col_head".to_owned()],
                    ..Default::default()
                },
            ],
            colliders: vec![ColliderDef {
                id: "col_head".to_owned(),
                parent_node: "Armature/Hips/Spine/Chest/Neck/Head".to_owned(),
                shape: ColliderShape::Sphere,
                radius: 0.16,
                height: 0.0,
                position: [0.0, 0.13, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                inside_bounds: false,
            }],
            materials: vec![
                MaterialDef {
                    gltf_material_index: 0,
                    name: "Skin".to_owned(),
                    shader_family: ShaderFamily::LilToon,
                    render_queue: 2000,
                    alpha_mode: AlphaMode::Opaque,
                    params: MaterialParams {
                        main_color: [1.0, 0.91, 0.86, 1.0],
                        shadow_color_1: [0.92, 0.74, 0.74, 1.0],
                        shadow_color_2: [0.80, 0.60, 0.65, 1.0],
                        rim_enable: true,
                        rim_color: [1.0, 0.95, 0.92, 0.35],
                        outline_enable: true,
                        outline_width: 0.0025,
                        outline_color: [0.28, 0.18, 0.18, 1.0],
                        cull_mode: CullMode::Back,
                        ..Default::default()
                    },
                },
                MaterialDef {
                    gltf_material_index: 1,
                    name: "Outfit".to_owned(),
                    shader_family: ShaderFamily::LilToon,
                    render_queue: 2000,
                    alpha_mode: AlphaMode::Opaque,
                    params: MaterialParams {
                        main_color: [0.46, 0.56, 0.88, 1.0],
                        shadow_color_1: [0.30, 0.36, 0.68, 1.0],
                        shadow_color_2: [0.20, 0.24, 0.50, 1.0],
                        rim_enable: true,
                        rim_color: [0.75, 0.88, 1.0, 0.5],
                        outline_enable: true,
                        outline_width: 0.003,
                        outline_color: [0.12, 0.14, 0.26, 1.0],
                        audiolink_enable: true,
                        audiolink_band: 0,
                        audiolink_strength: 0.6,
                        cull_mode: CullMode::Back,
                        ..Default::default()
                    },
                },
                MaterialDef {
                    gltf_material_index: 2,
                    name: "Hair".to_owned(),
                    shader_family: ShaderFamily::LilToon,
                    render_queue: 2000,
                    alpha_mode: AlphaMode::Opaque,
                    params: MaterialParams {
                        main_color: [0.97, 0.80, 0.88, 1.0],
                        shadow_color_1: [0.78, 0.56, 0.70, 1.0],
                        shadow_color_2: [0.62, 0.42, 0.58, 1.0],
                        rim_enable: true,
                        rim_color: [1.0, 0.92, 0.98, 0.55],
                        outline_enable: true,
                        outline_width: 0.0025,
                        outline_color: [0.24, 0.15, 0.22, 1.0],
                        cull_mode: CullMode::Back,
                        ..Default::default()
                    },
                },
                MaterialDef {
                    gltf_material_index: 3,
                    name: "Eyes".to_owned(),
                    shader_family: ShaderFamily::LilToon,
                    render_queue: 2000,
                    alpha_mode: AlphaMode::Opaque,
                    params: MaterialParams {
                        main_color: [0.18, 0.22, 0.42, 1.0],
                        shadow_color_1: [0.12, 0.14, 0.30, 1.0],
                        emission_enable: true,
                        emission_color: [0.16, 0.24, 0.48, 1.0],
                        cull_mode: CullMode::Back,
                        ..Default::default()
                    },
                },
            ],
            ..Default::default()
        };

        (model, manifest)
    }

    /// Computes rest-pose world matrices for every node in [`Self::nodes`].
    pub fn compute_rest_world_matrices(&self) -> Vec<Mat4> {
        compute_world_matrices_from_nodes(&self.nodes)
    }

    /// Returns the slash-separated hierarchy path for `node_index`.
    pub fn node_path(&self, node_index: usize) -> String {
        let mut parts = Vec::new();
        let mut cursor = Some(node_index);
        let mut guard = 0;
        while let Some(idx) = cursor {
            if guard > self.nodes.len() {
                break;
            }
            guard += 1;
            if let Some(node) = self.nodes.get(idx) {
                parts.push(node.name.as_str());
                cursor = node.parent;
            } else {
                break;
            }
        }
        parts.reverse();
        parts.join("/")
    }

    /// Finds a node index matching either an exact node name, full path, or trailing path suffix.
    pub fn find_node_index(&self, name_or_path: &str) -> Option<usize> {
        if let Some(idx) = self.nodes.iter().position(|n| n.name == name_or_path) {
            return Some(idx);
        }
        for idx in 0..self.nodes.len() {
            let full = self.node_path(idx);
            if full == name_or_path || full.ends_with(&format!("/{name_or_path}")) {
                return Some(idx);
            }
        }
        if let Some((_, leaf)) = name_or_path.rsplit_once('/') {
            return self.nodes.iter().position(|n| n.name == leaf);
        }
        None
    }

    /// Returns `true` if `node_index` is `head_node_index`, a descendant of it, or has a head-like name.
    pub fn is_head_or_descendant(&self, node_index: usize, head_node_index: Option<usize>) -> bool {
        if let Some(head_idx) = head_node_index {
            let mut cur = Some(node_index);
            let mut guard = 0;
            while let Some(idx) = cur {
                if idx == head_idx {
                    return true;
                }
                if guard > self.nodes.len() {
                    break;
                }
                guard += 1;
                cur = self.nodes.get(idx).and_then(|n| n.parent);
            }
        }
        if let Some(node) = self.nodes.get(node_index) {
            let lower = node.name.to_ascii_lowercase();
            return lower.contains("head")
                || lower.contains("face")
                || lower.contains("hair")
                || lower.contains("ear")
                || lower.contains("eye");
        }
        false
    }

    fn recompute_bounds(&mut self) {
        let mut min_v = Vec3::splat(f32::MAX);
        let mut max_v = Vec3::splat(f32::MIN);
        let mut any = false;
        let world = self.compute_rest_world_matrices();

        for prim in &self.primitives {
            let node_xf = world
                .get(prim.node_index)
                .copied()
                .unwrap_or(Mat4::IDENTITY);
            let xf = if prim.skin_index.is_some() {
                Mat4::IDENTITY
            } else {
                node_xf
            };
            for v in &prim.vertices {
                let p = xf.transform_point3(Vec3::from_array(v.position));
                min_v = min_v.min(p);
                max_v = max_v.max(p);
                any = true;
            }
        }
        if any {
            self.bounds_min = min_v;
            self.bounds_max = max_v;
        } else {
            self.bounds_min = Vec3::new(-0.5, 0.0, -0.5);
            self.bounds_max = Vec3::new(0.5, 1.5, 0.5);
        }
    }
}

/// Portrait auto-framing camera rig with screen projection and plane raycasting.
#[derive(Debug, Clone)]
pub struct CameraRig {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y_rad: f32,
    pub near: f32,
    pub far: f32,
    pub aspect: f32,
}

impl Default for CameraRig {
    fn default() -> Self {
        Self::for_bounds(
            Vec3::new(-0.35, 0.0, -0.25),
            Vec3::new(0.35, 1.45, 0.25),
            400,
            600,
        )
    }
}

impl CameraRig {
    /// Creates an auto-framed portrait camera fitting `[bounds_min, bounds_max]` in `(width, height)`.
    pub fn for_bounds(bounds_min: Vec3, bounds_max: Vec3, width: u32, height: u32) -> Self {
        let aspect = width.max(1) as f32 / height.max(1) as f32;
        let fov_y_rad = 32.0_f32.to_radians();
        let center = (bounds_min + bounds_max) * 0.5;
        let extents = (bounds_max - bounds_min).max(Vec3::splat(0.25));

        let half_h = extents.y * 0.5;
        let half_w = extents.x * 0.5;
        let tan_half_y = (fov_y_rad * 0.5).tan().max(1e-4);
        let tan_half_x = tan_half_y * aspect.max(1e-3);

        let dist_y = (half_h * 1.18) / tan_half_y;
        let dist_x = (half_w * 1.18) / tan_half_x;
        let dist = dist_y.max(dist_x) + extents.z * 0.5;

        Self {
            eye: Vec3::new(center.x, center.y + extents.y * 0.02, center.z + dist),
            target: center,
            up: Vec3::Y,
            fov_y_rad,
            near: 0.05,
            far: (dist * 10.0).max(50.0),
            aspect,
        }
    }

    /// Updates aspect ratio when the viewport resizes.
    pub fn set_viewport(&mut self, width: u32, height: u32) {
        self.aspect = width.max(1) as f32 / height.max(1) as f32;
    }

    pub fn view_matrix(&self) -> Mat4 {
        glam::camera::rh::view::look_at_mat4(self.eye, self.target, self.up)
    }

    pub fn proj_matrix(&self) -> Mat4 {
        glam::camera::rh::proj::directx::perspective(
            self.fov_y_rad,
            self.aspect.max(1e-3),
            self.near,
            self.far,
        )
    }

    pub fn view_proj_matrix(&self) -> Mat4 {
        self.proj_matrix() * self.view_matrix()
    }

    /// Projects a world-space point into logical/pixel screen coordinates `(0..width, 0..height)`.
    pub fn project_world_point(
        &self,
        world_pos: Vec3,
        width: u32,
        height: u32,
    ) -> Option<[f32; 2]> {
        if width == 0 || height == 0 {
            return None;
        }
        let clip = self.view_proj_matrix() * Vec4::new(world_pos.x, world_pos.y, world_pos.z, 1.0);
        if clip.w <= 1e-5 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        if !(0.0..=1.0).contains(&ndc.z) {
            return None;
        }
        let sx = (ndc.x * 0.5 + 0.5) * width as f32;
        let sy = (0.5 - ndc.y * 0.5) * height as f32;
        Some([sx, sy])
    }

    /// Casts a ray from `(screen_x, screen_y)` and intersects it with the world plane `z = plane_z`.
    pub fn screen_ray_to_world_plane(
        &self,
        screen_x: f32,
        screen_y: f32,
        width: u32,
        height: u32,
        plane_z: f32,
    ) -> Vec3 {
        let w = width.max(1) as f32;
        let h = height.max(1) as f32;
        let ndc_x = (screen_x / w) * 2.0 - 1.0;
        let ndc_y = 1.0 - (screen_y / h) * 2.0;
        let inv_vp = self.view_proj_matrix().inverse();

        let near_h = inv_vp * Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
        let far_h = inv_vp * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
        let near_p = near_h.truncate() / near_h.w.max(1e-6);
        let far_p = far_h.truncate() / far_h.w.max(1e-6);

        let dir = (far_p - near_p).normalize_or_zero();
        if dir.z.abs() > 1e-6 {
            let t = (plane_z - near_p.z) / dir.z;
            near_p + dir * t
        } else {
            let dist = self.eye.distance(self.target);
            near_p + dir * dist
        }
    }
}

fn compute_world_matrices_from_nodes(nodes: &[ModelNode]) -> Vec<Mat4> {
    let mut world = vec![Mat4::IDENTITY; nodes.len()];
    let mut visited = vec![false; nodes.len()];
    fn resolve(
        idx: usize,
        nodes: &[ModelNode],
        world: &mut [Mat4],
        visited: &mut [bool],
        depth: usize,
    ) -> Mat4 {
        if visited[idx] || depth > nodes.len() {
            return world[idx];
        }
        let local = nodes[idx].rest_local_matrix();
        let mat = match nodes[idx].parent {
            Some(p) if p < nodes.len() => resolve(p, nodes, world, visited, depth + 1) * local,
            _ => local,
        };
        world[idx] = mat;
        visited[idx] = true;
        mat
    }
    for i in 0..nodes.len() {
        resolve(i, nodes, &mut world, &mut visited, 0);
    }
    world
}

struct MeshTargetNames {
    mesh_level: Vec<Vec<String>>,
    prim_level: Vec<Vec<Vec<String>>>,
}

impl MeshTargetNames {
    fn resolve(&self, mesh_idx: usize, prim_idx: usize, target_idx: usize) -> Option<String> {
        self.prim_level
            .get(mesh_idx)
            .and_then(|prims| prims.get(prim_idx))
            .and_then(|names| names.get(target_idx))
            .or_else(|| {
                self.mesh_level
                    .get(mesh_idx)
                    .and_then(|names| names.get(target_idx))
            })
            .cloned()
    }
}

fn parse_target_names_array(val: Option<&serde_json::Value>) -> Vec<String> {
    val.and_then(|v| v.get("extras"))
        .and_then(|e| e.get("targetNames"))
        .and_then(|t| t.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| item.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

fn extract_gltf_target_names(glb_path: &Path) -> MeshTargetNames {
    let Ok(bytes) = std::fs::read(glb_path) else {
        return MeshTargetNames {
            mesh_level: Vec::new(),
            prim_level: Vec::new(),
        };
    };

    let json_val: Option<serde_json::Value> = if bytes.starts_with(b"glTF") {
        gltf::binary::Glb::from_slice(&bytes)
            .ok()
            .and_then(|glb| serde_json::from_slice(&glb.json).ok())
    } else {
        serde_json::from_slice(&bytes).ok()
    };

    let Some(root) = json_val else {
        return MeshTargetNames {
            mesh_level: Vec::new(),
            prim_level: Vec::new(),
        };
    };

    let mut mesh_level = Vec::new();
    let mut prim_level = Vec::new();
    if let Some(meshes) = root.get("meshes").and_then(|v| v.as_array()) {
        for mesh_obj in meshes {
            mesh_level.push(parse_target_names_array(Some(mesh_obj)));
            let mut prims_names = Vec::new();
            if let Some(prims) = mesh_obj.get("primitives").and_then(|v| v.as_array()) {
                for prim_obj in prims {
                    prims_names.push(parse_target_names_array(Some(prim_obj)));
                }
            }
            prim_level.push(prims_names);
        }
    }

    MeshTargetNames {
        mesh_level,
        prim_level,
    }
}

fn convert_gltf_image_to_rgba8(img: &gltf::image::Data) -> Vec<u8> {
    use gltf::image::Format;
    let px_count = (img.width as usize) * (img.height as usize);
    let mut out = Vec::with_capacity(px_count * 4);
    match img.format {
        Format::R8 => {
            for &r in &img.pixels {
                out.extend_from_slice(&[r, r, r, 255]);
            }
        }
        Format::R8G8 => {
            for chunk in img.pixels.chunks_exact(2) {
                out.extend_from_slice(&[chunk[0], chunk[1], 0, 255]);
            }
        }
        Format::R8G8B8 => {
            for chunk in img.pixels.chunks_exact(3) {
                out.extend_from_slice(&[chunk[0], chunk[1], chunk[2], 255]);
            }
        }
        Format::R8G8B8A8 => return img.pixels.clone(),
        Format::R16 => {
            for chunk in img.pixels.chunks_exact(2) {
                let r = chunk[1];
                out.extend_from_slice(&[r, r, r, 255]);
            }
        }
        Format::R16G16 => {
            for chunk in img.pixels.chunks_exact(4) {
                out.extend_from_slice(&[chunk[1], chunk[3], 0, 255]);
            }
        }
        Format::R16G16B16 => {
            for chunk in img.pixels.chunks_exact(6) {
                out.extend_from_slice(&[chunk[1], chunk[3], chunk[5], 255]);
            }
        }
        Format::R16G16B16A16 => {
            for chunk in img.pixels.chunks_exact(8) {
                out.extend_from_slice(&[chunk[1], chunk[3], chunk[5], chunk[7]]);
            }
        }
        Format::R32G32B32FLOAT => {
            for chunk in img.pixels.chunks_exact(12) {
                let r = f32::from_le_bytes(chunk[0..4].try_into().unwrap_or([0; 4]));
                let g = f32::from_le_bytes(chunk[4..8].try_into().unwrap_or([0; 4]));
                let b = f32::from_le_bytes(chunk[8..12].try_into().unwrap_or([0; 4]));
                out.extend_from_slice(&[
                    (r.clamp(0.0, 1.0) * 255.0) as u8,
                    (g.clamp(0.0, 1.0) * 255.0) as u8,
                    (b.clamp(0.0, 1.0) * 255.0) as u8,
                    255,
                ]);
            }
        }
        Format::R32G32B32A32FLOAT => {
            for chunk in img.pixels.chunks_exact(16) {
                let r = f32::from_le_bytes(chunk[0..4].try_into().unwrap_or([0; 4]));
                let g = f32::from_le_bytes(chunk[4..8].try_into().unwrap_or([0; 4]));
                let b = f32::from_le_bytes(chunk[8..12].try_into().unwrap_or([0; 4]));
                let a = f32::from_le_bytes(chunk[12..16].try_into().unwrap_or([0; 4]));
                out.extend_from_slice(&[
                    (r.clamp(0.0, 1.0) * 255.0) as u8,
                    (g.clamp(0.0, 1.0) * 255.0) as u8,
                    (b.clamp(0.0, 1.0) * 255.0) as u8,
                    (a.clamp(0.0, 1.0) * 255.0) as u8,
                ]);
            }
        }
    }
    if out.len() < px_count * 4 {
        out.resize(px_count * 4, 255);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn append_uv_sphere(
    vertices: &mut Vec<MeshVertex>,
    indices: &mut Vec<u32>,
    center: Vec3,
    radii: Vec3,
    slices: u32,
    stacks: u32,
    joint: u16,
    color: [f32; 4],
) {
    let base_idx = vertices.len() as u32;
    for stack in 0..=stacks {
        let v = stack as f32 / stacks as f32;
        let phi = v * PI;
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();
        for slice in 0..=slices {
            let u = slice as f32 / slices as f32;
            let theta = u * PI * 2.0;
            let sin_theta = theta.sin();
            let cos_theta = theta.cos();
            let dir = Vec3::new(sin_phi * cos_theta, cos_phi, sin_phi * sin_theta);
            let pos = center + dir * radii;
            let normal = (dir / radii.max(Vec3::splat(1e-4))).normalize_or_zero();
            vertices.push(MeshVertex {
                position: pos.to_array(),
                normal: normal.to_array(),
                tangent: [1.0, 0.0, 0.0, 1.0],
                texcoord_0: Vec2::new(u, v).to_array(),
                color_0: color,
                joints_0: [joint, 0, 0, 0],
                weights_0: [1.0, 0.0, 0.0, 0.0],
            });
        }
    }
    let stride = slices + 1;
    for stack in 0..stacks {
        for slice in 0..slices {
            let i0 = base_idx + stack * stride + slice;
            let i1 = i0 + 1;
            let i2 = i0 + stride;
            let i3 = i2 + 1;
            indices.extend_from_slice(&[i0, i1, i2, i1, i3, i2]);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn append_cone_frustum(
    vertices: &mut Vec<MeshVertex>,
    indices: &mut Vec<u32>,
    p0: Vec3,
    p1: Vec3,
    r0: f32,
    r1: f32,
    segments: u32,
    joint: u16,
    color: [f32; 4],
) {
    let axis = (p1 - p0).normalize_or_zero();
    if axis.length_squared() < 1e-6 {
        return;
    }
    let ref_up = if axis.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let tangent = axis.cross(ref_up).normalize();
    let bitangent = axis.cross(tangent).normalize();

    let base_idx = vertices.len() as u32;
    for ring_idx in 0..=1 {
        let (center, radius, v_coord) = if ring_idx == 0 {
            (p0, r0, 0.0)
        } else {
            (p1, r1, 1.0)
        };
        for seg in 0..=segments {
            let u = seg as f32 / segments as f32;
            let angle = u * PI * 2.0;
            let radial = tangent * angle.cos() + bitangent * angle.sin();
            let pos = center + radial * radius;
            let normal = (radial + axis * ((r0 - r1) / p0.distance(p1).max(1e-4))).normalize();
            vertices.push(MeshVertex {
                position: pos.to_array(),
                normal: normal.to_array(),
                tangent: [tangent.x, tangent.y, tangent.z, 1.0],
                texcoord_0: [u, v_coord],
                color_0: color,
                joints_0: [joint, 0, 0, 0],
                weights_0: [1.0, 0.0, 0.0, 0.0],
            });
        }
    }
    let stride = segments + 1;
    for seg in 0..segments {
        let i0 = base_idx + seg;
        let i1 = i0 + 1;
        let i2 = i0 + stride;
        let i3 = i2 + 1;
        indices.extend_from_slice(&[i0, i1, i2, i1, i3, i2]);
    }
}
