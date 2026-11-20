use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};

use gjson::validation::{Checked, USize64, Validate};
use gltf::json as gjson;
use image::imageops::FilterType;
use image::ImageReader;
use vrc_schema::{AlphaMode, CullMode, MaterialParams};

use crate::error::ExtractorError;
use crate::fbx_loader::FbxMeshData;

/// Node input for building the glTF 2.0 scene graph.
#[derive(Debug, Clone)]
pub struct GlbNodeInput {
    pub name: String,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub mesh: Option<usize>,
    pub skin: Option<usize>,
}

/// Mesh + material slot mapping input for glTF export.
#[derive(Debug, Clone)]
pub struct GlbMeshAttachment {
    pub name: String,
    pub fbx_mesh: FbxMeshData,
    /// Maps FBX primitive `material_slot` -> global glTF material index.
    pub material_indices: Vec<usize>,
}

/// Skinning input linking a glTF mesh node to skeleton joint node indices.
#[derive(Debug, Clone)]
pub struct GlbSkinInput {
    pub name: String,
    pub skeleton_root: Option<usize>,
    pub joints: Vec<usize>,
    pub inverse_bind_matrices: Vec<[f32; 16]>,
}

/// Material & optional texture file paths for glTF export.
#[derive(Debug, Clone)]
pub struct GlbMaterialInput {
    pub name: String,
    pub alpha_mode: AlphaMode,
    pub params: MaterialParams,
    pub main_texture_path: Option<PathBuf>,
    pub matcap_texture_path: Option<PathBuf>,
    pub emission_texture_path: Option<PathBuf>,
    pub fur_mask_texture_path: Option<PathBuf>,
}

/// Resolved texture indices generated during GLB export so `MaterialDef` stays synchronized.
#[derive(Debug, Clone, Default)]
pub struct ExportedMaterialTextures {
    pub matcap_texture_index: Option<usize>,
    pub emission_texture_index: Option<usize>,
    pub fur_mask_texture_index: Option<usize>,
}

/// Full input bundle for generating `avatar.glb`.
#[derive(Debug, Clone)]
pub struct GlbExportInput {
    pub nodes: Vec<GlbNodeInput>,
    pub root_nodes: Vec<usize>,
    pub meshes: Vec<GlbMeshAttachment>,
    pub skins: Vec<GlbSkinInput>,
    pub materials: Vec<GlbMaterialInput>,
    pub max_texture_size: u32,
}

/// Exports the assembled scene, meshes, morph targets, skins, materials, and PNG textures to a GLB 2.0 file.
pub fn export_glb(
    output_path: &Path,
    input: &GlbExportInput,
    warnings: &mut Vec<String>,
) -> Result<Vec<ExportedMaterialTextures>, ExtractorError> {
    if let Some(parent) = output_path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }

    let mut root = gjson::Root {
        asset: gjson::Asset {
            version: "2.0".to_owned(),
            generator: Some("vrc-extractor".to_owned()),
            ..Default::default()
        },
        ..Default::default()
    };

    let mut bin_data: Vec<u8> = Vec::new();
    let mut texture_cache: HashMap<PathBuf, usize> = HashMap::new();
    let mut default_sampler_idx: Option<gjson::Index<gjson::texture::Sampler>> = None;

    let mut ensure_sampler = |root: &mut gjson::Root| -> gjson::Index<gjson::texture::Sampler> {
        if let Some(idx) = default_sampler_idx {
            return idx;
        }
        let sampler = gjson::texture::Sampler {
            mag_filter: Some(Checked::Valid(gjson::texture::MagFilter::Linear)),
            min_filter: Some(Checked::Valid(
                gjson::texture::MinFilter::LinearMipmapLinear,
            )),
            wrap_s: Checked::Valid(gjson::texture::WrappingMode::Repeat),
            wrap_t: Checked::Valid(gjson::texture::WrappingMode::Repeat),
            ..Default::default()
        };
        let idx = gjson::Index::push(&mut root.samplers, sampler);
        default_sampler_idx = Some(idx);
        idx
    };

    let mut load_or_reuse_texture = |tex_path: &Path,
                                     root: &mut gjson::Root,
                                     bin: &mut Vec<u8>,
                                     warnings: &mut Vec<String>|
     -> Option<usize> {
        let key = tex_path.to_path_buf();
        if let Some(&existing) = texture_cache.get(&key) {
            return Some(existing);
        }
        if !tex_path.is_file() {
            warnings.push(format!("Texture file not found: {}", tex_path.display()));
            return None;
        }
        let png_bytes = match encode_texture_png(tex_path, input.max_texture_size) {
            Ok(bytes) => bytes,
            Err(err) => {
                warnings.push(format!(
                    "Failed to process texture '{}': {}",
                    tex_path.display(),
                    err
                ));
                return None;
            }
        };

        let bv_idx = push_buffer_view(root, bin, &png_bytes, None);
        let img_idx = gjson::Index::push(
            &mut root.images,
            gjson::Image {
                buffer_view: Some(bv_idx),
                mime_type: Some(gjson::image::MimeType("image/png".to_owned())),
                name: tex_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(str::to_owned),
                uri: None,
                extensions: None,
                extras: Default::default(),
            },
        );
        let sampler_idx = ensure_sampler(root);
        let tex_idx = gjson::Index::push(
            &mut root.textures,
            gjson::Texture {
                name: tex_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(str::to_owned),
                sampler: Some(sampler_idx),
                source: img_idx,
                extensions: None,
                extras: Default::default(),
            },
        );
        let idx_usize = tex_idx.value();
        texture_cache.insert(key, idx_usize);
        Some(idx_usize)
    };

    // 1. Export Materials & Textures
    let mut mat_tex_indices = Vec::with_capacity(input.materials.len());
    for mat_in in &input.materials {
        let main_tex = mat_in
            .main_texture_path
            .as_deref()
            .and_then(|p| load_or_reuse_texture(p, &mut root, &mut bin_data, warnings));
        let matcap_tex = mat_in
            .matcap_texture_path
            .as_deref()
            .and_then(|p| load_or_reuse_texture(p, &mut root, &mut bin_data, warnings));
        let emission_tex = mat_in
            .emission_texture_path
            .as_deref()
            .and_then(|p| load_or_reuse_texture(p, &mut root, &mut bin_data, warnings));
        let fur_mask_tex = mat_in
            .fur_mask_texture_path
            .as_deref()
            .and_then(|p| load_or_reuse_texture(p, &mut root, &mut bin_data, warnings));

        let gltf_alpha = match mat_in.alpha_mode {
            AlphaMode::Opaque => gjson::material::AlphaMode::Opaque,
            AlphaMode::Mask => gjson::material::AlphaMode::Mask,
            AlphaMode::Blend => gjson::material::AlphaMode::Blend,
        };
        let alpha_cutoff = if mat_in.alpha_mode == AlphaMode::Mask {
            Some(gjson::material::AlphaCutoff(mat_in.params.alpha_cutoff))
        } else {
            None
        };
        let emissive_factor = if mat_in.params.emission_enable {
            gjson::material::EmissiveFactor([
                mat_in.params.emission_color[0].clamp(0.0, 1.0),
                mat_in.params.emission_color[1].clamp(0.0, 1.0),
                mat_in.params.emission_color[2].clamp(0.0, 1.0),
            ])
        } else {
            gjson::material::EmissiveFactor([0.0, 0.0, 0.0])
        };

        root.materials.push(gjson::Material {
            name: Some(mat_in.name.clone()),
            alpha_mode: Checked::Valid(gltf_alpha),
            alpha_cutoff,
            double_sided: mat_in.params.cull_mode == CullMode::None,
            pbr_metallic_roughness: gjson::material::PbrMetallicRoughness {
                base_color_factor: gjson::material::PbrBaseColorFactor(mat_in.params.main_color),
                base_color_texture: main_tex.map(|i| gjson::texture::Info {
                    index: gjson::Index::new(i as u32),
                    tex_coord: 0,
                    extensions: None,
                    extras: Default::default(),
                }),
                metallic_factor: gjson::material::StrengthFactor(0.0),
                roughness_factor: gjson::material::StrengthFactor(1.0),
                metallic_roughness_texture: None,
                extensions: None,
                extras: Default::default(),
            },
            emissive_factor,
            emissive_texture: emission_tex.map(|i| gjson::texture::Info {
                index: gjson::Index::new(i as u32),
                tex_coord: 0,
                extensions: None,
                extras: Default::default(),
            }),
            normal_texture: None,
            occlusion_texture: None,
            extensions: None,
            extras: Default::default(),
        });

        mat_tex_indices.push(ExportedMaterialTextures {
            matcap_texture_index: matcap_tex,
            emission_texture_index: emission_tex,
            fur_mask_texture_index: fur_mask_tex,
        });
    }

    // 2. Export Meshes & Morph Targets
    let mut mesh_target_names: Vec<Vec<String>> = Vec::with_capacity(input.meshes.len());
    for mesh_att in &input.meshes {
        let mut primitives = Vec::new();
        for prim in &mesh_att.fbx_mesh.primitives {
            if prim.positions.is_empty() || prim.indices.is_empty() {
                continue;
            }
            let mut attributes = BTreeMap::new();

            let (pos_min, pos_max) = compute_vec3_bounds(&prim.positions);
            let pos_acc = push_vec3_accessor(
                &mut root,
                &mut bin_data,
                &prim.positions,
                Some(gjson::buffer::Target::ArrayBuffer),
                Some((pos_min, pos_max)),
            );
            attributes.insert(Checked::Valid(gjson::mesh::Semantic::Positions), pos_acc);

            let norm_acc = push_vec3_accessor(
                &mut root,
                &mut bin_data,
                &prim.normals,
                Some(gjson::buffer::Target::ArrayBuffer),
                None,
            );
            attributes.insert(Checked::Valid(gjson::mesh::Semantic::Normals), norm_acc);

            if let Some(ref tangents) = prim.tangents {
                if tangents.len() == prim.positions.len() {
                    let tan_acc = push_vec4_accessor(
                        &mut root,
                        &mut bin_data,
                        tangents,
                        Some(gjson::buffer::Target::ArrayBuffer),
                    );
                    attributes.insert(Checked::Valid(gjson::mesh::Semantic::Tangents), tan_acc);
                }
            }

            if let Some(ref uvs) = prim.texcoords_0 {
                if uvs.len() == prim.positions.len() {
                    let uv_acc = push_vec2_accessor(
                        &mut root,
                        &mut bin_data,
                        uvs,
                        Some(gjson::buffer::Target::ArrayBuffer),
                    );
                    attributes.insert(Checked::Valid(gjson::mesh::Semantic::TexCoords(0)), uv_acc);
                }
            }

            if let Some(ref cols) = prim.colors_0 {
                if cols.len() == prim.positions.len() {
                    let col_acc = push_vec4_accessor(
                        &mut root,
                        &mut bin_data,
                        cols,
                        Some(gjson::buffer::Target::ArrayBuffer),
                    );
                    attributes.insert(Checked::Valid(gjson::mesh::Semantic::Colors(0)), col_acc);
                }
            }

            if let (Some(ref joints), Some(ref weights)) = (&prim.joints_0, &prim.weights_0) {
                if joints.len() == prim.positions.len() && weights.len() == prim.positions.len() {
                    let j_acc = push_u16x4_accessor(
                        &mut root,
                        &mut bin_data,
                        joints,
                        Some(gjson::buffer::Target::ArrayBuffer),
                    );
                    let w_acc = push_vec4_accessor(
                        &mut root,
                        &mut bin_data,
                        weights,
                        Some(gjson::buffer::Target::ArrayBuffer),
                    );
                    attributes.insert(Checked::Valid(gjson::mesh::Semantic::Joints(0)), j_acc);
                    attributes.insert(Checked::Valid(gjson::mesh::Semantic::Weights(0)), w_acc);
                }
            }

            let idx_acc = push_u32_index_accessor(&mut root, &mut bin_data, &prim.indices);

            let mat_idx = mesh_att
                .material_indices
                .get(prim.material_slot)
                .copied()
                .or_else(|| mesh_att.material_indices.first().copied())
                .filter(|&m| m < root.materials.len())
                .map(|m| gjson::Index::new(m as u32));

            let targets = if prim.morph_targets.is_empty() {
                None
            } else {
                let mut morphs = Vec::with_capacity(prim.morph_targets.len());
                for mt in &prim.morph_targets {
                    let mut mt_map = BTreeMap::new();
                    let (dmin, dmax) = compute_vec3_bounds(&mt.delta_positions);
                    let dp_acc = push_vec3_accessor(
                        &mut root,
                        &mut bin_data,
                        &mt.delta_positions,
                        Some(gjson::buffer::Target::ArrayBuffer),
                        Some((dmin, dmax)),
                    );
                    mt_map.insert(Checked::Valid(gjson::mesh::Semantic::Positions), dp_acc);

                    if mt.delta_normals.len() == prim.positions.len() {
                        let dn_acc = push_vec3_accessor(
                            &mut root,
                            &mut bin_data,
                            &mt.delta_normals,
                            Some(gjson::buffer::Target::ArrayBuffer),
                            None,
                        );
                        mt_map.insert(Checked::Valid(gjson::mesh::Semantic::Normals), dn_acc);
                    }

                    morphs.push(gjson::mesh::MorphTarget {
                        positions: mt_map
                            .get(&Checked::Valid(gjson::mesh::Semantic::Positions))
                            .copied(),
                        normals: mt_map
                            .get(&Checked::Valid(gjson::mesh::Semantic::Normals))
                            .copied(),
                        tangents: None,
                    });
                }
                Some(morphs)
            };

            primitives.push(gjson::mesh::Primitive {
                attributes,
                indices: Some(idx_acc),
                material: mat_idx,
                mode: Checked::Valid(gjson::mesh::Mode::Triangles),
                targets,
                extensions: None,
                extras: Default::default(),
            });
        }

        let weights = if mesh_att.fbx_mesh.morph_target_names.is_empty() {
            None
        } else {
            Some(vec![0.0; mesh_att.fbx_mesh.morph_target_names.len()])
        };

        root.meshes.push(gjson::Mesh {
            name: Some(mesh_att.name.clone()),
            primitives,
            weights,
            extensions: None,
            extras: Default::default(),
        });
        mesh_target_names.push(mesh_att.fbx_mesh.morph_target_names.clone());
    }

    // 3. Export Skins
    for skin_in in &input.skins {
        if skin_in.joints.is_empty() {
            continue;
        }
        let ibm_acc = if skin_in.inverse_bind_matrices.len() == skin_in.joints.len() {
            Some(push_mat4_accessor(
                &mut root,
                &mut bin_data,
                &skin_in.inverse_bind_matrices,
            ))
        } else {
            None
        };
        root.skins.push(gjson::Skin {
            name: Some(skin_in.name.clone()),
            inverse_bind_matrices: ibm_acc,
            joints: skin_in
                .joints
                .iter()
                .map(|&j| gjson::Index::new(j as u32))
                .collect(),
            skeleton: skin_in.skeleton_root.map(|s| gjson::Index::new(s as u32)),
            extensions: None,
            extras: Default::default(),
        });
    }

    // 4. Export Nodes & Default Scene
    for node_in in &input.nodes {
        let children = if node_in.children.is_empty() {
            None
        } else {
            Some(
                node_in
                    .children
                    .iter()
                    .map(|&c| gjson::Index::new(c as u32))
                    .collect(),
            )
        };
        let rot = glam::Quat::from_array(node_in.rotation).normalize();
        root.nodes.push(gjson::Node {
            name: Some(node_in.name.clone()),
            children,
            translation: Some(node_in.translation),
            rotation: Some(gjson::scene::UnitQuaternion(rot.to_array())),
            scale: Some(node_in.scale),
            mesh: node_in
                .mesh
                .filter(|&m| m < root.meshes.len())
                .map(|m| gjson::Index::new(m as u32)),
            skin: node_in
                .skin
                .filter(|&s| s < root.skins.len())
                .map(|s| gjson::Index::new(s as u32)),
            camera: None,
            matrix: None,
            weights: None,
            extensions: None,
            extras: Default::default(),
        });
    }

    root.scenes.push(gjson::Scene {
        name: Some("AvatarScene".to_owned()),
        nodes: input
            .root_nodes
            .iter()
            .filter(|&&r| r < root.nodes.len())
            .map(|&r| gjson::Index::new(r as u32))
            .collect(),
        extensions: None,
        extras: Default::default(),
    });
    root.scene = Some(gjson::Index::new(0));

    // 5. Finalize single GLB BIN buffer
    if !bin_data.is_empty() {
        while !bin_data.len().is_multiple_of(4) {
            bin_data.push(0);
        }
        root.buffers.push(gjson::Buffer {
            byte_length: USize64(bin_data.len() as u64),
            name: Some("buffer".to_owned()),
            uri: None,
            extensions: None,
            extras: Default::default(),
        });
    }

    // Validate glTF JSON structure
    let mut validation_errors = Vec::new();
    root.validate(&root, gjson::Path::new, &mut |path, err| {
        validation_errors.push(format!("{}: {:?}", path().as_str(), err));
    });
    if !validation_errors.is_empty() {
        return Err(ExtractorError::GltfExport(format!(
            "glTF validation failed: {}",
            validation_errors.join("; ")
        )));
    }

    // Serialize to Value so we can inject `extras.targetNames` on meshes with morph targets
    let mut root_value = serde_json::to_value(&root)?;
    if let Some(meshes_arr) = root_value.get_mut("meshes").and_then(|v| v.as_array_mut()) {
        for (i, names) in mesh_target_names.iter().enumerate() {
            if !names.is_empty() {
                if let Some(mesh_obj) = meshes_arr.get_mut(i).and_then(|v| v.as_object_mut()) {
                    mesh_obj.insert(
                        "extras".to_owned(),
                        serde_json::json!({ "targetNames": names }),
                    );
                }
            }
        }
    }

    let json_bytes = serde_json::to_vec(&root_value)?;
    let glb = gltf::binary::Glb {
        header: gltf::binary::Header {
            magic: *b"glTF",
            version: 2,
            length: 0, // recomputed by Glb::to_vec()
        },
        json: Cow::Borrowed(&json_bytes),
        bin: if bin_data.is_empty() {
            None
        } else {
            Some(Cow::Borrowed(&bin_data))
        },
    };
    let bytes = glb
        .to_vec()
        .map_err(|e| ExtractorError::GltfExport(e.to_string()))?;
    fs::write(output_path, bytes)?;

    Ok(mat_tex_indices)
}

fn encode_texture_png(path: &Path, max_texture_size: u32) -> Result<Vec<u8>, ExtractorError> {
    let img = ImageReader::open(path)
        .map_err(|e| ExtractorError::ImageError {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?
        .with_guessed_format()
        .map_err(|e| ExtractorError::ImageError {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?
        .decode()
        .map_err(|e| ExtractorError::ImageError {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;

    let resized = if max_texture_size > 0
        && (img.width() > max_texture_size || img.height() > max_texture_size)
    {
        img.resize(max_texture_size, max_texture_size, FilterType::Lanczos3)
    } else {
        img
    };

    let mut out = Vec::new();
    resized
        .write_to(&mut Cursor::new(&mut out), image::ImageFormat::Png)
        .map_err(|e| ExtractorError::ImageError {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;
    Ok(out)
}

fn push_buffer_view(
    root: &mut gjson::Root,
    bin: &mut Vec<u8>,
    bytes: &[u8],
    target: Option<gjson::buffer::Target>,
) -> gjson::Index<gjson::buffer::View> {
    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }
    let byte_offset = bin.len() as u64;
    bin.extend_from_slice(bytes);

    gjson::Index::push(
        &mut root.buffer_views,
        gjson::buffer::View {
            buffer: gjson::Index::new(0),
            byte_length: USize64(bytes.len() as u64),
            byte_offset: Some(USize64(byte_offset)),
            byte_stride: None,
            name: None,
            target: target.map(Checked::Valid),
            extensions: None,
            extras: Default::default(),
        },
    )
}

fn push_vec2_accessor(
    root: &mut gjson::Root,
    bin: &mut Vec<u8>,
    data: &[[f32; 2]],
    target: Option<gjson::buffer::Target>,
) -> gjson::Index<gjson::Accessor> {
    let mut bytes = Vec::with_capacity(data.len() * 8);
    for v in data {
        bytes.extend_from_slice(&v[0].to_le_bytes());
        bytes.extend_from_slice(&v[1].to_le_bytes());
    }
    let bv = push_buffer_view(root, bin, &bytes, target);
    push_accessor_raw(
        root,
        bv,
        data.len(),
        gjson::accessor::ComponentType::F32,
        gjson::accessor::Type::Vec2,
        None,
        None,
    )
}

fn push_vec3_accessor(
    root: &mut gjson::Root,
    bin: &mut Vec<u8>,
    data: &[[f32; 3]],
    target: Option<gjson::buffer::Target>,
    bounds: Option<([f32; 3], [f32; 3])>,
) -> gjson::Index<gjson::Accessor> {
    let mut bytes = Vec::with_capacity(data.len() * 12);
    for v in data {
        bytes.extend_from_slice(&v[0].to_le_bytes());
        bytes.extend_from_slice(&v[1].to_le_bytes());
        bytes.extend_from_slice(&v[2].to_le_bytes());
    }
    let bv = push_buffer_view(root, bin, &bytes, target);
    let (min_val, max_val) = match bounds {
        Some((mn, mx)) => (Some(serde_json::json!(mn)), Some(serde_json::json!(mx))),
        None => (None, None),
    };
    push_accessor_raw(
        root,
        bv,
        data.len(),
        gjson::accessor::ComponentType::F32,
        gjson::accessor::Type::Vec3,
        min_val,
        max_val,
    )
}

fn push_vec4_accessor(
    root: &mut gjson::Root,
    bin: &mut Vec<u8>,
    data: &[[f32; 4]],
    target: Option<gjson::buffer::Target>,
) -> gjson::Index<gjson::Accessor> {
    let mut bytes = Vec::with_capacity(data.len() * 16);
    for v in data {
        bytes.extend_from_slice(&v[0].to_le_bytes());
        bytes.extend_from_slice(&v[1].to_le_bytes());
        bytes.extend_from_slice(&v[2].to_le_bytes());
        bytes.extend_from_slice(&v[3].to_le_bytes());
    }
    let bv = push_buffer_view(root, bin, &bytes, target);
    push_accessor_raw(
        root,
        bv,
        data.len(),
        gjson::accessor::ComponentType::F32,
        gjson::accessor::Type::Vec4,
        None,
        None,
    )
}

fn push_u16x4_accessor(
    root: &mut gjson::Root,
    bin: &mut Vec<u8>,
    data: &[[u16; 4]],
    target: Option<gjson::buffer::Target>,
) -> gjson::Index<gjson::Accessor> {
    let mut bytes = Vec::with_capacity(data.len() * 8);
    for v in data {
        for &j in v {
            bytes.extend_from_slice(&j.to_le_bytes());
        }
    }
    let bv = push_buffer_view(root, bin, &bytes, target);
    push_accessor_raw(
        root,
        bv,
        data.len(),
        gjson::accessor::ComponentType::U16,
        gjson::accessor::Type::Vec4,
        None,
        None,
    )
}

fn push_u32_index_accessor(
    root: &mut gjson::Root,
    bin: &mut Vec<u8>,
    data: &[u32],
) -> gjson::Index<gjson::Accessor> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &idx in data {
        bytes.extend_from_slice(&idx.to_le_bytes());
    }
    let bv = push_buffer_view(
        root,
        bin,
        &bytes,
        Some(gjson::buffer::Target::ElementArrayBuffer),
    );
    push_accessor_raw(
        root,
        bv,
        data.len(),
        gjson::accessor::ComponentType::U32,
        gjson::accessor::Type::Scalar,
        None,
        None,
    )
}

fn push_mat4_accessor(
    root: &mut gjson::Root,
    bin: &mut Vec<u8>,
    data: &[[f32; 16]],
) -> gjson::Index<gjson::Accessor> {
    let mut bytes = Vec::with_capacity(data.len() * 64);
    for m in data {
        for f in m {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
    }
    let bv = push_buffer_view(root, bin, &bytes, None);
    push_accessor_raw(
        root,
        bv,
        data.len(),
        gjson::accessor::ComponentType::F32,
        gjson::accessor::Type::Mat4,
        None,
        None,
    )
}

fn push_accessor_raw(
    root: &mut gjson::Root,
    buffer_view: gjson::Index<gjson::buffer::View>,
    count: usize,
    comp_type: gjson::accessor::ComponentType,
    accessor_type: gjson::accessor::Type,
    min: Option<serde_json::Value>,
    max: Option<serde_json::Value>,
) -> gjson::Index<gjson::Accessor> {
    gjson::Index::push(
        &mut root.accessors,
        gjson::Accessor {
            buffer_view: Some(buffer_view),
            byte_offset: Some(USize64(0)),
            count: USize64(count as u64),
            component_type: Checked::Valid(gjson::accessor::GenericComponentType(comp_type)),
            type_: Checked::Valid(accessor_type),
            min,
            max,
            name: None,
            normalized: false,
            sparse: None,
            extensions: None,
            extras: Default::default(),
        },
    )
}

fn compute_vec3_bounds(pts: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for p in pts {
        for i in 0..3 {
            if p[i] < min[i] {
                min[i] = p[i];
            }
            if p[i] > max[i] {
                max[i] = p[i];
            }
        }
    }
    if pts.is_empty() {
        return ([0.0; 3], [0.0; 3]);
    }
    (min, max)
}
