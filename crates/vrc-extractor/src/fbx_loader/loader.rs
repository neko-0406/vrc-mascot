use std::collections::HashMap;
use std::path::Path;

use glam::{Vec3, Vec4};

use crate::error::ExtractorError;

/// Node extracted from an FBX scene in glTF right-handed Y-up meter coordinates.
#[derive(Debug, Clone)]
pub struct FbxNodeData {
    pub name: String,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub mesh_index: Option<usize>,
}

/// Morph target delta arrays aligned with its parent `FbxPrimitive` vertices.
#[derive(Debug, Clone)]
pub struct FbxMorphTarget {
    pub name: String,
    pub delta_positions: Vec<[f32; 3]>,
    pub delta_normals: Vec<[f32; 3]>,
}

/// Single triangulated material-part primitive extracted from an FBX mesh.
#[derive(Debug, Clone)]
pub struct FbxPrimitive {
    pub material_slot: usize,
    pub material_name: String,
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub tangents: Option<Vec<[f32; 4]>>,
    pub texcoords_0: Option<Vec<[f32; 2]>>,
    pub colors_0: Option<Vec<[f32; 4]>>,
    pub joints_0: Option<Vec<[u16; 4]>>,
    pub weights_0: Option<Vec<[f32; 4]>>,
    pub indices: Vec<u32>,
    pub morph_targets: Vec<FbxMorphTarget>,
}

/// Skinning joint list and inverse bind matrices for an `FbxMeshData`.
#[derive(Debug, Clone, Default)]
pub struct FbxSkinData {
    pub joint_names: Vec<String>,
    pub inverse_bind_matrices: Vec<[f32; 16]>,
}

/// Extracted mesh with primitives, material slot names, morph target names, and optional skinning.
#[derive(Debug, Clone)]
pub struct FbxMeshData {
    pub name: String,
    pub material_names: Vec<String>,
    pub primitives: Vec<FbxPrimitive>,
    pub morph_target_names: Vec<String>,
    pub skin: Option<FbxSkinData>,
}

/// Entire FBX scene converted to glTF right-handed Y-up meter space.
#[derive(Debug, Clone, Default)]
pub struct FbxSceneData {
    pub nodes: Vec<FbxNodeData>,
    pub meshes: Vec<FbxMeshData>,
}

/// Converts a Unity Left-Handed Y-up transform into glTF Right-Handed Y-up by mirroring X.
pub fn unity_lh_to_gltf_rh_transform(
    pos: [f32; 3],
    rot: [f32; 4],
    scale: [f32; 3],
) -> ([f32; 3], [f32; 4], [f32; 3]) {
    let rh_pos = [-pos[0], pos[1], pos[2]];
    let rh_rot = [rot[0], -rot[1], -rot[2], rot[3]];
    (rh_pos, rh_rot, scale)
}

/// Loads an FBX file using `ufbx`, normalizing coordinates to glTF RH Y-up meters.
pub fn load_fbx_scene(fbx_path: &Path) -> Result<FbxSceneData, ExtractorError> {
    load_fbx_with_opts(fbx_path, false)
}

/// Loads only the FBX node hierarchy and mesh/material metadata (`ignore_geometry: true`)
/// for fast scene/prefab resolution without decoding vertex buffers.
pub fn load_fbx_hierarchy(fbx_path: &Path) -> Result<FbxSceneData, ExtractorError> {
    load_fbx_with_opts(fbx_path, true)
}

fn load_fbx_with_opts(
    fbx_path: &Path,
    ignore_geometry: bool,
) -> Result<FbxSceneData, ExtractorError> {
    let path_str = fbx_path.to_str().ok_or_else(|| ExtractorError::FbxLoad {
        path: fbx_path.to_path_buf(),
        message: "non-UTF-8 FBX file path".to_owned(),
    })?;

    let opts = ufbx::LoadOpts {
        target_axes: ufbx::CoordinateAxes::right_handed_y_up(),
        target_unit_meters: 1.0,
        space_conversion: ufbx::SpaceConversion::ModifyGeometry,
        generate_missing_normals: !ignore_geometry,
        normalize_normals: !ignore_geometry,
        normalize_tangents: !ignore_geometry,
        clean_skin_weights: !ignore_geometry,
        ignore_geometry,
        ignore_animation: true,
        ..Default::default()
    };

    let scene = ufbx::load_file(path_str, opts).map_err(|err| ExtractorError::FbxLoad {
        path: fbx_path.to_path_buf(),
        message: format!("{err:?}"),
    })?;

    // Map ufbx Mesh pointer/typed_id -> mesh index
    let mut mesh_id_to_index: HashMap<u32, usize> = HashMap::new();
    let mut meshes = Vec::with_capacity(scene.meshes.len());
    for mesh_ref in &scene.meshes {
        let mesh: &ufbx::Mesh = mesh_ref;
        let idx = meshes.len();
        mesh_id_to_index.insert(mesh.element.typed_id, idx);
        if ignore_geometry {
            meshes.push(extract_mesh_metadata(mesh));
        } else {
            meshes.push(extract_mesh(mesh));
        }
    }

    // Map ufbx Node typed_id -> node index (skip synthetic root_node if unnamed or keep consistent)
    let mut node_id_to_index: HashMap<u32, usize> = HashMap::new();
    let mut raw_nodes: Vec<&ufbx::Node> = Vec::new();
    for node_ref in &scene.nodes {
        let node: &ufbx::Node = node_ref;
        if node.is_root {
            continue;
        }
        let idx = raw_nodes.len();
        node_id_to_index.insert(node.element.typed_id, idx);
        raw_nodes.push(node);
    }

    let mut nodes = Vec::with_capacity(raw_nodes.len());
    for node in &raw_nodes {
        let parent = node
            .parent
            .as_ref()
            .and_then(|p| node_id_to_index.get(&p.element.typed_id).copied());
        let children = node
            .children
            .iter()
            .filter_map(|c| node_id_to_index.get(&c.element.typed_id).copied())
            .collect();
        let mesh_index = node
            .mesh
            .as_ref()
            .and_then(|m| mesh_id_to_index.get(&m.element.typed_id).copied());

        let t = node.local_transform.translation;
        let r = node.local_transform.rotation;
        let s = node.local_transform.scale;
        let q = glam::Quat::from_xyzw(r.x as f32, r.y as f32, r.z as f32, r.w as f32).normalize();

        nodes.push(FbxNodeData {
            name: node.element.name.to_string(),
            parent,
            children,
            translation: [t.x as f32, t.y as f32, t.z as f32],
            rotation: q.to_array(),
            scale: [s.x as f32, s.y as f32, s.z as f32],
            mesh_index,
        });
    }

    Ok(FbxSceneData { nodes, meshes })
}

fn collect_mesh_material_names(mesh: &ufbx::Mesh) -> Vec<String> {
    mesh.materials
        .as_ref()
        .iter()
        .map(|m| m.element.name.to_string())
        .collect()
}

fn extract_mesh_metadata(mesh: &ufbx::Mesh) -> FbxMeshData {
    FbxMeshData {
        name: mesh.element.name.to_string(),
        material_names: collect_mesh_material_names(mesh),
        primitives: Vec::new(),
        morph_target_names: Vec::new(),
        skin: None,
    }
}

fn extract_mesh(mesh: &ufbx::Mesh) -> FbxMeshData {
    let mesh_name = mesh.element.name.to_string();
    let material_names = collect_mesh_material_names(mesh);

    // 1. Extract skinning clusters & per-control-vertex top-4 normalized weights
    let (skin_data, ctrl_skin_weights) = extract_skinning(mesh);

    // 2. Collect BlendShape deltas indexed by control-point vertex index
    let blend_channels = extract_blend_shapes(mesh);
    let morph_target_names: Vec<String> = blend_channels.iter().map(|c| c.name.clone()).collect();

    // 3. Build primitives grouped by material_parts (or single part if mesh has no material_parts)
    let mut primitives = Vec::new();
    if mesh.material_parts.count > 0 {
        for part in &mesh.material_parts {
            if part.num_triangles == 0 {
                continue;
            }
            let mat_slot = part.index as usize;
            let mat_name = mesh
                .materials
                .as_ref()
                .get(mat_slot)
                .map(|m| m.element.name.to_string())
                .unwrap_or_default();

            let face_indices: Vec<u32> = part.face_indices.as_ref().to_vec();
            let prim = build_primitive(
                mesh,
                &face_indices,
                mat_slot,
                mat_name,
                ctrl_skin_weights.as_deref(),
                &blend_channels,
            );
            if !prim.indices.is_empty() {
                primitives.push(prim);
            }
        }
    }

    if primitives.is_empty() && mesh.num_faces > 0 {
        let all_faces: Vec<u32> = (0..mesh.num_faces as u32).collect();
        let mat_name = mesh
            .materials
            .as_ref()
            .first()
            .map(|m| m.element.name.to_string())
            .unwrap_or_default();
        let prim = build_primitive(
            mesh,
            &all_faces,
            0,
            mat_name,
            ctrl_skin_weights.as_deref(),
            &blend_channels,
        );
        if !prim.indices.is_empty() {
            primitives.push(prim);
        }
    }

    FbxMeshData {
        name: mesh_name,
        material_names,
        primitives,
        morph_target_names,
        skin: skin_data,
    }
}

type CtrlSkin = ([u16; 4], [f32; 4]);

fn extract_skinning(mesh: &ufbx::Mesh) -> (Option<FbxSkinData>, Option<Vec<CtrlSkin>>) {
    if mesh.skin_deformers.count == 0 {
        return (None, None);
    }
    let deformer = &mesh.skin_deformers[0];
    if deformer.clusters.count == 0 {
        return (None, None);
    }

    let mut joint_names = Vec::with_capacity(deformer.clusters.count);
    let mut inverse_bind_matrices = Vec::with_capacity(deformer.clusters.count);
    let mut cluster_to_joint: Vec<Option<u16>> = Vec::with_capacity(deformer.clusters.count);

    for cluster in &deformer.clusters {
        let Some(bone_node) = cluster.bone_node.as_ref() else {
            cluster_to_joint.push(None);
            continue;
        };
        let joint_idx = joint_names.len() as u16;
        joint_names.push(bone_node.element.name.to_string());
        inverse_bind_matrices.push(ufbx_matrix_to_gltf_mat4(&cluster.geometry_to_bone));
        cluster_to_joint.push(Some(joint_idx));
    }

    if joint_names.is_empty() {
        return (None, None);
    }

    let num_ctrl = mesh.num_vertices;
    let mut ctrl_weights = vec![([0u16; 4], [1.0f32, 0.0, 0.0, 0.0]); num_ctrl];

    for (v_idx, sv) in deformer.vertices.iter().enumerate().take(num_ctrl) {
        let begin = sv.weight_begin as usize;
        let end = (begin + sv.num_weights as usize).min(deformer.weights.len());
        let mut pairs: Vec<(u16, f32)> = deformer.weights.as_ref()[begin..end]
            .iter()
            .filter_map(|w| {
                let j = *cluster_to_joint.get(w.cluster_index as usize)?.as_ref()?;
                let weight = w.weight as f32;
                if weight > 0.0 && weight.is_finite() {
                    Some((j, weight))
                } else {
                    None
                }
            })
            .collect();

        pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
        pairs.truncate(4);

        let sum: f32 = pairs.iter().map(|(_, w)| *w).sum();
        let mut joints = [0u16; 4];
        let mut weights = [0.0f32; 4];
        if sum > 1e-8 {
            for (slot, (j, w)) in pairs.into_iter().enumerate() {
                joints[slot] = j;
                weights[slot] = w / sum;
            }
        } else {
            weights[0] = 1.0;
        }
        ctrl_weights[v_idx] = (joints, weights);
    }

    (
        Some(FbxSkinData {
            joint_names,
            inverse_bind_matrices,
        }),
        Some(ctrl_weights),
    )
}

struct ExtractedBlendChannel {
    name: String,
    pos_deltas: Vec<[f32; 3]>,
    norm_deltas: Vec<[f32; 3]>,
}

fn extract_blend_shapes(mesh: &ufbx::Mesh) -> Vec<ExtractedBlendChannel> {
    let mut out = Vec::new();
    let num_ctrl = mesh.num_vertices;
    if num_ctrl == 0 {
        return out;
    }

    for deformer in &mesh.blend_deformers {
        for channel in &deformer.channels {
            let shape_opt = channel
                .target_shape
                .as_ref()
                .map(|r| r.as_ref())
                .or_else(|| channel.keyframes.last().map(|kf| kf.shape.as_ref()));
            let Some(shape) = shape_opt else {
                continue;
            };

            let name = if !channel.element.name.is_empty() {
                channel.element.name.to_string()
            } else {
                shape.element.name.to_string()
            };

            let mut pos_deltas = vec![[0.0f32; 3]; num_ctrl];
            let mut norm_deltas = vec![[0.0f32; 3]; num_ctrl];

            let has_norms = shape.normal_offsets.len() >= shape.num_offsets;
            let has_weights = shape.offset_weights.len() >= shape.num_offsets;

            for i in 0..shape.num_offsets {
                let ctrl_idx = shape.offset_vertices[i] as usize;
                if ctrl_idx >= num_ctrl {
                    continue;
                }
                let scale = if has_weights {
                    shape.offset_weights[i] as f32
                } else {
                    1.0
                };
                let dp = shape.position_offsets[i];
                pos_deltas[ctrl_idx] = [
                    dp.x as f32 * scale,
                    dp.y as f32 * scale,
                    dp.z as f32 * scale,
                ];
                if has_norms {
                    let dn = shape.normal_offsets[i];
                    norm_deltas[ctrl_idx] = [
                        dn.x as f32 * scale,
                        dn.y as f32 * scale,
                        dn.z as f32 * scale,
                    ];
                }
            }

            out.push(ExtractedBlendChannel {
                name,
                pos_deltas,
                norm_deltas,
            });
        }
    }
    out
}

#[derive(Hash, PartialEq, Eq)]
struct VertexDedupKey {
    ctrl_idx: u32,
    pos_bits: [u32; 3],
    norm_bits: [u32; 3],
    uv_bits: [u32; 2],
    tan_bits: [u32; 4],
    col_bits: [u32; 4],
}

fn build_primitive(
    mesh: &ufbx::Mesh,
    face_indices: &[u32],
    material_slot: usize,
    material_name: String,
    ctrl_skin: Option<&[CtrlSkin]>,
    blend_channels: &[ExtractedBlendChannel],
) -> FbxPrimitive {
    let has_uv = mesh.vertex_uv.exists;
    let has_tangent = mesh.vertex_tangent.exists;
    let has_bitangent = mesh.vertex_bitangent.exists;
    let has_color = mesh.vertex_color.exists;
    let has_skin = ctrl_skin.is_some();

    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut tangents = has_tangent.then(Vec::new);
    let mut texcoords_0 = has_uv.then(Vec::new);
    let mut colors_0 = has_color.then(Vec::new);
    let mut joints_0 = has_skin.then(Vec::new);
    let mut weights_0 = has_skin.then(Vec::new);
    let mut indices = Vec::new();
    let mut out_to_ctrl: Vec<usize> = Vec::new();

    let mut dedup_map: HashMap<VertexDedupKey, u32> = HashMap::new();
    let mut tri_buf = Vec::new();

    for &face_idx in face_indices {
        let Some(&face) = mesh.faces.as_ref().get(face_idx as usize) else {
            continue;
        };
        if face.num_indices < 3 {
            continue;
        }
        let num_tris = ufbx::triangulate_face_vec(&mut tri_buf, mesh, face);
        for corner_idx in tri_buf.iter().copied().take(num_tris as usize * 3) {
            let c = corner_idx as usize;
            let ctrl_idx = *mesh.vertex_indices.as_ref().get(c).unwrap_or(&0);

            let p = if mesh.vertex_position.exists {
                let v = mesh.vertex_position[c];
                [v.x as f32, v.y as f32, v.z as f32]
            } else if let Some(v) = mesh.vertices.as_ref().get(ctrl_idx as usize) {
                [v.x as f32, v.y as f32, v.z as f32]
            } else {
                [0.0, 0.0, 0.0]
            };

            let n = if mesh.vertex_normal.exists {
                let v = mesh.vertex_normal[c];
                Vec3::new(v.x as f32, v.y as f32, v.z as f32)
                    .normalize_or(Vec3::Y)
                    .to_array()
            } else {
                [0.0, 1.0, 0.0]
            };

            let uv = if has_uv {
                let v = mesh.vertex_uv[c];
                [v.x as f32, 1.0 - v.y as f32]
            } else {
                [0.0, 0.0]
            };

            let tan = if has_tangent {
                let t_raw = mesh.vertex_tangent[c];
                let t_vec =
                    Vec3::new(t_raw.x as f32, t_raw.y as f32, t_raw.z as f32).normalize_or(Vec3::X);
                let w = if has_bitangent {
                    let b_raw = mesh.vertex_bitangent[c];
                    let b_vec = Vec3::new(b_raw.x as f32, b_raw.y as f32, b_raw.z as f32);
                    let n_vec = Vec3::from_array(n);
                    if n_vec.cross(t_vec).dot(b_vec) < 0.0 {
                        -1.0
                    } else {
                        1.0
                    }
                } else {
                    1.0
                };
                Vec4::new(t_vec.x, t_vec.y, t_vec.z, w).to_array()
            } else {
                [1.0, 0.0, 0.0, 1.0]
            };

            let col = if has_color {
                let v = mesh.vertex_color[c];
                [v.x as f32, v.y as f32, v.z as f32, v.w as f32]
            } else {
                [1.0, 1.0, 1.0, 1.0]
            };

            let key = VertexDedupKey {
                ctrl_idx,
                pos_bits: [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()],
                norm_bits: [n[0].to_bits(), n[1].to_bits(), n[2].to_bits()],
                uv_bits: [uv[0].to_bits(), uv[1].to_bits()],
                tan_bits: [
                    tan[0].to_bits(),
                    tan[1].to_bits(),
                    tan[2].to_bits(),
                    tan[3].to_bits(),
                ],
                col_bits: [
                    col[0].to_bits(),
                    col[1].to_bits(),
                    col[2].to_bits(),
                    col[3].to_bits(),
                ],
            };

            let out_idx = if let Some(&existing) = dedup_map.get(&key) {
                existing
            } else {
                let new_idx = positions.len() as u32;
                dedup_map.insert(key, new_idx);
                positions.push(p);
                normals.push(n);
                if let Some(ref mut ts) = tangents {
                    ts.push(tan);
                }
                if let Some(ref mut uvs) = texcoords_0 {
                    uvs.push(uv);
                }
                if let Some(ref mut cols) = colors_0 {
                    cols.push(col);
                }
                if let Some(skin_slice) = ctrl_skin {
                    let (j, w) = skin_slice
                        .get(ctrl_idx as usize)
                        .copied()
                        .unwrap_or(([0; 4], [1.0, 0.0, 0.0, 0.0]));
                    if let Some(ref mut js) = joints_0 {
                        js.push(j);
                    }
                    if let Some(ref mut ws) = weights_0 {
                        ws.push(w);
                    }
                }
                out_to_ctrl.push(ctrl_idx as usize);
                new_idx
            };

            indices.push(out_idx);
        }
    }

    let morph_targets = blend_channels
        .iter()
        .map(|ch| {
            let mut delta_positions = Vec::with_capacity(out_to_ctrl.len());
            let mut delta_normals = Vec::with_capacity(out_to_ctrl.len());
            for &ctrl in &out_to_ctrl {
                delta_positions.push(ch.pos_deltas.get(ctrl).copied().unwrap_or([0.0; 3]));
                delta_normals.push(ch.norm_deltas.get(ctrl).copied().unwrap_or([0.0; 3]));
            }
            FbxMorphTarget {
                name: ch.name.clone(),
                delta_positions,
                delta_normals,
            }
        })
        .collect();

    FbxPrimitive {
        material_slot,
        material_name,
        positions,
        normals,
        tangents,
        texcoords_0,
        colors_0,
        joints_0,
        weights_0,
        indices,
        morph_targets,
    }
}

fn ufbx_matrix_to_gltf_mat4(m: &ufbx::Matrix) -> [f32; 16] {
    [
        m.m00 as f32,
        m.m10 as f32,
        m.m20 as f32,
        0.0,
        m.m01 as f32,
        m.m11 as f32,
        m.m21 as f32,
        0.0,
        m.m02 as f32,
        m.m12 as f32,
        m.m22 as f32,
        0.0,
        m.m03 as f32,
        m.m13 as f32,
        m.m23 as f32,
        1.0,
    ]
}
