//! Unity project external parser and intermediate GLB + `mascot_manifest.json` exporter.

pub mod error;
pub mod exporter;
pub mod fbx_loader;
pub mod indexer;
pub mod modular_avatar;
pub mod yaml_parser;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use vrc_schema::{ExpressionsDef, MascotManifest, MaterialDef, MANIFEST_FILE_NAME, SCHEMA_VERSION};

pub use error::ExtractorError;
use exporter::{
    export_glb, export_manifest, GlbExportInput, GlbMaterialInput, GlbMeshAttachment, GlbNodeInput,
    GlbSkinInput,
};
use fbx_loader::{load_fbx_scene, unity_lh_to_gltf_rh_transform, FbxSceneData};
use indexer::{parse_fbx_meta_human_description, AssetIndex, UnityRef};
use modular_avatar::apply_modular_avatar;
use yaml_parser::avatar::{
    build_viseme_map_for_mesh, extract_avatar_descriptor_data, extract_physbones_and_colliders,
    find_avatar_candidates, parse_expression_parameters_asset, parse_expressions_menu_asset,
};
use yaml_parser::helpers::{yaml_as_virtual_id, yaml_get};
use yaml_parser::prefab::VirtualId;
use yaml_parser::{
    parse_animator_controller, parse_material_asset, ExtractedMaterial, SceneHierarchy,
    VirtualScene,
};

/// Extraction options for converting a Unity VRChat avatar project into `avatar.glb` + `mascot_manifest.json`.
#[derive(Debug, Clone)]
pub struct ExtractOptions {
    pub project_dir: PathBuf,
    pub scene_path: PathBuf,
    pub avatar_name: String,
    pub output_dir: PathBuf,
    pub modular_avatar_enabled: bool,
    pub max_texture_size: u32,
}

/// Alias kept for backward compatibility with earlier skeleton callers.
pub type ExtractRequest = ExtractOptions;

/// Progress notification emitted during [`extract_avatar`].
#[derive(Debug, Clone)]
pub struct ExtractProgress {
    pub stage: String,
    pub progress: f32,
}

/// Result returned by [`extract_avatar`].
#[derive(Debug, Clone)]
pub struct ExtractReport {
    pub glb_path: PathBuf,
    pub manifest_path: PathBuf,
    pub avatar_name: String,
    pub warnings: Vec<String>,
}

/// Candidate avatar GameObject discovered inside a Unity scene.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredAvatar {
    pub name: String,
    pub has_descriptor: bool,
}

/// Discovers all `.unity` scene files under `<project_dir>/Assets`, returning paths relative to `project_dir`.
pub fn discover_scenes(project_dir: &Path) -> Result<Vec<PathBuf>, ExtractorError> {
    if !project_dir.is_dir() {
        return Err(ExtractorError::InvalidInput(format!(
            "Unity project directory does not exist: {}",
            project_dir.display()
        )));
    }
    let assets_dir = project_dir.join("Assets");
    let mut scenes = Vec::new();
    if assets_dir.is_dir() {
        collect_scenes_recursive(project_dir, &assets_dir, &mut scenes)?;
    }
    scenes.sort();
    Ok(scenes)
}

fn collect_scenes_recursive(
    project_dir: &Path,
    dir: &Path,
    out: &mut Vec<PathBuf>,
) -> Result<(), ExtractorError> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_scenes_recursive(project_dir, &path, out)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some("unity") {
            if let Ok(rel) = path.strip_prefix(project_dir) {
                out.push(rel.to_path_buf());
            }
        }
    }
    Ok(())
}

/// Discovers avatar candidates in `scene_path` (GameObjects with `VRCAvatarDescriptor`, or root nodes as fallback).
pub fn discover_avatars(
    project_dir: &Path,
    scene_path: &Path,
) -> Result<Vec<DiscoveredAvatar>, ExtractorError> {
    let index = AssetIndex::build(project_dir)?;
    let abs_scene = index.resolve_scene_path(scene_path);
    if !abs_scene.is_file() {
        return Err(ExtractorError::InvalidInput(format!(
            "Scene file does not exist: {}",
            abs_scene.display()
        )));
    }
    let (vscene, _) = VirtualScene::load_and_resolve(&abs_scene, &index)?;
    let hierarchy = SceneHierarchy::from_virtual_scene(&vscene);

    let desc_ids = find_avatar_candidates(&hierarchy, &index);
    let mut out = Vec::new();
    let mut seen_names = HashSet::new();

    if !desc_ids.is_empty() {
        for id in desc_ids {
            if let Some(node) = hierarchy.nodes.get(&id) {
                if !node.name.trim().is_empty() && seen_names.insert(node.name.clone()) {
                    out.push(DiscoveredAvatar {
                        name: node.name.clone(),
                        has_descriptor: true,
                    });
                }
            }
        }
    } else {
        for &id in &hierarchy.roots {
            if let Some(node) = hierarchy.nodes.get(&id) {
                if node.active
                    && !node.name.trim().is_empty()
                    && seen_names.insert(node.name.clone())
                {
                    out.push(DiscoveredAvatar {
                        name: node.name.clone(),
                        has_descriptor: false,
                    });
                }
            }
        }
    }

    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// Extracts an avatar from a Unity project without Unity Editor, writing `avatar.glb` and `mascot_manifest.json`.
pub fn extract_avatar<F>(
    opts: &ExtractOptions,
    mut progress_cb: F,
) -> Result<ExtractReport, ExtractorError>
where
    F: FnMut(ExtractProgress),
{
    if opts.avatar_name.trim().is_empty() {
        return Err(ExtractorError::InvalidInput(
            "avatar_name must not be empty".to_owned(),
        ));
    }

    let mut report_progress = |stage: &str, progress: f32| {
        progress_cb(ExtractProgress {
            stage: stage.to_owned(),
            progress: progress.clamp(0.0, 1.0),
        });
    };

    let mut warnings = Vec::new();

    report_progress("Indexing Unity project", 0.05);
    let index = AssetIndex::build(&opts.project_dir)?;

    let abs_scene = index.resolve_scene_path(&opts.scene_path);
    if !abs_scene.is_file() {
        return Err(ExtractorError::InvalidInput(format!(
            "Scene file does not exist: {}",
            abs_scene.display()
        )));
    }

    report_progress("Resolving Scene & Prefabs", 0.20);
    let (vscene, _) = VirtualScene::load_and_resolve(&abs_scene, &index)?;
    let mut hierarchy = SceneHierarchy::from_virtual_scene(&vscene);

    let avatar_root =
        find_target_avatar_node(&hierarchy, &index, &opts.avatar_name).ok_or_else(|| {
            ExtractorError::AvatarNotFound {
                avatar_name: opts.avatar_name.clone(),
                scene_path: opts.scene_path.clone(),
            }
        })?;

    report_progress("Extracting AvatarDescriptor & Expressions", 0.35);
    let desc_data = extract_avatar_descriptor_data(&hierarchy, avatar_root, &index);
    let mut expressions = ExpressionsDef::default();

    if let Some(ref params_guid) = desc_data.expression_parameters_guid {
        if let Some(path) = index.resolve_guid(params_guid) {
            expressions.parameters = parse_expression_parameters_asset(path)?;
        } else {
            warnings.push(format!(
                "VRCExpressionParameters GUID '{params_guid}' not found"
            ));
        }
    }

    if let Some(ref menu_guid) = desc_data.expressions_menu_guid {
        if let Some(path) = index.resolve_guid(menu_guid) {
            expressions.menu = parse_expressions_menu_asset(path, &index)?;
        } else {
            warnings.push(format!("VRCExpressionsMenu GUID '{menu_guid}' not found"));
        }
    }

    if let Some(ref fx_guid) = desc_data.fx_controller_guid {
        if let Some(path) = index.resolve_guid(fx_guid) {
            let parsed_ctrl = parse_animator_controller(path, &index, &mut warnings)?;
            let mut existing: HashSet<String> = expressions
                .parameters
                .iter()
                .map(|p| p.name.clone())
                .collect();
            for p in parsed_ctrl.parameters {
                if existing.insert(p.name.clone()) {
                    expressions.parameters.push(p);
                }
            }
            expressions.fx_layers = parsed_ctrl.layers;
        } else {
            warnings.push(format!("FX AnimatorController GUID '{fx_guid}' not found"));
        }
    }

    report_progress("Parsing Humanoid rig metadata", 0.45);
    let raw_human_map = collect_human_description_map(&hierarchy, avatar_root, &index);
    let mut humanoid_bones = resolve_humanoid_paths(&hierarchy, avatar_root, &raw_human_map);
    if !humanoid_bones.contains_key("LeftEye") {
        if let Some(ref left_eye) = desc_data.left_eye_node {
            humanoid_bones.insert("LeftEye".to_owned(), left_eye.clone());
        }
    }
    if !humanoid_bones.contains_key("RightEye") {
        if let Some(ref right_eye) = desc_data.right_eye_node {
            humanoid_bones.insert("RightEye".to_owned(), right_eye.clone());
        }
    }

    if opts.modular_avatar_enabled {
        report_progress("Applying Modular Avatar components", 0.60);
        let hb_hash: HashMap<String, String> = humanoid_bones
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        apply_modular_avatar(
            &mut hierarchy,
            avatar_root,
            &hb_hash,
            &mut expressions,
            &index,
            &mut warnings,
        )?;
        // Refresh relative bone paths in case MergeArmature or BoneProxy moved bones
        humanoid_bones = resolve_humanoid_paths(&hierarchy, avatar_root, &raw_human_map);
        if !humanoid_bones.contains_key("LeftEye") {
            if let Some(ref left_eye) = desc_data.left_eye_node {
                humanoid_bones.insert("LeftEye".to_owned(), left_eye.clone());
            }
        }
        if !humanoid_bones.contains_key("RightEye") {
            if let Some(ref right_eye) = desc_data.right_eye_node {
                humanoid_bones.insert("RightEye".to_owned(), right_eye.clone());
            }
        }
    }

    report_progress("Extracting PhysBones & Colliders", 0.70);
    let (phys_bones, colliders) = extract_physbones_and_colliders(&hierarchy, avatar_root, &index);

    report_progress("Loading FBX meshes & Materials", 0.80);
    let (glb_input, mut material_defs) = build_glb_export_input(
        &hierarchy,
        avatar_root,
        &index,
        opts.max_texture_size,
        &mut warnings,
    )?;

    report_progress("Exporting avatar.glb & mascot_manifest.json", 0.90);
    fs::create_dir_all(&opts.output_dir)?;
    let glb_path = opts.output_dir.join("avatar.glb");
    let manifest_path = opts.output_dir.join(MANIFEST_FILE_NAME);

    let exported_textures = export_glb(&glb_path, &glb_input, &mut warnings)?;
    for (mat_def, tex_info) in material_defs.iter_mut().zip(exported_textures.iter()) {
        mat_def.params.matcap_texture_index = tex_info.matcap_texture_index;
        mat_def.params.emission_texture_index = tex_info.emission_texture_index;
        mat_def.params.fur_mask_texture_index = tex_info.fur_mask_texture_index;
    }

    let mut viseme_map = desc_data.viseme_map;
    if viseme_map.sil.is_none()
        && viseme_map.aa.is_none()
        && viseme_map.ee.is_none()
        && viseme_map.ih.is_none()
        && viseme_map.oh.is_none()
        && viseme_map.ou.is_none()
        && !desc_data.viseme_blend_shapes.is_empty()
    {
        if let Some(best_mesh) =
            find_viseme_mesh_by_blendshapes(&glb_input.meshes, &desc_data.viseme_blend_shapes)
        {
            viseme_map = build_viseme_map_for_mesh(&best_mesh.name, &desc_data.viseme_blend_shapes);
        }
    }

    let manifest = MascotManifest {
        schema_version: SCHEMA_VERSION.to_owned(),
        avatar_name: opts.avatar_name.clone(),
        glb_file: "avatar.glb".to_owned(),
        humanoid_bones,
        viseme_map,
        phys_bones,
        colliders,
        expressions,
        materials: material_defs,
    };

    export_manifest(&manifest_path, &manifest)?;

    report_progress("Extraction complete", 1.0);
    Ok(ExtractReport {
        glb_path,
        manifest_path,
        avatar_name: opts.avatar_name.clone(),
        warnings,
    })
}

fn find_target_avatar_node(
    hierarchy: &SceneHierarchy,
    index: &AssetIndex,
    avatar_name: &str,
) -> Option<VirtualId> {
    let candidates = find_avatar_candidates(hierarchy, index);
    if let Some(exact_desc) = candidates.iter().copied().find(|&id| {
        hierarchy
            .nodes
            .get(&id)
            .is_some_and(|n| n.name == avatar_name)
    }) {
        return Some(exact_desc);
    }
    if let Some(exact) = hierarchy.find_by_name(avatar_name) {
        return Some(exact);
    }
    let lower = avatar_name.to_ascii_lowercase();
    let mut keys: Vec<VirtualId> = hierarchy.nodes.keys().copied().collect();
    keys.sort_unstable();
    keys.into_iter().find(|&id| {
        hierarchy
            .nodes
            .get(&id)
            .is_some_and(|n| n.name.to_ascii_lowercase() == lower)
    })
}

fn collect_human_description_map(
    hierarchy: &SceneHierarchy,
    avatar_root: VirtualId,
    index: &AssetIndex,
) -> BTreeMap<String, String> {
    let mut result = BTreeMap::new();
    let mut seen_fbx = HashSet::new();

    for go_id in hierarchy.collect_subtree(avatar_root) {
        let Some(node) = hierarchy.nodes.get(&go_id) else {
            continue;
        };
        for comp in &node.components {
            let Some(mesh_guid) = yaml_get(&comp.data, "m_Mesh")
                .and_then(UnityRef::from_yaml)
                .and_then(|r| r.guid)
            else {
                continue;
            };
            if !seen_fbx.insert(mesh_guid.clone()) {
                continue;
            }
            if let Some(fbx_path) = index.resolve_guid(&mesh_guid) {
                let meta_path = PathBuf::from(format!("{}.meta", fbx_path.display()));
                if meta_path.is_file() {
                    if let Ok(map) = parse_fbx_meta_human_description(&meta_path) {
                        for (human, bone) in map {
                            result.entry(human).or_insert(bone);
                        }
                    }
                }
            }
        }
    }

    result
}

fn resolve_humanoid_paths(
    hierarchy: &SceneHierarchy,
    avatar_root: VirtualId,
    human_to_bone_name: &BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    let subtree = hierarchy.collect_subtree(avatar_root);
    let mut name_to_rel_path: HashMap<String, String> = HashMap::new();
    for go_id in subtree {
        if go_id == avatar_root {
            continue;
        }
        if let (Some(node), Some(rel)) = (
            hierarchy.nodes.get(&go_id),
            hierarchy.relative_path(avatar_root, go_id),
        ) {
            name_to_rel_path.entry(node.name.clone()).or_insert(rel);
        }
    }

    let mut out = BTreeMap::new();
    for (human_name, bone_name) in human_to_bone_name {
        if let Some(rel) = name_to_rel_path.get(bone_name) {
            out.insert(human_name.clone(), rel.clone());
        }
    }
    out
}

fn build_glb_export_input(
    hierarchy: &SceneHierarchy,
    avatar_root: VirtualId,
    index: &AssetIndex,
    max_texture_size: u32,
    warnings: &mut Vec<String>,
) -> Result<(GlbExportInput, Vec<MaterialDef>), ExtractorError> {
    // Decide whether avatar_root itself carries a renderer; if not, emit its active children as glTF root nodes
    // so paths like "Armature/Hips" match glTF scene root paths directly.
    let root_has_renderer = hierarchy.nodes.get(&avatar_root).is_some_and(|n| {
        n.components.iter().any(|c| {
            c.class_id == 137
                || c.class_id == 23
                || c.root_key == "SkinnedMeshRenderer"
                || c.root_key == "MeshRenderer"
        })
    });

    let top_nodes = if root_has_renderer {
        vec![avatar_root]
    } else {
        hierarchy
            .nodes
            .get(&avatar_root)
            .map(|n| {
                n.children
                    .iter()
                    .copied()
                    .filter(|cid| hierarchy.nodes.get(cid).is_some_and(|c| c.active))
                    .collect()
            })
            .unwrap_or_default()
    };

    // Collect active GameObject tree into GlbNodeInputs
    let mut go_to_glb_idx: HashMap<VirtualId, usize> = HashMap::new();
    let mut glb_nodes: Vec<GlbNodeInput> = Vec::new();
    let mut node_order: Vec<VirtualId> = Vec::new();
    let mut root_indices: Vec<usize> = Vec::new();

    for &top_id in &top_nodes {
        let idx = collect_active_glb_nodes(
            hierarchy,
            top_id,
            None,
            &mut go_to_glb_idx,
            &mut glb_nodes,
            &mut node_order,
        );
        if let Some(i) = idx {
            root_indices.push(i);
        }
    }

    // Populate children vectors
    for (idx, &go_id) in node_order.iter().enumerate() {
        if let Some(node) = hierarchy.nodes.get(&go_id) {
            let children: Vec<usize> = node
                .children
                .iter()
                .filter_map(|cid| go_to_glb_idx.get(cid).copied())
                .collect();
            glb_nodes[idx].children = children;
        }
    }

    let mut name_to_glb_idx: HashMap<String, usize> = HashMap::new();
    for (idx, n) in glb_nodes.iter().enumerate() {
        name_to_glb_idx.entry(n.name.clone()).or_insert(idx);
    }

    // Material deduplication & parsing
    let mut mat_cache: HashMap<(String, i64), usize> = HashMap::new();
    let mut glb_materials: Vec<GlbMaterialInput> = Vec::new();
    let mut material_defs: Vec<MaterialDef> = Vec::new();

    let mut resolve_material = |mat_ref: &UnityRef, warnings: &mut Vec<String>| -> Option<usize> {
        let guid = mat_ref.guid.as_ref()?;
        let key = (guid.clone(), mat_ref.file_id);
        if let Some(&idx) = mat_cache.get(&key) {
            return Some(idx);
        }
        let Some(mat_path) = index.resolve_guid(guid) else {
            warnings.push(format!("Material GUID '{guid}' not found"));
            return None;
        };
        let gltf_idx = glb_materials.len();
        match parse_material_asset(mat_path, gltf_idx, index) {
            Ok(ExtractedMaterial {
                def,
                main_tex_guid,
                shadow_tex_guid: _,
                matcap_tex_guid,
                emission_tex_guid,
                fur_mask_tex_guid,
            }) => {
                let resolve_tex =
                    |g: Option<String>| g.and_then(|s| index.resolve_guid(&s).cloned());
                glb_materials.push(GlbMaterialInput {
                    name: def.name.clone(),
                    alpha_mode: def.alpha_mode,
                    params: def.params.clone(),
                    main_texture_path: resolve_tex(main_tex_guid),
                    matcap_texture_path: resolve_tex(matcap_tex_guid),
                    emission_texture_path: resolve_tex(emission_tex_guid),
                    fur_mask_texture_path: resolve_tex(fur_mask_tex_guid),
                });
                material_defs.push(def);
                mat_cache.insert(key, gltf_idx);
                Some(gltf_idx)
            }
            Err(err) => {
                warnings.push(format!(
                    "Failed to parse material '{}': {}",
                    mat_path.display(),
                    err
                ));
                None
            }
        }
    };

    // FBX cache
    let mut fbx_cache: HashMap<PathBuf, FbxSceneData> = HashMap::new();
    let mut glb_meshes: Vec<GlbMeshAttachment> = Vec::new();
    let mut glb_skins: Vec<GlbSkinInput> = Vec::new();

    for (glb_idx, &go_id) in node_order.iter().enumerate() {
        let Some(node) = hierarchy.nodes.get(&go_id) else {
            continue;
        };

        let smr = node
            .components
            .iter()
            .find(|c| c.class_id == 137 || c.root_key == "SkinnedMeshRenderer");
        let mr = node
            .components
            .iter()
            .find(|c| c.class_id == 23 || c.root_key == "MeshRenderer");
        let mf = node
            .components
            .iter()
            .find(|c| c.class_id == 33 || c.root_key == "MeshFilter");

        let (renderer_comp, mesh_ref_yaml) = if let Some(s) = smr {
            (s, yaml_get(&s.data, "m_Mesh"))
        } else if let (Some(r), Some(f)) = (mr, mf) {
            (r, yaml_get(&f.data, "m_Mesh"))
        } else {
            continue;
        };

        let Some(mesh_guid) = mesh_ref_yaml
            .and_then(UnityRef::from_yaml)
            .and_then(|r| r.guid)
        else {
            continue;
        };

        let Some(fbx_path) = index.resolve_guid(&mesh_guid).cloned() else {
            warnings.push(format!(
                "Mesh FBX GUID '{mesh_guid}' on '{}' not found",
                node.name
            ));
            continue;
        };

        if !fbx_cache.contains_key(&fbx_path) {
            let scene_data = load_fbx_scene(&fbx_path)?;
            fbx_cache.insert(fbx_path.clone(), scene_data);
        }
        let fbx_scene = &fbx_cache[&fbx_path];
        if fbx_scene.meshes.is_empty() {
            continue;
        }

        let fbx_mesh = select_fbx_mesh(fbx_scene, &node.name).clone();

        // Resolve materials (preserving slot alignment for multi-material FBX meshes)
        let mut mat_indices: Vec<Option<usize>> = Vec::new();
        if let Some(mats_arr) =
            yaml_get(&renderer_comp.data, "m_Materials").and_then(|v| v.as_vec())
        {
            for m_yaml in mats_arr {
                let resolved = UnityRef::from_yaml(m_yaml)
                    .filter(|r| !r.is_null())
                    .and_then(|m_ref| resolve_material(&m_ref, warnings));
                mat_indices.push(resolved);
            }
        }

        // Resolve Skin if present
        if let Some(skin_data) = &fbx_mesh.skin {
            let unity_bones: Vec<Option<usize>> = yaml_get(&renderer_comp.data, "m_Bones")
                .and_then(|v| v.as_vec())
                .map(|arr| {
                    arr.iter()
                        .map(|b| {
                            yaml_as_virtual_id(b).and_then(|vid| {
                                let go =
                                    hierarchy.transform_to_go.get(&vid).copied().unwrap_or(vid);
                                go_to_glb_idx.get(&go).copied()
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();

            let mut joints = Vec::with_capacity(skin_data.joint_names.len());
            let mut ibms = Vec::with_capacity(skin_data.joint_names.len());
            let mut valid = true;

            for (cluster_idx, jname) in skin_data.joint_names.iter().enumerate() {
                let resolved_joint = find_joint_glb_node(
                    jname,
                    cluster_idx,
                    &unity_bones,
                    &glb_nodes,
                    &name_to_glb_idx,
                    skin_data.joint_names.len(),
                );
                if let Some(j_glb_idx) = resolved_joint {
                    joints.push(j_glb_idx);
                    ibms.push(skin_data.inverse_bind_matrices[cluster_idx]);
                } else {
                    valid = false;
                    warnings.push(format!(
                        "Could not map skin joint '{jname}' on mesh '{}'",
                        node.name
                    ));
                    break;
                }
            }

            if valid && !joints.is_empty() {
                let skeleton_root = yaml_get(&renderer_comp.data, "m_RootBone")
                    .and_then(|v| hierarchy.resolve_ref_to_go(v))
                    .and_then(|go| go_to_glb_idx.get(&go).copied())
                    .or_else(|| joints.first().copied());

                let skin_idx = glb_skins.len();
                glb_skins.push(GlbSkinInput {
                    name: format!("{}_Skin", node.name),
                    skeleton_root,
                    joints,
                    inverse_bind_matrices: ibms,
                });
                glb_nodes[glb_idx].skin = Some(skin_idx);
            }
        }

        let mesh_idx = glb_meshes.len();
        glb_meshes.push(GlbMeshAttachment {
            name: node.name.clone(),
            fbx_mesh,
            material_indices: mat_indices,
        });
        glb_nodes[glb_idx].mesh = Some(mesh_idx);
    }

    Ok((
        GlbExportInput {
            nodes: glb_nodes,
            root_nodes: root_indices,
            meshes: glb_meshes,
            skins: glb_skins,
            materials: glb_materials,
            max_texture_size,
        },
        material_defs,
    ))
}

fn collect_active_glb_nodes(
    hierarchy: &SceneHierarchy,
    go_id: VirtualId,
    parent_glb_idx: Option<usize>,
    go_to_glb_idx: &mut HashMap<VirtualId, usize>,
    glb_nodes: &mut Vec<GlbNodeInput>,
    node_order: &mut Vec<VirtualId>,
) -> Option<usize> {
    let node = hierarchy.nodes.get(&go_id)?;
    if !node.active {
        return None;
    }

    let idx = glb_nodes.len();
    let (translation, rotation, scale) =
        unity_lh_to_gltf_rh_transform(node.local_position, node.local_rotation, node.local_scale);

    go_to_glb_idx.insert(go_id, idx);
    node_order.push(go_id);
    glb_nodes.push(GlbNodeInput {
        name: node.name.clone(),
        parent: parent_glb_idx,
        children: Vec::new(),
        translation,
        rotation,
        scale,
        mesh: None,
        skin: None,
    });

    let children = node.children.clone();
    for child_id in children {
        collect_active_glb_nodes(
            hierarchy,
            child_id,
            Some(idx),
            go_to_glb_idx,
            glb_nodes,
            node_order,
        );
    }

    Some(idx)
}

fn select_fbx_mesh<'a>(fbx_scene: &'a FbxSceneData, go_name: &str) -> &'a fbx_loader::FbxMeshData {
    if fbx_scene.meshes.len() == 1 {
        return &fbx_scene.meshes[0];
    }
    if let Some(node) = fbx_scene.nodes.iter().find(|n| n.name == go_name) {
        if let Some(m_idx) = node.mesh_index {
            if let Some(m) = fbx_scene.meshes.get(m_idx) {
                return m;
            }
        }
    }
    if let Some(m) = fbx_scene.meshes.iter().find(|m| m.name == go_name) {
        return m;
    }
    &fbx_scene.meshes[0]
}

fn find_joint_glb_node(
    joint_name: &str,
    cluster_idx: usize,
    unity_bones: &[Option<usize>],
    glb_nodes: &[GlbNodeInput],
    name_to_glb_idx: &HashMap<String, usize>,
    total_clusters: usize,
) -> Option<usize> {
    // 1. Exact name match among m_Bones
    for idx in unity_bones.iter().flatten() {
        if glb_nodes.get(*idx).is_some_and(|n| n.name == joint_name) {
            return Some(*idx);
        }
    }
    // 2. Positional match in m_Bones (handles ModularAvatarMergeArmature when outfit bone was prefixed/suffixed)
    if unity_bones.len() == total_clusters {
        if let Some(Some(idx)) = unity_bones.get(cluster_idx) {
            return Some(*idx);
        }
    }
    // 3. Exact name match in avatar skeleton
    if let Some(&idx) = name_to_glb_idx.get(joint_name) {
        return Some(idx);
    }
    None
}

fn find_viseme_mesh_by_blendshapes<'a>(
    meshes: &'a [GlbMeshAttachment],
    viseme_shapes: &[String],
) -> Option<&'a GlbMeshAttachment> {
    let non_empty: Vec<&str> = viseme_shapes
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    if non_empty.is_empty() {
        return None;
    }
    meshes
        .iter()
        .filter_map(|m| {
            let hits = non_empty
                .iter()
                .filter(|shape| m.fbx_mesh.morph_target_names.iter().any(|mt| mt == *shape))
                .count();
            if hits > 0 {
                Some((hits, m))
            } else {
                None
            }
        })
        .max_by_key(|(hits, _)| *hits)
        .map(|(_, m)| m)
}
