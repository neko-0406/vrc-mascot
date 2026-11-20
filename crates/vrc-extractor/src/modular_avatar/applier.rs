use std::collections::{HashMap, HashSet};

use glam::{Mat4, Quat, Vec3};
use vrc_schema::{
    ExpressionMenuItem, ExpressionMenuItemType, ExpressionParamType, ExpressionParameterDef,
    ExpressionsDef,
};
use yaml_rust2::yaml::Hash;
use yaml_rust2::Yaml;

use crate::error::ExtractorError;
use crate::indexer::{AssetIndex, MaComponentKind, UnityRef};
use crate::yaml_parser::avatar::{map_vrc_control_type, parse_expressions_menu_asset};
use crate::yaml_parser::helpers::{
    yaml_as_bool, yaml_as_f32, yaml_as_i64, yaml_as_string, yaml_as_virtual_id, yaml_get,
};
use crate::yaml_parser::parse_animator_controller;
use crate::yaml_parser::prefab::{SceneHierarchy, VirtualComponent, VirtualId};

/// Summary of virtual Modular Avatar operations applied to an avatar.
#[derive(Debug, Clone, Default)]
pub struct MaApplyReport {
    pub merged_bones_count: usize,
    pub transplanted_bones_count: usize,
    pub reparented_proxies_count: usize,
    pub updated_mesh_settings_count: usize,
    pub added_menu_items_count: usize,
    pub added_parameters_count: usize,
    pub merged_animator_layers_count: usize,
}

/// Identifies a `VirtualComponent` as a `MaComponentKind` via script GUID, filename, or field signature.
pub fn identify_ma_component(
    comp: &VirtualComponent,
    index: &AssetIndex,
) -> Option<MaComponentKind> {
    if let Some(guid) = comp.script_guid.as_deref() {
        if let Some(kind) = index.ma_kind_by_guid(guid) {
            return Some(kind);
        }
        if let Some(path) = index.resolve_guid(guid) {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                let lower = stem.to_ascii_lowercase();
                if lower.contains("mergearmature") {
                    return Some(MaComponentKind::MergeArmature);
                } else if lower.contains("menuinstaller") {
                    return Some(MaComponentKind::MenuInstaller);
                } else if lower.contains("menuitem") {
                    return Some(MaComponentKind::MenuItem);
                } else if lower.contains("modularavatarparameters") {
                    return Some(MaComponentKind::Parameters);
                } else if lower.contains("meshsettings") {
                    return Some(MaComponentKind::MeshSettings);
                } else if lower.contains("boneproxy") {
                    return Some(MaComponentKind::BoneProxy);
                } else if lower.contains("mergeanimator") {
                    return Some(MaComponentKind::MergeAnimator);
                }
            }
        }
    }

    // Field-signature fallback
    if yaml_get(&comp.data, "mergeTarget").is_some() {
        return Some(MaComponentKind::MergeArmature);
    }
    if yaml_get(&comp.data, "installTargetMenu").is_some()
        || yaml_get(&comp.data, "menuToAppend").is_some()
    {
        return Some(MaComponentKind::MenuInstaller);
    }
    if yaml_get(&comp.data, "Control").is_some()
        || yaml_get(&comp.data, "PortableControl").is_some()
    {
        return Some(MaComponentKind::MenuItem);
    }
    if let Some(params) = yaml_get(&comp.data, "parameters").and_then(|v| v.as_vec()) {
        if params.iter().any(|p| yaml_get(p, "nameOrPrefix").is_some()) {
            return Some(MaComponentKind::Parameters);
        }
    }
    if yaml_get(&comp.data, "InheritProbeAnchor").is_some()
        || yaml_get(&comp.data, "inheritProbeAnchor").is_some()
        || yaml_get(&comp.data, "InheritBounds").is_some()
        || yaml_get(&comp.data, "inheritBounds").is_some()
    {
        return Some(MaComponentKind::MeshSettings);
    }
    if yaml_get(&comp.data, "boneReference").is_some()
        || (yaml_get(&comp.data, "target").is_some()
            && yaml_get(&comp.data, "attachmentMode").is_some())
    {
        return Some(MaComponentKind::BoneProxy);
    }
    if yaml_get(&comp.data, "animator").is_some()
        && (yaml_get(&comp.data, "layerType").is_some()
            || yaml_get(&comp.data, "pathMode").is_some()
            || yaml_get(&comp.data, "matchAvatarWriteDefaults").is_some())
    {
        return Some(MaComponentKind::MergeAnimator);
    }

    None
}

/// Applies all 6 Modular Avatar components virtually onto `hierarchy` and `expressions`.
pub fn apply_modular_avatar(
    hierarchy: &mut SceneHierarchy,
    avatar_root: VirtualId,
    humanoid_bones: &HashMap<String, String>,
    expressions: &mut ExpressionsDef,
    index: &AssetIndex,
    warnings: &mut Vec<String>,
) -> Result<MaApplyReport, ExtractorError> {
    let mut report = MaApplyReport::default();

    // 1. ModularAvatarBoneProxy (reparent props/accessories before armature merge or bone lookup)
    apply_bone_proxies(hierarchy, avatar_root, humanoid_bones, index, &mut report);

    // 2. ModularAvatarMergeArmature (merge outfit armatures & rebind SkinnedMeshRenderer.m_Bones)
    apply_merge_armatures(hierarchy, avatar_root, index, &mut report, warnings);

    // 3. ModularAvatarMeshSettings (propagate Anchor Override & Bounds)
    apply_mesh_settings(hierarchy, avatar_root, index, &mut report);

    // 4. ModularAvatarParameters (merge parameters with deduplication)
    apply_parameters(hierarchy, avatar_root, expressions, index, &mut report);

    // 5. ModularAvatarMenuInstaller & ModularAvatarMenuItem
    apply_menu_installers(
        hierarchy,
        avatar_root,
        expressions,
        index,
        &mut report,
        warnings,
    )?;

    // 6. ModularAvatarMergeAnimator (merge FX controller layers & parameters)
    apply_merge_animators(
        hierarchy,
        avatar_root,
        expressions,
        index,
        &mut report,
        warnings,
    )?;

    Ok(report)
}

fn apply_bone_proxies(
    hierarchy: &mut SceneHierarchy,
    avatar_root: VirtualId,
    humanoid_bones: &HashMap<String, String>,
    index: &AssetIndex,
    report: &mut MaApplyReport,
) {
    let subtree = hierarchy.collect_subtree(avatar_root);
    let mut proxies = Vec::new();
    for &go_id in &subtree {
        let Some(node) = hierarchy.nodes.get(&go_id) else {
            continue;
        };
        if !node.active {
            continue;
        }
        for comp in &node.components {
            if identify_ma_component(comp, index) == Some(MaComponentKind::BoneProxy) {
                proxies.push((go_id, comp.data.clone()));
            }
        }
    }

    for (go_id, data) in proxies {
        let target_go =
            resolve_ma_target_ref(&data["target"], hierarchy, avatar_root).or_else(|| {
                let bone_ref = yaml_get(&data, "boneReference")
                    .and_then(yaml_as_string)
                    .unwrap_or_default();
                let sub_path = yaml_get(&data, "subPath")
                    .and_then(yaml_as_string)
                    .unwrap_or_default();
                resolve_bone_reference(hierarchy, avatar_root, humanoid_bones, &bone_ref, &sub_path)
            });

        let Some(target_go) = target_go else {
            continue;
        };
        if target_go == go_id {
            continue;
        }

        let attachment_mode = yaml_get(&data, "attachmentMode")
            .and_then(yaml_as_i64)
            .unwrap_or(1);

        let old_world = compute_world_matrix(hierarchy, go_id);
        let new_parent_world = compute_world_matrix(hierarchy, target_go);

        hierarchy.reparent(go_id, target_go);

        match attachment_mode {
            // 0=Unset, 1=AsChildAtRoot -> align cleanly at target bone origin
            0 | 1 => {
                if let Some(node) = hierarchy.nodes.get_mut(&go_id) {
                    node.local_position = [0.0, 0.0, 0.0];
                    node.local_rotation = [0.0, 0.0, 0.0, 1.0];
                }
            }
            // 2=AsChildKeepWorld -> preserve world transform under new parent
            2 => {
                let local_mat = new_parent_world.inverse() * old_world;
                let (scale, rot, pos) = local_mat.to_scale_rotation_translation();
                if let Some(node) = hierarchy.nodes.get_mut(&go_id) {
                    node.local_position = pos.to_array();
                    node.local_rotation = rot.to_array();
                    node.local_scale = scale.to_array();
                }
            }
            // 3=AsChildKeepPosition -> preserve world position, reset rotation
            3 => {
                let (_, _, world_pos) = old_world.to_scale_rotation_translation();
                let local_pos = new_parent_world.inverse().transform_point3(world_pos);
                if let Some(node) = hierarchy.nodes.get_mut(&go_id) {
                    node.local_position = local_pos.to_array();
                    node.local_rotation = [0.0, 0.0, 0.0, 1.0];
                }
            }
            // 4=AsChildKeepRotation -> zero position, preserve world rotation
            4 => {
                let (_, world_rot, _) = old_world.to_scale_rotation_translation();
                let (_, parent_rot, _) = new_parent_world.to_scale_rotation_translation();
                let local_rot = parent_rot.inverse() * world_rot;
                if let Some(node) = hierarchy.nodes.get_mut(&go_id) {
                    node.local_position = [0.0, 0.0, 0.0];
                    node.local_rotation = local_rot.to_array();
                }
            }
            _ => {}
        }

        report.reparented_proxies_count += 1;
    }
}

fn apply_merge_armatures(
    hierarchy: &mut SceneHierarchy,
    avatar_root: VirtualId,
    index: &AssetIndex,
    report: &mut MaApplyReport,
    warnings: &mut Vec<String>,
) {
    let subtree = hierarchy.collect_subtree(avatar_root);
    let mut merge_specs = Vec::new();
    for &go_id in &subtree {
        let Some(node) = hierarchy.nodes.get(&go_id) else {
            continue;
        };
        if !node.active {
            continue;
        }
        for comp in &node.components {
            if identify_ma_component(comp, index) == Some(MaComponentKind::MergeArmature) {
                merge_specs.push((go_id, comp.data.clone()));
            }
        }
    }

    let mut global_bone_remap: HashMap<VirtualId, VirtualId> = HashMap::new();

    for (outfit_armature_go, data) in merge_specs {
        let Some(target_armature_go) =
            resolve_ma_target_ref(&data["mergeTarget"], hierarchy, avatar_root)
        else {
            warnings.push(format!(
                "ModularAvatarMergeArmature on '{}' could not resolve mergeTarget",
                hierarchy
                    .nodes
                    .get(&outfit_armature_go)
                    .map(|n| n.name.as_str())
                    .unwrap_or("?")
            ));
            continue;
        };

        if outfit_armature_go == target_armature_go {
            continue;
        }

        let prefix = yaml_get(&data, "prefix")
            .and_then(yaml_as_string)
            .unwrap_or_default();
        let suffix = yaml_get(&data, "suffix")
            .and_then(yaml_as_string)
            .unwrap_or_default();

        global_bone_remap.insert(outfit_armature_go, target_armature_go);

        let outfit_children = hierarchy
            .nodes
            .get(&outfit_armature_go)
            .map(|n| n.children.clone())
            .unwrap_or_default();

        for child_go in outfit_children {
            merge_bone_recursive(
                hierarchy,
                child_go,
                target_armature_go,
                &prefix,
                &suffix,
                &mut global_bone_remap,
                report,
            );
        }
    }

    if global_bone_remap.is_empty() {
        return;
    }

    // Rebind SkinnedMeshRenderer m_Bones & m_RootBone and PhysBone/Collider transform refs
    let transform_to_go = hierarchy.transform_to_go.clone();
    let all_nodes: Vec<VirtualId> = hierarchy.collect_subtree(avatar_root);
    for go_id in all_nodes {
        let Some(node) = hierarchy.nodes.get_mut(&go_id) else {
            continue;
        };
        for comp in &mut node.components {
            let is_smr = comp.class_id == 137 || comp.root_key == "SkinnedMeshRenderer";
            if is_smr {
                remap_smr_bones(&mut comp.data, &transform_to_go, &global_bone_remap);
            } else {
                // Also remap rootTransform / ignoreTransforms on PhysBone & PhysBoneCollider
                remap_transform_field(
                    &mut comp.data,
                    "rootTransform",
                    &transform_to_go,
                    &global_bone_remap,
                );
            }
        }
    }
}

fn merge_bone_recursive(
    hierarchy: &mut SceneHierarchy,
    outfit_bone_go: VirtualId,
    target_parent_go: VirtualId,
    prefix: &str,
    suffix: &str,
    bone_remap: &mut HashMap<VirtualId, VirtualId>,
    report: &mut MaApplyReport,
) {
    let Some(outfit_name) = hierarchy.nodes.get(&outfit_bone_go).map(|n| n.name.clone()) else {
        return;
    };
    let normalized = strip_affixes(&outfit_name, prefix, suffix);

    let matched_target = hierarchy.nodes.get(&target_parent_go).and_then(|parent| {
        parent.children.iter().copied().find(|&cid| {
            hierarchy
                .nodes
                .get(&cid)
                .is_some_and(|c| c.name == normalized)
        })
    });

    if let Some(target_bone_go) = matched_target {
        bone_remap.insert(outfit_bone_go, target_bone_go);
        report.merged_bones_count += 1;

        let children = hierarchy
            .nodes
            .get(&outfit_bone_go)
            .map(|n| n.children.clone())
            .unwrap_or_default();
        for child_go in children {
            merge_bone_recursive(
                hierarchy,
                child_go,
                target_bone_go,
                prefix,
                suffix,
                bone_remap,
                report,
            );
        }
    } else {
        // Unmatched bone (e.g. outfit hair, ribbon, tail): transplant entire subtree under target_parent_go
        hierarchy.reparent(outfit_bone_go, target_parent_go);
        report.transplanted_bones_count += 1;
    }
}

fn strip_affixes<'a>(name: &'a str, prefix: &str, suffix: &str) -> &'a str {
    let mut s = name;
    if !prefix.is_empty() {
        if let Some(rest) = s.strip_prefix(prefix) {
            s = rest;
        }
    }
    if !suffix.is_empty() {
        if let Some(rest) = s.strip_suffix(suffix) {
            s = rest;
        }
    }
    s
}

fn remap_smr_bones(
    smr_data: &mut Yaml,
    transform_to_go: &HashMap<VirtualId, VirtualId>,
    go_remap: &HashMap<VirtualId, VirtualId>,
) {
    remap_transform_field(smr_data, "m_RootBone", transform_to_go, go_remap);
    if let Yaml::Hash(h) = smr_data {
        if let Some(Yaml::Array(bones)) = h.get_mut(&Yaml::String("m_Bones".to_owned())) {
            for bone_ref in bones.iter_mut() {
                if let Some(old_tf_vid) = yaml_as_virtual_id(bone_ref) {
                    let old_go = transform_to_go
                        .get(&old_tf_vid)
                        .copied()
                        .unwrap_or(old_tf_vid);
                    if let Some(&new_go) = go_remap.get(&old_go) {
                        // Store remapped GameObject ID (or matching transform)
                        set_virtual_id_in_place(bone_ref, new_go);
                    }
                }
            }
        }
    }
}

fn remap_transform_field(
    data: &mut Yaml,
    field: &str,
    transform_to_go: &HashMap<VirtualId, VirtualId>,
    go_remap: &HashMap<VirtualId, VirtualId>,
) {
    if let Yaml::Hash(h) = data {
        if let Some(val) = h.get_mut(&Yaml::String(field.to_owned())) {
            if let Some(old_vid) = yaml_as_virtual_id(val) {
                let old_go = transform_to_go.get(&old_vid).copied().unwrap_or(old_vid);
                if let Some(&new_go) = go_remap.get(&old_go) {
                    set_virtual_id_in_place(val, new_go);
                }
            }
        }
    }
}

fn set_virtual_id_in_place(yaml: &mut Yaml, vid: VirtualId) {
    let mut h = Hash::new();
    h.insert(
        Yaml::String("__virtual_id".to_owned()),
        Yaml::Integer(vid as i64),
    );
    *yaml = Yaml::Hash(h);
}

#[derive(Debug, Clone, Default)]
struct InheritedMeshSettings {
    probe_anchor: Option<VirtualId>,
    root_bone: Option<VirtualId>,
    bounds: Option<Yaml>,
}

fn apply_mesh_settings(
    hierarchy: &mut SceneHierarchy,
    avatar_root: VirtualId,
    index: &AssetIndex,
    report: &mut MaApplyReport,
) {
    apply_mesh_settings_rec(
        hierarchy,
        avatar_root,
        avatar_root,
        InheritedMeshSettings::default(),
        index,
        report,
    );
}

fn apply_mesh_settings_rec(
    hierarchy: &mut SceneHierarchy,
    avatar_root: VirtualId,
    node_id: VirtualId,
    mut current: InheritedMeshSettings,
    index: &AssetIndex,
    report: &mut MaApplyReport,
) {
    let Some(node_snapshot) = hierarchy.nodes.get(&node_id).cloned() else {
        return;
    };

    for comp in &node_snapshot.components {
        if identify_ma_component(comp, index) != Some(MaComponentKind::MeshSettings) {
            continue;
        }
        // InheritMode: 0=Inherit, 1=Set, 2=DontSet, 3=SetOrInherit
        let probe_mode = yaml_get(&comp.data, "InheritProbeAnchor")
            .or_else(|| yaml_get(&comp.data, "inheritProbeAnchor"))
            .and_then(yaml_as_i64)
            .unwrap_or(1);
        if matches!(probe_mode, 1 | 3) {
            if let Some(anchor_go) = yaml_get(&comp.data, "ProbeAnchor")
                .or_else(|| yaml_get(&comp.data, "probeAnchor"))
                .and_then(|v| resolve_ma_target_ref(v, hierarchy, avatar_root))
            {
                current.probe_anchor = Some(anchor_go);
            }
        } else if probe_mode == 2 {
            current.probe_anchor = None;
        }

        let bounds_mode = yaml_get(&comp.data, "InheritBounds")
            .or_else(|| yaml_get(&comp.data, "inheritBounds"))
            .and_then(yaml_as_i64)
            .unwrap_or(1);
        if matches!(bounds_mode, 1 | 3) {
            if let Some(rb) = yaml_get(&comp.data, "RootBone")
                .or_else(|| yaml_get(&comp.data, "rootBone"))
                .and_then(|v| resolve_ma_target_ref(v, hierarchy, avatar_root))
            {
                current.root_bone = Some(rb);
            }
            if let Some(b) =
                yaml_get(&comp.data, "Bounds").or_else(|| yaml_get(&comp.data, "bounds"))
            {
                if !matches!(b, Yaml::Null) {
                    current.bounds = Some(b.clone());
                }
            }
        } else if bounds_mode == 2 {
            current.root_bone = None;
            current.bounds = None;
        }
    }

    if current.probe_anchor.is_some() || current.root_bone.is_some() || current.bounds.is_some() {
        if let Some(node) = hierarchy.nodes.get_mut(&node_id) {
            for comp in &mut node.components {
                let is_renderer = comp.class_id == 137
                    || comp.class_id == 23
                    || comp.root_key == "SkinnedMeshRenderer"
                    || comp.root_key == "MeshRenderer";
                if !is_renderer {
                    continue;
                }
                if let Yaml::Hash(h) = &mut comp.data {
                    if let Some(anchor_go) = current.probe_anchor {
                        let mut ref_h = Hash::new();
                        ref_h.insert(
                            Yaml::String("__virtual_id".to_owned()),
                            Yaml::Integer(anchor_go as i64),
                        );
                        h.insert(Yaml::String("m_ProbeAnchor".to_owned()), Yaml::Hash(ref_h));
                    }
                    if let Some(rb_go) = current.root_bone {
                        let mut ref_h = Hash::new();
                        ref_h.insert(
                            Yaml::String("__virtual_id".to_owned()),
                            Yaml::Integer(rb_go as i64),
                        );
                        h.insert(Yaml::String("m_RootBone".to_owned()), Yaml::Hash(ref_h));
                    }
                    if let Some(ref b) = current.bounds {
                        h.insert(Yaml::String("m_AABB".to_owned()), b.clone());
                    }
                    report.updated_mesh_settings_count += 1;
                }
            }
        }
    }

    for child_id in node_snapshot.children {
        apply_mesh_settings_rec(
            hierarchy,
            avatar_root,
            child_id,
            current.clone(),
            index,
            report,
        );
    }
}

fn apply_parameters(
    hierarchy: &SceneHierarchy,
    avatar_root: VirtualId,
    expressions: &mut ExpressionsDef,
    index: &AssetIndex,
    report: &mut MaApplyReport,
) {
    let mut existing: HashSet<String> = expressions
        .parameters
        .iter()
        .map(|p| p.name.clone())
        .collect();

    for go_id in hierarchy.collect_subtree(avatar_root) {
        let Some(node) = hierarchy.nodes.get(&go_id) else {
            continue;
        };
        if !node.active {
            continue;
        }
        for comp in &node.components {
            if identify_ma_component(comp, index) != Some(MaComponentKind::Parameters) {
                continue;
            }
            let Some(params) = yaml_get(&comp.data, "parameters").and_then(|v| v.as_vec()) else {
                continue;
            };
            for p in params {
                let raw_name = yaml_get(p, "nameOrPrefix")
                    .and_then(yaml_as_string)
                    .unwrap_or_default();
                let remap_to = yaml_get(p, "remapTo")
                    .and_then(yaml_as_string)
                    .unwrap_or_default();
                let name = if remap_to.trim().is_empty() {
                    raw_name
                } else {
                    remap_to
                };
                if name.trim().is_empty() {
                    continue;
                }

                // MA ParameterSyncType: 0=NotSynced, 1=Int, 2=Float, 3=Bool
                let sync_type = yaml_get(p, "syncType").and_then(yaml_as_i64).unwrap_or(3);
                let param_type = match sync_type {
                    1 => ExpressionParamType::Int,
                    2 => ExpressionParamType::Float,
                    _ => ExpressionParamType::Bool,
                };
                let default = yaml_get(p, "defaultValue")
                    .and_then(yaml_as_f32)
                    .unwrap_or(0.0);
                let saved = yaml_get(p, "saved").and_then(yaml_as_bool).unwrap_or(true);
                let _internal = yaml_get(p, "internalParameter")
                    .and_then(yaml_as_bool)
                    .unwrap_or(false);

                if existing.insert(name.clone()) {
                    expressions.parameters.push(ExpressionParameterDef {
                        name,
                        param_type,
                        default,
                        saved,
                    });
                    report.added_parameters_count += 1;
                }
            }
        }
    }
}

fn apply_menu_installers(
    hierarchy: &SceneHierarchy,
    avatar_root: VirtualId,
    expressions: &mut ExpressionsDef,
    index: &AssetIndex,
    report: &mut MaApplyReport,
    warnings: &mut Vec<String>,
) -> Result<(), ExtractorError> {
    for go_id in hierarchy.collect_subtree(avatar_root) {
        let Some(node) = hierarchy.nodes.get(&go_id) else {
            continue;
        };
        if !node.active {
            continue;
        }
        for comp in &node.components {
            if identify_ma_component(comp, index) != Some(MaComponentKind::MenuInstaller) {
                continue;
            }

            let mut items = Vec::new();
            if let Some(menu_guid) = yaml_get(&comp.data, "menuToAppend")
                .and_then(UnityRef::from_yaml)
                .and_then(|r| r.guid)
            {
                if let Some(menu_path) = index.resolve_guid(&menu_guid) {
                    items.extend(parse_expressions_menu_asset(menu_path, index)?);
                } else {
                    warnings.push(format!(
                        "ModularAvatarMenuInstaller menuToAppend GUID '{menu_guid}' not found"
                    ));
                }
            }

            if items.is_empty() {
                // Fall back to ModularAvatarMenuItem on the same GameObject
                items.extend(collect_menu_items_on_node(hierarchy, go_id, index));
            }

            for item in items {
                report.added_menu_items_count += count_menu_items(&item);
                expressions.menu.push(item);
            }
        }
    }
    Ok(())
}

fn collect_menu_items_on_node(
    hierarchy: &SceneHierarchy,
    go_id: VirtualId,
    index: &AssetIndex,
) -> Vec<ExpressionMenuItem> {
    let Some(node) = hierarchy.nodes.get(&go_id) else {
        return Vec::new();
    };
    let mut items = Vec::new();
    for comp in &node.components {
        if identify_ma_component(comp, index) != Some(MaComponentKind::MenuItem) {
            continue;
        }
        let ctrl = yaml_get(&comp.data, "PortableControl")
            .or_else(|| yaml_get(&comp.data, "Control"))
            .unwrap_or(&comp.data);

        let name = yaml_get(ctrl, "name")
            .or_else(|| yaml_get(&comp.data, "label"))
            .and_then(yaml_as_string)
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| node.name.clone());

        let ctrl_type = yaml_get(ctrl, "type").and_then(yaml_as_i64).unwrap_or(102);
        let item_type = map_vrc_control_type(ctrl_type);
        let parameter = yaml_get(&ctrl["parameter"], "name")
            .or_else(|| yaml_get(&comp.data, "parameterName"))
            .and_then(yaml_as_string)
            .filter(|s| !s.trim().is_empty());
        let value = yaml_get(ctrl, "value").and_then(yaml_as_f32).unwrap_or(1.0);

        let sub_menu = if item_type == ExpressionMenuItemType::SubMenu {
            let mut sub = Vec::new();
            for &child_id in &node.children {
                sub.extend(collect_menu_items_on_node(hierarchy, child_id, index));
            }
            sub
        } else {
            Vec::new()
        };

        items.push(ExpressionMenuItem {
            name,
            item_type,
            parameter,
            value,
            sub_menu,
        });
    }
    items
}

fn count_menu_items(item: &ExpressionMenuItem) -> usize {
    1 + item.sub_menu.iter().map(count_menu_items).sum::<usize>()
}

fn apply_merge_animators(
    hierarchy: &SceneHierarchy,
    avatar_root: VirtualId,
    expressions: &mut ExpressionsDef,
    index: &AssetIndex,
    report: &mut MaApplyReport,
    warnings: &mut Vec<String>,
) -> Result<(), ExtractorError> {
    let mut existing_params: HashSet<String> = expressions
        .parameters
        .iter()
        .map(|p| p.name.clone())
        .collect();

    let mut pending = Vec::new();
    for go_id in hierarchy.collect_subtree(avatar_root) {
        let Some(node) = hierarchy.nodes.get(&go_id) else {
            continue;
        };
        if !node.active {
            continue;
        }
        for comp in &node.components {
            if identify_ma_component(comp, index) != Some(MaComponentKind::MergeAnimator) {
                continue;
            }
            let layer_type = yaml_get(&comp.data, "layerType")
                .and_then(yaml_as_i64)
                .unwrap_or(5);
            // Only merge FX layers (type 5) into fx_layers
            if layer_type != 5 {
                continue;
            }
            let priority = yaml_get(&comp.data, "layerPriority")
                .and_then(yaml_as_i64)
                .unwrap_or(0);
            if let Some(ctrl_guid) = yaml_get(&comp.data, "animator")
                .and_then(UnityRef::from_yaml)
                .and_then(|r| r.guid)
            {
                pending.push((priority, ctrl_guid));
            }
        }
    }

    pending.sort_by_key(|(prio, _)| *prio);

    for (_prio, ctrl_guid) in pending {
        let Some(ctrl_path) = index.resolve_guid(&ctrl_guid) else {
            warnings.push(format!(
                "ModularAvatarMergeAnimator animator GUID '{ctrl_guid}' not found"
            ));
            continue;
        };
        let parsed = parse_animator_controller(ctrl_path, index, warnings)?;
        for p in parsed.parameters {
            if existing_params.insert(p.name.clone()) {
                expressions.parameters.push(p);
                report.added_parameters_count += 1;
            }
        }
        for layer in parsed.layers {
            expressions.fx_layers.push(layer);
            report.merged_animator_layers_count += 1;
        }
    }

    Ok(())
}

/// Resolves a Modular Avatar `AvatarObjectReference` (`{ targetObject }` / `{ referencePath }`) or direct `__virtual_id`.
fn resolve_ma_target_ref(
    yaml: &Yaml,
    hierarchy: &SceneHierarchy,
    avatar_root: VirtualId,
) -> Option<VirtualId> {
    if let Some(go) = hierarchy.resolve_ref_to_go(yaml) {
        return Some(go);
    }
    if let Some(go) = yaml_get(yaml, "targetObject").and_then(|v| hierarchy.resolve_ref_to_go(v)) {
        return Some(go);
    }
    if let Some(ref_path) = yaml_get(yaml, "referencePath").and_then(yaml_as_string) {
        let trimmed = ref_path.trim().trim_matches('/');
        if trimmed.is_empty() {
            return Some(avatar_root);
        }
        if let Some(found) = find_node_by_rel_path(hierarchy, avatar_root, trimmed) {
            return Some(found);
        }
        // Fallback: search by leaf name inside avatar subtree
        let leaf = trimmed.rsplit('/').next().unwrap_or(trimmed);
        return hierarchy
            .collect_subtree(avatar_root)
            .into_iter()
            .find(|&id| hierarchy.nodes.get(&id).is_some_and(|n| n.name == leaf));
    }
    None
}

fn find_node_by_rel_path(
    hierarchy: &SceneHierarchy,
    root: VirtualId,
    rel_path: &str,
) -> Option<VirtualId> {
    let mut cur = root;
    for seg in rel_path.split('/').filter(|s| !s.is_empty()) {
        let node = hierarchy.nodes.get(&cur)?;
        cur = node
            .children
            .iter()
            .copied()
            .find(|&cid| hierarchy.nodes.get(&cid).is_some_and(|c| c.name == seg))?;
    }
    Some(cur)
}

fn resolve_bone_reference(
    hierarchy: &SceneHierarchy,
    avatar_root: VirtualId,
    humanoid_bones: &HashMap<String, String>,
    bone_ref: &str,
    sub_path: &str,
) -> Option<VirtualId> {
    let trimmed = bone_ref.trim();
    if trimmed.is_empty() {
        return None;
    }
    let base = if let Some(rel) = humanoid_bones.get(trimmed) {
        find_node_by_rel_path(hierarchy, avatar_root, rel).or_else(|| {
            let leaf = rel.rsplit('/').next().unwrap_or(rel);
            hierarchy
                .collect_subtree(avatar_root)
                .into_iter()
                .find(|&id| hierarchy.nodes.get(&id).is_some_and(|n| n.name == leaf))
        })
    } else {
        find_node_by_rel_path(hierarchy, avatar_root, trimmed).or_else(|| {
            hierarchy
                .collect_subtree(avatar_root)
                .into_iter()
                .find(|&id| hierarchy.nodes.get(&id).is_some_and(|n| n.name == trimmed))
        })
    }?;

    if sub_path.trim().is_empty() {
        Some(base)
    } else {
        find_node_by_rel_path(hierarchy, base, sub_path.trim())
    }
}

fn compute_world_matrix(hierarchy: &SceneHierarchy, node_id: VirtualId) -> Mat4 {
    let mut chain = Vec::new();
    let mut cur = Some(node_id);
    while let Some(id) = cur {
        if let Some(n) = hierarchy.nodes.get(&id) {
            chain.push(id);
            cur = n.parent;
        } else {
            break;
        }
    }
    let mut world = Mat4::IDENTITY;
    for id in chain.into_iter().rev() {
        let n = &hierarchy.nodes[&id];
        let t = Vec3::from_array(n.local_position);
        let r = Quat::from_array(n.local_rotation).normalize();
        let s = Vec3::from_array(n.local_scale);
        world *= Mat4::from_scale_rotation_translation(s, r, t);
    }
    world
}
