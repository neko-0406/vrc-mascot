use std::collections::{HashMap, HashSet};
use std::path::Path;

use vrc_schema::{
    BlendShapeBinding, ColliderDef, ColliderShape, ExpressionMenuItem, ExpressionMenuItemType,
    ExpressionParamType, ExpressionParameterDef, PhysBoneDef, PhysBoneLimitType, VisemeMap,
};
use yaml_rust2::Yaml;

use crate::error::ExtractorError;
use crate::indexer::{AssetIndex, UnityRef, UnityYamlFile, VrcComponentKind};
use crate::yaml_parser::helpers::{
    yaml_as_bool, yaml_as_f32, yaml_as_i64, yaml_as_quat, yaml_as_string, yaml_as_vec3, yaml_get,
};
use crate::yaml_parser::prefab::{SceneHierarchy, VirtualComponent, VirtualId};

/// Extracted `VRCAvatarDescriptor` configuration.
#[derive(Debug, Clone, Default)]
pub struct AvatarDescriptorData {
    pub fx_controller_guid: Option<String>,
    pub expressions_menu_guid: Option<String>,
    pub expression_parameters_guid: Option<String>,
    pub viseme_map: VisemeMap,
    pub left_eye_node: Option<String>,
    pub right_eye_node: Option<String>,
}

/// Identifies a `VirtualComponent` as a `VrcComponentKind` via GUID, filename, or field signature.
pub fn identify_vrc_component(
    comp: &VirtualComponent,
    index: &AssetIndex,
) -> Option<VrcComponentKind> {
    if let Some(guid) = comp.script_guid.as_deref() {
        if let Some(kind) = index.vrc_kind_by_guid(guid) {
            return Some(kind);
        }
        if let Some(path) = index.resolve_guid(guid) {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                let lower = stem.to_ascii_lowercase();
                if lower.contains("avatardescriptor") {
                    return Some(VrcComponentKind::AvatarDescriptor);
                } else if lower == "vrcphysbone" {
                    return Some(VrcComponentKind::PhysBone);
                } else if lower == "vrcphysbonecollider" {
                    return Some(VrcComponentKind::PhysBoneCollider);
                }
            }
        }
    }

    // Field-signature fallback (essential for precompiled VRCSDK DLL MonoBehaviours)
    if yaml_get(&comp.data, "baseAnimationLayers").is_some()
        && (yaml_get(&comp.data, "VisemeSkinnedMesh").is_some()
            || yaml_get(&comp.data, "expressionParameters").is_some()
            || yaml_get(&comp.data, "expressionsMenu").is_some()
            || yaml_get(&comp.data, "lipSync").is_some())
    {
        return Some(VrcComponentKind::AvatarDescriptor);
    }
    if yaml_get(&comp.data, "pull").is_some()
        && yaml_get(&comp.data, "spring").is_some()
        && yaml_get(&comp.data, "stiffness").is_some()
        && (yaml_get(&comp.data, "rootTransform").is_some()
            || yaml_get(&comp.data, "immovable").is_some()
            || yaml_get(&comp.data, "limitType").is_some())
    {
        return Some(VrcComponentKind::PhysBone);
    }
    if yaml_get(&comp.data, "shapeType").is_some()
        && yaml_get(&comp.data, "radius").is_some()
        && (yaml_get(&comp.data, "insideBounds").is_some()
            || yaml_get(&comp.data, "rootTransform").is_some())
    {
        return Some(VrcComponentKind::PhysBoneCollider);
    }

    None
}

/// Finds all GameObject IDs in `hierarchy` that carry a `VRCAvatarDescriptor` component.
pub fn find_avatar_candidates(hierarchy: &SceneHierarchy, index: &AssetIndex) -> Vec<VirtualId> {
    let mut ids: Vec<VirtualId> = hierarchy
        .nodes
        .values()
        .filter(|node| {
            node.components.iter().any(|c| {
                identify_vrc_component(c, index) == Some(VrcComponentKind::AvatarDescriptor)
            })
        })
        .map(|node| node.id)
        .collect();
    ids.sort_unstable();
    ids
}

/// Extracts `VRCAvatarDescriptor` settings from the avatar root node (if present).
pub fn extract_avatar_descriptor_data(
    hierarchy: &SceneHierarchy,
    avatar_root: VirtualId,
    index: &AssetIndex,
) -> AvatarDescriptorData {
    let mut out = AvatarDescriptorData::default();
    let Some(root_node) = hierarchy.nodes.get(&avatar_root) else {
        return out;
    };
    let Some(desc) = root_node
        .components
        .iter()
        .find(|c| identify_vrc_component(c, index) == Some(VrcComponentKind::AvatarDescriptor))
    else {
        return out;
    };

    // FX layer (type == 5 in VRC_AvatarDescriptor.AnimLayerType)
    if let Some(layers) = yaml_get(&desc.data, "baseAnimationLayers").and_then(|v| v.as_vec()) {
        for layer in layers {
            let layer_type = yaml_get(layer, "type").and_then(yaml_as_i64).unwrap_or(-1);
            if layer_type == 5 {
                if let Some(ctrl_ref) =
                    yaml_get(layer, "animatorController").and_then(UnityRef::from_yaml)
                {
                    out.fx_controller_guid = ctrl_ref.guid;
                }
            }
        }
    }

    out.expressions_menu_guid = yaml_get(&desc.data, "expressionsMenu")
        .and_then(UnityRef::from_yaml)
        .and_then(|r| r.guid);
    out.expression_parameters_guid = yaml_get(&desc.data, "expressionParameters")
        .and_then(UnityRef::from_yaml)
        .and_then(|r| r.guid);

    // Visemes
    let viseme_mesh_name = yaml_get(&desc.data, "VisemeSkinnedMesh")
        .and_then(|v| hierarchy.resolve_component_ref(v))
        .and_then(|(go_id, _)| hierarchy.nodes.get(&go_id))
        .map(|n| n.name.clone());

    if let (Some(mesh_name), Some(shapes)) = (
        viseme_mesh_name,
        yaml_get(&desc.data, "VisemeBlendShapes").and_then(|v| v.as_vec()),
    ) {
        let shape_names: Vec<String> = shapes
            .iter()
            .map(|s| yaml_as_string(s).unwrap_or_default())
            .collect();
        let make_binding = |idx: usize, fallback_suffix: &str| -> Option<BlendShapeBinding> {
            let name = shape_names
                .get(idx)
                .filter(|s| !s.trim().is_empty())
                .cloned()
                .or_else(|| {
                    shape_names
                        .iter()
                        .find(|s| s.to_ascii_lowercase().ends_with(fallback_suffix))
                        .cloned()
                })?;
            Some(BlendShapeBinding {
                mesh: mesh_name.clone(),
                blendshape: name,
            })
        };

        out.viseme_map = VisemeMap {
            sil: make_binding(0, "sil"),
            aa: make_binding(10, "_aa"),
            ee: make_binding(11, "_ee"),
            ih: make_binding(12, "_ih"),
            oh: make_binding(13, "_oh"),
            ou: make_binding(14, "_ou"),
        };
    }

    // Eye look bones (optional fallback for humanoid_bones)
    let eye_settings = &desc.data["customEyeLookSettings"];
    if let Some(left_go) = yaml_get(eye_settings, "leftEye")
        .and_then(|v| hierarchy.resolve_ref_to_go(v))
        .and_then(|gid| hierarchy.relative_path(avatar_root, gid))
    {
        out.left_eye_node = Some(left_go);
    }
    if let Some(right_go) = yaml_get(eye_settings, "rightEye")
        .and_then(|v| hierarchy.resolve_ref_to_go(v))
        .and_then(|gid| hierarchy.relative_path(avatar_root, gid))
    {
        out.right_eye_node = Some(right_go);
    }

    out
}

/// Parses a `VRCExpressionParameters` `.asset` file.
pub fn parse_expression_parameters_asset(
    asset_path: &Path,
) -> Result<Vec<ExpressionParameterDef>, ExtractorError> {
    let yaml_file = UnityYamlFile::load(asset_path)?;
    let Some(doc) = yaml_file.documents.first() else {
        return Ok(Vec::new());
    };

    let mut out = Vec::new();
    if let Some(arr) = yaml_get(&doc.data, "parameters").and_then(|v| v.as_vec()) {
        for item in arr {
            let name = yaml_get(item, "name")
                .and_then(yaml_as_string)
                .unwrap_or_default();
            if name.trim().is_empty() {
                continue;
            }
            let value_type = yaml_get(item, "valueType")
                .and_then(yaml_as_i64)
                .unwrap_or(2);
            let param_type = match value_type {
                0 => ExpressionParamType::Int,
                1 => ExpressionParamType::Float,
                _ => ExpressionParamType::Bool,
            };
            let default = yaml_get(item, "defaultValue")
                .and_then(yaml_as_f32)
                .unwrap_or(0.0);
            let saved = yaml_get(item, "saved")
                .and_then(yaml_as_bool)
                .unwrap_or(true);
            out.push(ExpressionParameterDef {
                name,
                param_type,
                default,
                saved,
            });
        }
    }
    Ok(out)
}

/// Parses a `VRCExpressionsMenu` `.asset` file recursively (handling `SubMenu` references).
pub fn parse_expressions_menu_asset(
    asset_path: &Path,
    index: &AssetIndex,
) -> Result<Vec<ExpressionMenuItem>, ExtractorError> {
    let mut visited = HashSet::new();
    parse_expressions_menu_recursive(asset_path, index, &mut visited)
}

fn parse_expressions_menu_recursive(
    asset_path: &Path,
    index: &AssetIndex,
    visited: &mut HashSet<String>,
) -> Result<Vec<ExpressionMenuItem>, ExtractorError> {
    let key = asset_path.to_string_lossy().to_ascii_lowercase();
    if !visited.insert(key.clone()) {
        return Ok(Vec::new());
    }

    let yaml_file = UnityYamlFile::load(asset_path)?;
    let Some(doc) = yaml_file.documents.first() else {
        visited.remove(&key);
        return Ok(Vec::new());
    };

    let mut items = Vec::new();
    if let Some(controls) = yaml_get(&doc.data, "controls").and_then(|v| v.as_vec()) {
        for ctrl in controls {
            let name = yaml_get(ctrl, "name")
                .and_then(yaml_as_string)
                .unwrap_or_default();
            if name.trim().is_empty() {
                continue;
            }
            // VRCExpressionsMenu.Control.ControlType:
            // 101 = Button, 102 = Toggle, 103 = SubMenu, 201 = TwoAxisPuppet, 202 = FourAxisPuppet, 203 = RadialPuppet
            // (Also handle 0..5 enum variants used by some tools/MA)
            let ctrl_type = yaml_get(ctrl, "type").and_then(yaml_as_i64).unwrap_or(102);
            let item_type = map_vrc_control_type(ctrl_type);

            let mut parameter = yaml_get(&ctrl["parameter"], "name")
                .and_then(yaml_as_string)
                .filter(|s| !s.is_empty());
            if parameter.is_none() {
                if let Some(sub_params) = yaml_get(ctrl, "subParameters").and_then(|v| v.as_vec()) {
                    parameter = sub_params
                        .first()
                        .and_then(|sp| yaml_get(sp, "name"))
                        .and_then(yaml_as_string)
                        .filter(|s| !s.is_empty());
                }
            }

            let value = yaml_get(ctrl, "value").and_then(yaml_as_f32).unwrap_or(1.0);
            let sub_menu = if item_type == ExpressionMenuItemType::SubMenu {
                if let Some(sub_guid) = yaml_get(ctrl, "subMenu")
                    .and_then(UnityRef::from_yaml)
                    .and_then(|r| r.guid)
                {
                    if let Some(sub_path) = index.resolve_guid(&sub_guid) {
                        parse_expressions_menu_recursive(sub_path, index, visited)?
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                }
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
    }

    visited.remove(&key);
    Ok(items)
}

pub(crate) fn map_vrc_control_type(code: i64) -> ExpressionMenuItemType {
    match code {
        101 | 1 => ExpressionMenuItemType::Button,
        102 | 2 => ExpressionMenuItemType::Toggle,
        103 | 3 => ExpressionMenuItemType::SubMenu,
        203 | 6 => ExpressionMenuItemType::RadialPuppet,
        _ => ExpressionMenuItemType::Toggle,
    }
}

/// Extracts all `VRCPhysBone` and `VRCPhysBoneCollider` definitions in the avatar subtree.
pub fn extract_physbones_and_colliders(
    hierarchy: &SceneHierarchy,
    avatar_root: VirtualId,
    index: &AssetIndex,
) -> (Vec<PhysBoneDef>, Vec<ColliderDef>) {
    let subtree = hierarchy.collect_subtree(avatar_root);
    let mut colliders = Vec::new();
    let mut collider_vid_to_id: HashMap<VirtualId, String> = HashMap::new();

    // Pass 1: Extract colliders and assign deterministic IDs
    for &go_id in &subtree {
        let Some(node) = hierarchy.nodes.get(&go_id) else {
            continue;
        };
        if !node.active {
            continue;
        }
        for comp in &node.components {
            if identify_vrc_component(comp, index) != Some(VrcComponentKind::PhysBoneCollider) {
                continue;
            }
            let parent_go = yaml_get(&comp.data, "rootTransform")
                .and_then(|v| hierarchy.resolve_ref_to_go(v))
                .unwrap_or(go_id);
            let Some(parent_node) = hierarchy.relative_path(avatar_root, parent_go) else {
                continue;
            };
            let parent_node = if parent_node.is_empty() {
                node.name.clone()
            } else {
                parent_node
            };

            let col_id = format!("col_{}_{}", sanitize_id(&node.name), colliders.len());
            let shape = match yaml_get(&comp.data, "shapeType")
                .and_then(yaml_as_i64)
                .unwrap_or(0)
            {
                1 => ColliderShape::Capsule,
                2 => ColliderShape::Plane,
                _ => ColliderShape::Sphere,
            };
            let radius = yaml_get(&comp.data, "radius")
                .and_then(yaml_as_f32)
                .unwrap_or(0.1)
                .max(0.0);
            let height = yaml_get(&comp.data, "height")
                .and_then(yaml_as_f32)
                .unwrap_or(0.0)
                .max(0.0);
            let position = yaml_get(&comp.data, "position")
                .and_then(yaml_as_vec3)
                .unwrap_or([0.0, 0.0, 0.0]);
            let rotation = yaml_get(&comp.data, "rotation")
                .and_then(yaml_as_quat)
                .unwrap_or([0.0, 0.0, 0.0, 1.0]);
            let inside_bounds = yaml_get(&comp.data, "insideBounds")
                .and_then(yaml_as_bool)
                .unwrap_or(false);

            collider_vid_to_id.insert(comp.object_id, col_id.clone());
            colliders.push(ColliderDef {
                id: col_id,
                parent_node,
                shape,
                radius,
                height,
                position,
                rotation,
                inside_bounds,
            });
        }
    }

    // Pass 2: Extract PhysBones and link collider IDs
    let mut phys_bones = Vec::new();
    for &go_id in &subtree {
        let Some(node) = hierarchy.nodes.get(&go_id) else {
            continue;
        };
        if !node.active {
            continue;
        }
        for comp in &node.components {
            if identify_vrc_component(comp, index) != Some(VrcComponentKind::PhysBone) {
                continue;
            }
            let root_go = yaml_get(&comp.data, "rootTransform")
                .and_then(|v| hierarchy.resolve_ref_to_go(v))
                .unwrap_or(go_id);
            let Some(root_node) = hierarchy.relative_path(avatar_root, root_go) else {
                continue;
            };
            if root_node.is_empty() {
                continue;
            }

            let mut ignore_nodes = Vec::new();
            if let Some(ignores) = yaml_get(&comp.data, "ignoreTransforms").and_then(|v| v.as_vec())
            {
                for ig in ignores {
                    if let Some(ig_go) = hierarchy.resolve_ref_to_go(ig) {
                        if let Some(rel) = hierarchy.relative_path(avatar_root, ig_go) {
                            if !rel.is_empty() {
                                ignore_nodes.push(rel);
                            }
                        }
                    }
                }
            }

            let mut collider_refs = Vec::new();
            if let Some(cols) = yaml_get(&comp.data, "colliders").and_then(|v| v.as_vec()) {
                for col_yaml in cols {
                    if let Some(cid) = resolve_collider_id(col_yaml, &collider_vid_to_id) {
                        if !collider_refs.contains(&cid) {
                            collider_refs.push(cid);
                        }
                    }
                }
            }

            let limit_type = match yaml_get(&comp.data, "limitType")
                .and_then(yaml_as_i64)
                .unwrap_or(0)
            {
                1 => PhysBoneLimitType::Angle,
                2 => PhysBoneLimitType::Hinge,
                3 => PhysBoneLimitType::Polar,
                _ => PhysBoneLimitType::None,
            };

            phys_bones.push(PhysBoneDef {
                root_node,
                ignore_nodes,
                pull: yaml_get(&comp.data, "pull")
                    .and_then(yaml_as_f32)
                    .unwrap_or(0.2),
                spring: yaml_get(&comp.data, "spring")
                    .and_then(yaml_as_f32)
                    .unwrap_or(0.2),
                stiffness: yaml_get(&comp.data, "stiffness")
                    .and_then(yaml_as_f32)
                    .unwrap_or(0.2),
                immovable: yaml_get(&comp.data, "immovable")
                    .and_then(yaml_as_f32)
                    .unwrap_or(0.0),
                gravity: yaml_get(&comp.data, "gravity")
                    .and_then(yaml_as_f32)
                    .unwrap_or(0.0),
                gravity_falloff: yaml_get(&comp.data, "gravityFalloff")
                    .and_then(yaml_as_f32)
                    .unwrap_or(0.0),
                radius: yaml_get(&comp.data, "radius")
                    .and_then(yaml_as_f32)
                    .unwrap_or(0.02)
                    .max(0.0),
                limit_type,
                max_angle_x: yaml_get(&comp.data, "maxAngleX")
                    .and_then(yaml_as_f32)
                    .unwrap_or(0.0),
                max_angle_z: yaml_get(&comp.data, "maxAngleZ")
                    .and_then(yaml_as_f32)
                    .unwrap_or(0.0),
                collider_refs,
                allow_grabbing: yaml_get(&comp.data, "allowGrabbing")
                    .and_then(yaml_as_bool)
                    .unwrap_or(true),
                allow_posing: yaml_get(&comp.data, "allowPosing")
                    .and_then(yaml_as_bool)
                    .unwrap_or(true),
            });
        }
    }

    (phys_bones, colliders)
}

fn resolve_collider_id(
    yaml: &Yaml,
    collider_vid_to_id: &HashMap<VirtualId, String>,
) -> Option<String> {
    crate::yaml_parser::helpers::yaml_as_virtual_id(yaml)
        .and_then(|vid| collider_vid_to_id.get(&vid).cloned())
}

fn sanitize_id(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    if cleaned.is_empty() {
        "node".to_owned()
    } else {
        cleaned
    }
}
