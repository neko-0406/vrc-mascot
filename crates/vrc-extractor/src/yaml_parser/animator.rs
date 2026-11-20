use std::collections::HashSet;
use std::path::Path;

use vrc_schema::{
    AnimationCurveDef, CurveProperty, ExpressionParamType, ExpressionParameterDef, FxConditionDef,
    FxConditionMode, FxLayerDef, FxStateDef, FxTransitionDef,
};
use yaml_rust2::Yaml;

use crate::error::ExtractorError;
use crate::indexer::{AssetIndex, UnityDocument, UnityRef, UnityYamlFile};
use crate::yaml_parser::helpers::{
    yaml_as_bool, yaml_as_f32, yaml_as_i64, yaml_as_string, yaml_get,
};

/// Result of parsing a Unity `.controller` AnimatorController asset.
#[derive(Debug, Clone, Default)]
pub struct ParsedController {
    pub parameters: Vec<ExpressionParameterDef>,
    pub layers: Vec<FxLayerDef>,
}

/// Parses an AnimatorController (`.controller`) file into parameters and `FxLayerDef`s.
pub fn parse_animator_controller(
    controller_path: &Path,
    index: &AssetIndex,
    warnings: &mut Vec<String>,
) -> Result<ParsedController, ExtractorError> {
    let yaml_file = UnityYamlFile::load(controller_path)?;
    let Some(ctrl_doc) = yaml_file
        .documents
        .iter()
        .find(|d| d.class_id == 91 || d.root_key == "AnimatorController")
    else {
        return Err(ExtractorError::YamlParse {
            path: controller_path.to_path_buf(),
            message: "no AnimatorController document found".to_owned(),
        });
    };

    let mut parameters = Vec::new();
    if let Some(params_arr) =
        yaml_get(&ctrl_doc.data, "m_AnimatorParameters").and_then(|v| v.as_vec())
    {
        for p in params_arr {
            let name = yaml_get(p, "m_Name")
                .and_then(yaml_as_string)
                .unwrap_or_default();
            if name.is_empty() {
                continue;
            }
            let ptype_num = yaml_get(p, "m_Type").and_then(yaml_as_i64).unwrap_or(4);
            let (param_type, default) = match ptype_num {
                1 => (
                    ExpressionParamType::Float,
                    yaml_get(p, "m_DefaultFloat")
                        .and_then(yaml_as_f32)
                        .unwrap_or(0.0),
                ),
                3 => (
                    ExpressionParamType::Int,
                    yaml_get(p, "m_DefaultInt")
                        .and_then(yaml_as_i64)
                        .unwrap_or(0) as f32,
                ),
                _ => (
                    ExpressionParamType::Bool,
                    if yaml_get(p, "m_DefaultBool")
                        .and_then(yaml_as_bool)
                        .unwrap_or(false)
                    {
                        1.0
                    } else {
                        0.0
                    },
                ),
            };
            parameters.push(ExpressionParameterDef {
                name,
                param_type,
                default,
                saved: false,
            });
        }
    }

    let mut layers = Vec::new();
    if let Some(layers_arr) = yaml_get(&ctrl_doc.data, "m_AnimatorLayers").and_then(|v| v.as_vec())
    {
        for layer_yaml in layers_arr {
            let layer_name = yaml_get(layer_yaml, "m_Name")
                .and_then(yaml_as_string)
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| "Layer".to_owned());
            let default_weight = yaml_get(layer_yaml, "m_DefaultWeight")
                .and_then(yaml_as_f32)
                .unwrap_or(1.0);
            let sm_fid = yaml_get(layer_yaml, "m_StateMachine")
                .and_then(UnityRef::from_yaml)
                .map(|r| r.file_id)
                .unwrap_or(0);

            if sm_fid == 0 {
                continue;
            }
            let Some(sm_doc) = yaml_file.get(sm_fid) else {
                continue;
            };

            let mut states = Vec::new();
            let mut transitions = Vec::new();
            let mut state_fid_to_name = std::collections::HashMap::new();
            let mut seen_state_machines = HashSet::new();

            collect_state_machine(
                &yaml_file,
                sm_doc,
                index,
                warnings,
                &mut states,
                &mut transitions,
                &mut state_fid_to_name,
                &mut seen_state_machines,
            )?;

            // Resolve transition target state names and filter out transitions targeting missing states
            let valid_state_names: HashSet<String> =
                states.iter().map(|s| s.name.clone()).collect();
            let resolved_transitions: Vec<FxTransitionDef> = transitions
                .into_iter()
                .filter_map(|(from_name, dst_fid, tr)| {
                    let to_name = state_fid_to_name.get(&dst_fid)?.clone();
                    if !valid_state_names.contains(&to_name) {
                        return None;
                    }
                    if let Some(ref f) = from_name {
                        if !valid_state_names.contains(f) {
                            return None;
                        }
                    }
                    Some(FxTransitionDef {
                        from_state: from_name,
                        to_state: to_name,
                        has_exit_time: tr.has_exit_time,
                        exit_time: tr.exit_time,
                        duration: tr.duration,
                        conditions: tr.conditions,
                    })
                })
                .collect();

            layers.push(FxLayerDef {
                name: layer_name,
                default_weight,
                states,
                transitions: resolved_transitions,
            });
        }
    }

    Ok(ParsedController { parameters, layers })
}

struct PendingTransition {
    has_exit_time: bool,
    exit_time: f32,
    duration: f32,
    conditions: Vec<FxConditionDef>,
}

#[allow(clippy::too_many_arguments)]
fn collect_state_machine(
    yaml_file: &UnityYamlFile,
    sm_doc: &UnityDocument,
    index: &AssetIndex,
    warnings: &mut Vec<String>,
    states: &mut Vec<FxStateDef>,
    pending_transitions: &mut Vec<(Option<String>, i64, PendingTransition)>,
    state_fid_to_name: &mut std::collections::HashMap<i64, String>,
    seen_sm: &mut HashSet<i64>,
) -> Result<(), ExtractorError> {
    if !seen_sm.insert(sm_doc.file_id) {
        return Ok(());
    }

    let default_state_fid = yaml_get(&sm_doc.data, "m_DefaultState")
        .and_then(UnityRef::from_yaml)
        .map(|r| r.file_id)
        .unwrap_or(0);

    // 1. Collect states in this state machine
    if let Some(child_states) = yaml_get(&sm_doc.data, "m_States").and_then(|v| v.as_vec()) {
        for cs in child_states {
            let state_fid = yaml_get(cs, "m_State")
                .or_else(|| yaml_get(cs, "state"))
                .and_then(UnityRef::from_yaml)
                .map(|r| r.file_id)
                .unwrap_or(0);
            if state_fid == 0 {
                continue;
            }
            let Some(state_doc) = yaml_file.get(state_fid) else {
                continue;
            };
            let state_name = yaml_get(&state_doc.data, "m_Name")
                .and_then(yaml_as_string)
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| format!("State_{state_fid}"));

            let is_default = state_fid == default_state_fid;
            let mut curves = Vec::new();
            if let Some(motion_ref) =
                yaml_get(&state_doc.data, "m_Motion").and_then(UnityRef::from_yaml)
            {
                if let Some(anim_guid) = &motion_ref.guid {
                    if let Some(anim_path) = index.resolve_guid(anim_guid) {
                        match parse_animation_clip(anim_path) {
                            Ok(clip_curves) => curves = clip_curves,
                            Err(err) => warnings.push(format!(
                                "Failed to parse animation clip '{}': {}",
                                anim_path.display(),
                                err
                            )),
                        }
                    }
                }
            }

            state_fid_to_name.insert(state_fid, state_name.clone());
            states.push(FxStateDef {
                name: state_name.clone(),
                is_default,
                curves,
            });

            if let Some(tr_arr) =
                yaml_get(&state_doc.data, "m_Transitions").and_then(|v| v.as_vec())
            {
                for tr_ref_yaml in tr_arr {
                    if let Some(tr_ref) = UnityRef::from_yaml(tr_ref_yaml) {
                        if let Some(tr_doc) = yaml_file.get(tr_ref.file_id) {
                            if let Some((dst_fid, pending)) = parse_transition_doc(tr_doc) {
                                pending_transitions.push((
                                    Some(state_name.clone()),
                                    dst_fid,
                                    pending,
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    // 2. AnyState transitions
    if let Some(any_arr) = yaml_get(&sm_doc.data, "m_AnyStateTransitions").and_then(|v| v.as_vec())
    {
        for tr_ref_yaml in any_arr {
            if let Some(tr_ref) = UnityRef::from_yaml(tr_ref_yaml) {
                if let Some(tr_doc) = yaml_file.get(tr_ref.file_id) {
                    if let Some((dst_fid, pending)) = parse_transition_doc(tr_doc) {
                        pending_transitions.push((None, dst_fid, pending));
                    }
                }
            }
        }
    }

    // 3. Nested sub-state machines
    if let Some(sub_sms) = yaml_get(&sm_doc.data, "m_ChildStateMachines").and_then(|v| v.as_vec()) {
        for sub in sub_sms {
            let sub_fid = yaml_get(sub, "m_StateMachine")
                .or_else(|| yaml_get(sub, "stateMachine"))
                .and_then(UnityRef::from_yaml)
                .map(|r| r.file_id)
                .unwrap_or(0);
            if let Some(sub_doc) = yaml_file.get(sub_fid) {
                collect_state_machine(
                    yaml_file,
                    sub_doc,
                    index,
                    warnings,
                    states,
                    pending_transitions,
                    state_fid_to_name,
                    seen_sm,
                )?;
            }
        }
    }

    Ok(())
}

fn parse_transition_doc(tr_doc: &UnityDocument) -> Option<(i64, PendingTransition)> {
    let is_exit = yaml_get(&tr_doc.data, "m_IsExit")
        .and_then(yaml_as_bool)
        .unwrap_or(false);
    if is_exit {
        return None;
    }
    let dst_fid = yaml_get(&tr_doc.data, "m_DstState")
        .and_then(UnityRef::from_yaml)
        .map(|r| r.file_id)
        .unwrap_or(0);
    if dst_fid == 0 {
        return None;
    }

    let has_exit_time = yaml_get(&tr_doc.data, "m_HasExitTime")
        .and_then(yaml_as_bool)
        .unwrap_or(false);
    let exit_time = yaml_get(&tr_doc.data, "m_ExitTime")
        .and_then(yaml_as_f32)
        .unwrap_or(1.0);
    let duration = yaml_get(&tr_doc.data, "m_TransitionDuration")
        .and_then(yaml_as_f32)
        .unwrap_or(0.15);

    let mut conditions = Vec::new();
    if let Some(conds_arr) = yaml_get(&tr_doc.data, "m_Conditions").and_then(|v| v.as_vec()) {
        for c in conds_arr {
            let param = yaml_get(c, "m_ConditionEvent")
                .and_then(yaml_as_string)
                .unwrap_or_default();
            if param.is_empty() {
                continue;
            }
            let mode_num = yaml_get(c, "m_ConditionMode")
                .and_then(yaml_as_i64)
                .unwrap_or(1);
            let mode = match mode_num {
                1 => FxConditionMode::If,
                2 => FxConditionMode::IfNot,
                3 => FxConditionMode::Greater,
                4 => FxConditionMode::Less,
                6 => FxConditionMode::Equals,
                7 => FxConditionMode::NotEqual,
                _ => FxConditionMode::If,
            };
            let threshold = yaml_get(c, "m_EventTreshold")
                .or_else(|| yaml_get(c, "m_EventThreshold"))
                .and_then(yaml_as_f32)
                .unwrap_or(0.0);
            conditions.push(FxConditionDef {
                parameter: param,
                mode,
                threshold,
            });
        }
    }

    Some((
        dst_fid,
        PendingTransition {
            has_exit_time,
            exit_time,
            duration,
            conditions,
        },
    ))
}

/// Parses a Unity `.anim` (`AnimationClip`) file into `AnimationCurveDef`s.
pub fn parse_animation_clip(anim_path: &Path) -> Result<Vec<AnimationCurveDef>, ExtractorError> {
    let yaml_file = UnityYamlFile::load(anim_path)?;
    let Some(clip_doc) = yaml_file
        .documents
        .iter()
        .find(|d| d.class_id == 74 || d.root_key == "AnimationClip")
        .or_else(|| yaml_file.documents.first())
    else {
        return Ok(Vec::new());
    };

    let mut curves = Vec::new();
    let mut seen_keys = HashSet::new();

    for section in ["m_FloatCurves", "m_EditorCurves"] {
        if let Some(arr) = yaml_get(&clip_doc.data, section).and_then(|v| v.as_vec()) {
            for curve_entry in arr {
                if let Some(parsed) = parse_single_float_curve(curve_entry) {
                    let dedup_key = format!("{}::{:?}", parsed.node_path, parsed.property);
                    if seen_keys.insert(dedup_key) {
                        curves.push(parsed);
                    }
                }
            }
        }
    }

    Ok(curves)
}

fn parse_single_float_curve(entry: &Yaml) -> Option<AnimationCurveDef> {
    let attr = yaml_get(entry, "attribute").and_then(yaml_as_string)?;
    let node_path = yaml_get(entry, "path")
        .and_then(yaml_as_string)
        .unwrap_or_default();

    let property = if let Some(bs) = attr.strip_prefix("blendShape.") {
        CurveProperty::BlendShape(bs.to_owned())
    } else if attr == "m_IsActive" {
        CurveProperty::GameObjectActive
    } else if let Some(mat_prop) = attr.strip_prefix("material.") {
        CurveProperty::MaterialFloat(mat_prop.to_owned())
    } else {
        return None;
    };

    let mut keyframes = Vec::new();
    if let Some(kf_arr) = entry["curve"]["m_Curve"].as_vec() {
        for kf in kf_arr {
            let time = yaml_get(kf, "time").and_then(yaml_as_f32).unwrap_or(0.0);
            let value = yaml_get(kf, "value").and_then(yaml_as_f32).unwrap_or(0.0);
            keyframes.push((time, value));
        }
    }

    Some(AnimationCurveDef {
        node_path,
        property,
        keyframes,
    })
}
