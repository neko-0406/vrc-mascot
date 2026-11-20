//! VRChat FX Animator state machine, parameter store, expression menu triggers, and curve evaluator.

use std::collections::HashMap;

use vrc_schema::{
    AnimationCurveDef, CurveProperty, EmotionTag, ExpressionMenuItem, ExpressionMenuItemType,
    ExpressionParamType, ExpressionsDef, FxConditionDef, FxConditionMode, FxLayerDef, FxStateDef,
};

/// Typed runtime parameter value stored in [`FxStateMachine`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParamValue {
    Bool(bool),
    Int(i32),
    Float(f32),
}

impl ParamValue {
    /// Converts this parameter to a boolean (`Int != 0`, `Float > 0.0`).
    pub fn as_bool(self) -> bool {
        match self {
            Self::Bool(b) => b,
            Self::Int(i) => i != 0,
            Self::Float(f) => f > 0.0,
        }
    }

    /// Converts this parameter to an `i32`.
    pub fn as_i32(self) -> i32 {
        match self {
            Self::Bool(b) => i32::from(b),
            Self::Int(i) => i,
            Self::Float(f) => f.round() as i32,
        }
    }

    /// Converts this parameter to an `f32`.
    pub fn as_f32(self) -> f32 {
        match self {
            Self::Bool(b) => {
                if b {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Int(i) => i as f32,
            Self::Float(f) => f,
        }
    }
}

/// Evaluated outputs from one tick of [`FxStateMachine`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FxOutput {
    pub blendshape_weights: HashMap<String, f32>,
    pub node_visibility: HashMap<String, bool>,
    pub material_floats: HashMap<(String, String), f32>,
}

#[derive(Debug, Clone)]
struct ActiveTransition {
    previous_state: usize,
    previous_state_time: f32,
    elapsed: f32,
    duration: f32,
}

#[derive(Debug, Clone)]
struct FxLayerRuntime {
    def: FxLayerDef,
    state_durations: Vec<f32>,
    current_state: usize,
    state_time: f32,
    transition: Option<ActiveTransition>,
}

impl FxLayerRuntime {
    fn from_def(mut def: FxLayerDef) -> Self {
        for state in &mut def.states {
            for curve in &mut state.curves {
                curve.keyframes.sort_by(|a, b| a.0.total_cmp(&b.0));
            }
        }

        let state_durations = def
            .states
            .iter()
            .map(|s| {
                let max_t = s
                    .curves
                    .iter()
                    .filter_map(|c| c.keyframes.last().map(|kf| kf.0))
                    .fold(0.0f32, f32::max);
                if max_t > 1e-4 {
                    max_t
                } else {
                    1.0
                }
            })
            .collect();

        let current_state = def.states.iter().position(|s| s.is_default).unwrap_or(0);

        Self {
            def,
            state_durations,
            current_state,
            state_time: 0.0,
            transition: None,
        }
    }

    fn current_state_name(&self) -> Option<&str> {
        self.def
            .states
            .get(self.current_state)
            .map(|s| s.name.as_str())
    }
}

/// Runtime FX Animator controller and VRChat Expression parameter store.
#[derive(Debug, Clone)]
pub struct FxStateMachine {
    param_defs: HashMap<String, (ExpressionParamType, f32)>,
    params: HashMap<String, ParamValue>,
    menu: Vec<ExpressionMenuItem>,
    layers: Vec<FxLayerRuntime>,
    fallback_emotion: Option<EmotionTag>,
    last_output: FxOutput,
}

impl FxStateMachine {
    /// Initializes the FX state machine from an avatar's [`ExpressionsDef`].
    pub fn new(def: &ExpressionsDef) -> Self {
        let mut param_defs = HashMap::with_capacity(def.parameters.len());
        let mut params = HashMap::with_capacity(def.parameters.len());

        for p in &def.parameters {
            let val = match p.param_type {
                ExpressionParamType::Bool => ParamValue::Bool(p.default != 0.0),
                ExpressionParamType::Int => ParamValue::Int(p.default.round() as i32),
                ExpressionParamType::Float => ParamValue::Float(p.default),
            };
            param_defs.insert(p.name.clone(), (p.param_type, p.default));
            params.insert(p.name.clone(), val);
        }

        let layers = def
            .fx_layers
            .iter()
            .cloned()
            .map(FxLayerRuntime::from_def)
            .collect();

        let mut machine = Self {
            param_defs,
            params,
            menu: def.menu.clone(),
            layers,
            fallback_emotion: None,
            last_output: FxOutput::default(),
        };
        machine.last_output = machine.evaluate_all_layers();
        machine
    }

    /// Sets a float parameter (converting to the declared parameter type if known).
    pub fn set_float(&mut self, name: &str, value: f32) {
        let coerced = match self.param_defs.get(name).map(|(t, _)| *t) {
            Some(ExpressionParamType::Bool) => ParamValue::Bool(value != 0.0),
            Some(ExpressionParamType::Int) => ParamValue::Int(value.round() as i32),
            Some(ExpressionParamType::Float) | None => ParamValue::Float(value),
        };
        self.params.insert(name.to_owned(), coerced);
    }

    /// Sets a bool parameter (converting to the declared parameter type if known).
    pub fn set_bool(&mut self, name: &str, value: bool) {
        let coerced = match self.param_defs.get(name).map(|(t, _)| *t) {
            Some(ExpressionParamType::Bool) | None => ParamValue::Bool(value),
            Some(ExpressionParamType::Int) => ParamValue::Int(i32::from(value)),
            Some(ExpressionParamType::Float) => ParamValue::Float(if value { 1.0 } else { 0.0 }),
        };
        self.params.insert(name.to_owned(), coerced);
    }

    /// Sets an int parameter (converting to the declared parameter type if known).
    pub fn set_int(&mut self, name: &str, value: i32) {
        let coerced = match self.param_defs.get(name).map(|(t, _)| *t) {
            Some(ExpressionParamType::Bool) => ParamValue::Bool(value != 0),
            Some(ExpressionParamType::Int) | None => ParamValue::Int(value),
            Some(ExpressionParamType::Float) => ParamValue::Float(value as f32),
        };
        self.params.insert(name.to_owned(), coerced);
    }

    /// Reads the current value of parameter `name`.
    pub fn get_param(&self, name: &str) -> Option<ParamValue> {
        self.params.get(name).copied()
    }

    /// Returns all current parameter values.
    pub fn all_params(&self) -> &HashMap<String, ParamValue> {
        &self.params
    }

    /// Returns the active state name for each FX layer (`(layer_name, active_state_name)`).
    pub fn active_states(&self) -> Vec<(&str, Option<&str>)> {
        self.layers
            .iter()
            .map(|l| (l.def.name.as_str(), l.current_state_name()))
            .collect()
    }

    /// Returns the most recently evaluated [`FxOutput`].
    pub fn last_output(&self) -> &FxOutput {
        &self.last_output
    }

    /// Triggers an expression menu item by name (searching nested sub-menus).
    pub fn trigger_menu_item(&mut self, item_name: &str) -> bool {
        let Some(item) = find_menu_item(&self.menu, item_name).cloned() else {
            return false;
        };
        self.apply_menu_item(&item, None)
    }

    /// Triggers a `RadialPuppet` (or other menu item) with an explicit value override.
    pub fn trigger_menu_item_with_value(&mut self, item_name: &str, value: f32) -> bool {
        let Some(item) = find_menu_item(&self.menu, item_name).cloned() else {
            return false;
        };
        self.apply_menu_item(&item, Some(value))
    }

    fn apply_menu_item(&mut self, item: &ExpressionMenuItem, override_val: Option<f32>) -> bool {
        let Some(param_name) = &item.parameter else {
            return false;
        };

        match item.item_type {
            ExpressionMenuItemType::Toggle => {
                let default_val = self
                    .param_defs
                    .get(param_name)
                    .map(|(_, d)| *d)
                    .unwrap_or(0.0);
                let current = self.params.get(param_name).copied();

                match current {
                    Some(ParamValue::Bool(b)) => {
                        self.set_bool(param_name, !b);
                    }
                    Some(ParamValue::Int(i)) => {
                        let target = item.value.round() as i32;
                        if i == target {
                            self.set_int(param_name, default_val.round() as i32);
                        } else {
                            self.set_int(param_name, target);
                        }
                    }
                    Some(ParamValue::Float(f)) if (f - item.value).abs() < 1e-4 => {
                        self.set_float(param_name, default_val);
                    }
                    Some(ParamValue::Float(_)) => {
                        self.set_float(param_name, item.value);
                    }
                    None => {
                        self.set_float(param_name, item.value);
                    }
                }
                true
            }
            ExpressionMenuItemType::Button => {
                let v = override_val.unwrap_or(item.value);
                self.set_float(param_name, v);
                true
            }
            ExpressionMenuItemType::RadialPuppet => {
                let v = override_val.unwrap_or(item.value).clamp(0.0, 1.0);
                self.set_float(param_name, v);
                true
            }
            ExpressionMenuItemType::SubMenu => false,
        }
    }

    /// Maps an AI [`EmotionTag`] onto matching FX expression parameters or fallback BlendShapes.
    pub fn set_emotion(&mut self, emotion: &EmotionTag) {
        let keywords = emotion_keywords(emotion);
        let mut matched_fx = false;

        // 1. Inspect FX layer states & incoming transitions to find a matching emotion state.
        let mut param_assignments: Vec<(String, f32)> = Vec::new();
        for layer in &self.layers {
            for tr in &layer.def.transitions {
                if matches_any_keyword(&tr.to_state, &keywords) {
                    for cond in &tr.conditions {
                        match cond.mode {
                            FxConditionMode::Equals => {
                                param_assignments.push((cond.parameter.clone(), cond.threshold));
                            }
                            FxConditionMode::If => {
                                param_assignments.push((cond.parameter.clone(), 1.0));
                            }
                            FxConditionMode::Greater => {
                                param_assignments
                                    .push((cond.parameter.clone(), cond.threshold + 0.5));
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        if !param_assignments.is_empty() {
            for (name, val) in param_assignments {
                self.set_float(&name, val);
            }
            matched_fx = true;
        } else if matches!(emotion, EmotionTag::Neutral) {
            // Reset common emotion-named Int/Bool parameters to defaults.
            let defaults: Vec<(String, f32)> = self
                .param_defs
                .iter()
                .filter(|(name, _)| {
                    let lower = name.to_lowercase();
                    lower.contains("emotion")
                        || lower.contains("expression")
                        || lower.contains("face")
                        || lower.contains("mood")
                })
                .map(|(name, (_, def))| (name.clone(), *def))
                .collect();
            if !defaults.is_empty() {
                for (name, def) in defaults {
                    self.set_float(&name, def);
                }
                matched_fx = true;
            }
        }

        // 2. Also toggle any Bool parameter directly named after an emotion.
        let all_emotion_words = [
            "happy",
            "smile",
            "sad",
            "angry",
            "surprised",
            "shy",
            "blush",
            "sleepy",
        ];
        let param_names: Vec<String> = self.param_defs.keys().cloned().collect();
        for pname in param_names {
            if let Some((ExpressionParamType::Bool, _)) = self.param_defs.get(&pname) {
                let lower = pname.to_lowercase();
                if matches_any_keyword(&lower, &keywords) {
                    self.set_bool(&pname, !matches!(emotion, EmotionTag::Neutral));
                    matched_fx = true;
                } else if all_emotion_words.iter().any(|w| lower.contains(w)) {
                    self.set_bool(&pname, false);
                }
            }
        }

        self.fallback_emotion = if matched_fx {
            None
        } else {
            Some(emotion.clone())
        };
    }

    /// Evaluates FX transitions and animation curves for `dt` seconds.
    pub fn step(&mut self, dt: f32) -> FxOutput {
        let safe_dt = if dt.is_finite() && dt > 0.0 {
            dt.min(0.25)
        } else {
            0.0
        };

        let params = &self.params;
        for layer in &mut self.layers {
            if layer.def.states.is_empty() {
                continue;
            }

            layer.state_time += safe_dt;
            if let Some(tr) = &mut layer.transition {
                tr.previous_state_time += safe_dt;
                tr.elapsed += safe_dt;
                if tr.elapsed >= tr.duration {
                    layer.transition = None;
                }
            }

            if layer.transition.is_none() {
                let curr_name = layer
                    .def
                    .states
                    .get(layer.current_state)
                    .map(|s| s.name.as_str())
                    .unwrap_or("");
                let duration = layer
                    .state_durations
                    .get(layer.current_state)
                    .copied()
                    .unwrap_or(1.0)
                    .max(1e-4);
                let normalized_time = layer.state_time / duration;

                // Evaluate AnyState transitions first, then transitions from current state.
                let matched_transition = layer
                    .def
                    .transitions
                    .iter()
                    .filter(|tr| tr.from_state.is_none() && tr.to_state != curr_name)
                    .chain(
                        layer
                            .def
                            .transitions
                            .iter()
                            .filter(|tr| tr.from_state.as_deref() == Some(curr_name)),
                    )
                    .find(|tr| {
                        if tr.has_exit_time && normalized_time < tr.exit_time {
                            return false;
                        }
                        tr.conditions
                            .iter()
                            .all(|cond| evaluate_condition(cond, params))
                    })
                    .cloned();

                if let Some(tr) = matched_transition {
                    if let Some(target_idx) =
                        layer.def.states.iter().position(|s| s.name == tr.to_state)
                    {
                        if tr.duration > 1e-4 {
                            layer.transition = Some(ActiveTransition {
                                previous_state: layer.current_state,
                                previous_state_time: layer.state_time,
                                elapsed: 0.0,
                                duration: tr.duration,
                            });
                        }
                        layer.current_state = target_idx;
                        layer.state_time = 0.0;
                    }
                }
            }
        }

        let mut output = self.evaluate_all_layers();
        if let Some(fallback) = &self.fallback_emotion {
            apply_fallback_emotion_weights(fallback, &mut output.blendshape_weights);
        }
        self.last_output = output.clone();
        output
    }

    fn evaluate_all_layers(&self) -> FxOutput {
        let mut output = FxOutput::default();

        for layer in &self.layers {
            let layer_weight = layer.def.default_weight.clamp(0.0, 1.0);
            if layer_weight <= 0.0 || layer.def.states.is_empty() {
                continue;
            }

            let Some(curr_state) = layer.def.states.get(layer.current_state) else {
                continue;
            };
            let mut layer_eval = evaluate_state_curves(curr_state, layer.state_time);

            if let Some(tr) = &layer.transition {
                if let Some(prev_state) = layer.def.states.get(tr.previous_state) {
                    let prev_eval = evaluate_state_curves(prev_state, tr.previous_state_time);
                    let alpha = (tr.elapsed / tr.duration.max(1e-4)).clamp(0.0, 1.0);
                    layer_eval = blend_state_evals(&prev_eval, &layer_eval, alpha);
                }
            }

            for (name, val) in layer_eval.blendshapes {
                output
                    .blendshape_weights
                    .entry(name)
                    .and_modify(|existing| {
                        *existing = *existing * (1.0 - layer_weight) + val * layer_weight
                    })
                    .or_insert(val * layer_weight);
            }
            for (node_path, active) in layer_eval.visibility {
                output.node_visibility.insert(node_path, active);
            }
            for (key, val) in layer_eval.material_floats {
                output
                    .material_floats
                    .entry(key)
                    .and_modify(|existing| {
                        *existing = *existing * (1.0 - layer_weight) + val * layer_weight
                    })
                    .or_insert(val * layer_weight);
            }
        }

        output
    }
}

#[derive(Debug, Default)]
struct StateEval {
    blendshapes: HashMap<String, f32>,
    visibility: HashMap<String, bool>,
    material_floats: HashMap<(String, String), f32>,
}

fn evaluate_state_curves(state: &FxStateDef, time: f32) -> StateEval {
    let mut eval = StateEval::default();
    for curve in &state.curves {
        let Some(val) = sample_curve(curve, time) else {
            continue;
        };
        match &curve.property {
            CurveProperty::BlendShape(name) => {
                eval.blendshapes.insert(name.clone(), val);
            }
            CurveProperty::GameObjectActive => {
                eval.visibility.insert(curve.node_path.clone(), val >= 0.5);
            }
            CurveProperty::MaterialFloat(prop) => {
                eval.material_floats
                    .insert((curve.node_path.clone(), prop.clone()), val);
            }
        }
    }
    eval
}

fn blend_state_evals(prev: &StateEval, next: &StateEval, alpha: f32) -> StateEval {
    let mut out = StateEval::default();

    for (k, &v0) in &prev.blendshapes {
        let v1 = next.blendshapes.get(k).copied().unwrap_or(0.0);
        out.blendshapes.insert(k.clone(), v0 + (v1 - v0) * alpha);
    }
    for (k, &v1) in &next.blendshapes {
        if !prev.blendshapes.contains_key(k) {
            out.blendshapes.insert(k.clone(), v1 * alpha);
        }
    }

    for (k, &b) in &prev.visibility {
        out.visibility.insert(k.clone(), b);
    }
    if alpha >= 0.5 {
        for (k, &b) in &next.visibility {
            out.visibility.insert(k.clone(), b);
        }
    }

    for (k, &v0) in &prev.material_floats {
        let v1 = next.material_floats.get(k).copied().unwrap_or(v0);
        out.material_floats
            .insert(k.clone(), v0 + (v1 - v0) * alpha);
    }
    for (k, &v1) in &next.material_floats {
        if !prev.material_floats.contains_key(k) {
            out.material_floats.insert(k.clone(), v1);
        }
    }

    out
}

fn sample_curve(curve: &AnimationCurveDef, time: f32) -> Option<f32> {
    let kfs = &curve.keyframes;
    if kfs.is_empty() {
        return None;
    }
    if kfs.len() == 1 || time <= kfs[0].0 {
        return Some(kfs[0].1);
    }
    let last = kfs[kfs.len() - 1];
    if time >= last.0 {
        return Some(last.1);
    }

    for window in kfs.windows(2) {
        let (t0, v0) = window[0];
        let (t1, v1) = window[1];
        if time >= t0 && time <= t1 {
            let span = t1 - t0;
            if span <= 1e-6 {
                return Some(v1);
            }
            let frac = (time - t0) / span;
            return Some(v0 + (v1 - v0) * frac);
        }
    }
    Some(last.1)
}

fn evaluate_condition(cond: &FxConditionDef, params: &HashMap<String, ParamValue>) -> bool {
    let val = params
        .get(&cond.parameter)
        .copied()
        .unwrap_or(ParamValue::Float(0.0));

    match cond.mode {
        FxConditionMode::If => val.as_bool(),
        FxConditionMode::IfNot => !val.as_bool(),
        FxConditionMode::Greater => val.as_f32() > cond.threshold,
        FxConditionMode::Less => val.as_f32() < cond.threshold,
        FxConditionMode::Equals => match val {
            ParamValue::Bool(b) => i32::from(b) == cond.threshold.round() as i32,
            ParamValue::Int(i) => i == cond.threshold.round() as i32,
            ParamValue::Float(f) => (f - cond.threshold).abs() < 1e-4,
        },
        FxConditionMode::NotEqual => match val {
            ParamValue::Bool(b) => i32::from(b) != cond.threshold.round() as i32,
            ParamValue::Int(i) => i != cond.threshold.round() as i32,
            ParamValue::Float(f) => (f - cond.threshold).abs() >= 1e-4,
        },
    }
}

fn find_menu_item<'a>(
    items: &'a [ExpressionMenuItem],
    name: &str,
) -> Option<&'a ExpressionMenuItem> {
    for item in items {
        if item.name == name {
            return Some(item);
        }
        if !item.sub_menu.is_empty() {
            if let Some(found) = find_menu_item(&item.sub_menu, name) {
                return Some(found);
            }
        }
    }
    None
}

fn emotion_keywords(emotion: &EmotionTag) -> Vec<String> {
    match emotion {
        EmotionTag::Neutral => vec![
            "neutral".to_owned(),
            "idle".to_owned(),
            "default".to_owned(),
            "normal".to_owned(),
        ],
        EmotionTag::Happy => vec![
            "happy".to_owned(),
            "smile".to_owned(),
            "joy".to_owned(),
            "laugh".to_owned(),
            "笑顔".to_owned(),
        ],
        EmotionTag::Sad => vec![
            "sad".to_owned(),
            "sorrow".to_owned(),
            "cry".to_owned(),
            "悲".to_owned(),
        ],
        EmotionTag::Angry => vec!["angry".to_owned(), "mad".to_owned(), "怒".to_owned()],
        EmotionTag::Surprised => vec![
            "surprised".to_owned(),
            "surprise".to_owned(),
            "shock".to_owned(),
            "驚".to_owned(),
        ],
        EmotionTag::Shy => vec![
            "shy".to_owned(),
            "blush".to_owned(),
            "embarrassed".to_owned(),
            "照".to_owned(),
        ],
        EmotionTag::Sleepy => vec![
            "sleepy".to_owned(),
            "sleep".to_owned(),
            "tired".to_owned(),
            "眠".to_owned(),
        ],
        EmotionTag::Custom(s) => vec![s.to_lowercase()],
    }
}

fn matches_any_keyword(target: &str, keywords: &[String]) -> bool {
    let lower = target.to_lowercase();
    keywords.iter().any(|k| lower.contains(k.as_str()))
}

fn apply_fallback_emotion_weights(emotion: &EmotionTag, weights: &mut HashMap<String, f32>) {
    match emotion {
        EmotionTag::Neutral => {}
        EmotionTag::Happy => {
            weights.entry("smile".to_owned()).or_insert(0.8);
            weights.entry("happy".to_owned()).or_insert(0.8);
        }
        EmotionTag::Sad => {
            weights.entry("sad".to_owned()).or_insert(0.8);
        }
        EmotionTag::Angry => {
            weights.entry("angry".to_owned()).or_insert(0.8);
        }
        EmotionTag::Surprised => {
            weights.entry("surprised".to_owned()).or_insert(0.8);
        }
        EmotionTag::Shy => {
            weights.entry("blush".to_owned()).or_insert(0.8);
            weights.entry("shy".to_owned()).or_insert(0.8);
        }
        EmotionTag::Sleepy => {
            weights.entry("sleepy".to_owned()).or_insert(0.7);
        }
        EmotionTag::Custom(name) => {
            weights.entry(name.clone()).or_insert(0.8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vrc_schema::{ExpressionParameterDef, FxTransitionDef};

    fn sample_expressions() -> ExpressionsDef {
        ExpressionsDef {
            parameters: vec![
                ExpressionParameterDef {
                    name: "FaceEmotion".to_owned(),
                    param_type: ExpressionParamType::Int,
                    default: 0.0,
                    saved: true,
                },
                ExpressionParameterDef {
                    name: "JacketOn".to_owned(),
                    param_type: ExpressionParamType::Bool,
                    default: 1.0,
                    saved: true,
                },
                ExpressionParameterDef {
                    name: "BlushAmount".to_owned(),
                    param_type: ExpressionParamType::Float,
                    default: 0.0,
                    saved: false,
                },
            ],
            menu: vec![
                ExpressionMenuItem {
                    name: "Smile".to_owned(),
                    item_type: ExpressionMenuItemType::Toggle,
                    parameter: Some("FaceEmotion".to_owned()),
                    value: 1.0,
                    sub_menu: vec![],
                },
                ExpressionMenuItem {
                    name: "Jacket".to_owned(),
                    item_type: ExpressionMenuItemType::Toggle,
                    parameter: Some("JacketOn".to_owned()),
                    value: 1.0,
                    sub_menu: vec![],
                },
            ],
            fx_layers: vec![
                FxLayerDef {
                    name: "FaceLayer".to_owned(),
                    default_weight: 1.0,
                    states: vec![
                        FxStateDef {
                            name: "Neutral".to_owned(),
                            is_default: true,
                            curves: vec![AnimationCurveDef {
                                node_path: "Body".to_owned(),
                                property: CurveProperty::BlendShape("smile".to_owned()),
                                keyframes: vec![(0.0, 0.0)],
                            }],
                        },
                        FxStateDef {
                            name: "Smile".to_owned(),
                            is_default: false,
                            curves: vec![AnimationCurveDef {
                                node_path: "Body".to_owned(),
                                property: CurveProperty::BlendShape("smile".to_owned()),
                                keyframes: vec![(0.0, 0.0), (0.2, 1.0)],
                            }],
                        },
                        FxStateDef {
                            name: "BlushOverride".to_owned(),
                            is_default: false,
                            curves: vec![AnimationCurveDef {
                                node_path: "Body".to_owned(),
                                property: CurveProperty::BlendShape("blush".to_owned()),
                                keyframes: vec![(0.0, 1.0)],
                            }],
                        },
                    ],
                    transitions: vec![
                        FxTransitionDef {
                            from_state: Some("Neutral".to_owned()),
                            to_state: "Smile".to_owned(),
                            has_exit_time: false,
                            exit_time: 1.0,
                            duration: 0.1,
                            conditions: vec![FxConditionDef {
                                parameter: "FaceEmotion".to_owned(),
                                mode: FxConditionMode::Equals,
                                threshold: 1.0,
                            }],
                        },
                        FxTransitionDef {
                            from_state: Some("Smile".to_owned()),
                            to_state: "Neutral".to_owned(),
                            has_exit_time: false,
                            exit_time: 1.0,
                            duration: 0.0,
                            conditions: vec![FxConditionDef {
                                parameter: "FaceEmotion".to_owned(),
                                mode: FxConditionMode::Equals,
                                threshold: 0.0,
                            }],
                        },
                        FxTransitionDef {
                            from_state: None,
                            to_state: "BlushOverride".to_owned(),
                            has_exit_time: false,
                            exit_time: 1.0,
                            duration: 0.0,
                            conditions: vec![FxConditionDef {
                                parameter: "BlushAmount".to_owned(),
                                mode: FxConditionMode::Greater,
                                threshold: 0.5,
                            }],
                        },
                    ],
                },
                FxLayerDef {
                    name: "OutfitLayer".to_owned(),
                    default_weight: 1.0,
                    states: vec![
                        FxStateDef {
                            name: "JacketVisible".to_owned(),
                            is_default: true,
                            curves: vec![AnimationCurveDef {
                                node_path: "Armature/Jacket".to_owned(),
                                property: CurveProperty::GameObjectActive,
                                keyframes: vec![(0.0, 1.0)],
                            }],
                        },
                        FxStateDef {
                            name: "JacketHidden".to_owned(),
                            is_default: false,
                            curves: vec![AnimationCurveDef {
                                node_path: "Armature/Jacket".to_owned(),
                                property: CurveProperty::GameObjectActive,
                                keyframes: vec![(0.0, 0.0)],
                            }],
                        },
                    ],
                    transitions: vec![
                        FxTransitionDef {
                            from_state: Some("JacketVisible".to_owned()),
                            to_state: "JacketHidden".to_owned(),
                            has_exit_time: false,
                            exit_time: 1.0,
                            duration: 0.0,
                            conditions: vec![FxConditionDef {
                                parameter: "JacketOn".to_owned(),
                                mode: FxConditionMode::IfNot,
                                threshold: 0.0,
                            }],
                        },
                        FxTransitionDef {
                            from_state: Some("JacketHidden".to_owned()),
                            to_state: "JacketVisible".to_owned(),
                            has_exit_time: false,
                            exit_time: 1.0,
                            duration: 0.0,
                            conditions: vec![FxConditionDef {
                                parameter: "JacketOn".to_owned(),
                                mode: FxConditionMode::If,
                                threshold: 0.0,
                            }],
                        },
                    ],
                },
            ],
        }
    }

    #[test]
    fn fx_transitions_equals_greater_if_ifnot_anystate_and_curves() {
        let mut fx = FxStateMachine::new(&sample_expressions());
        let init = fx.step(0.016);
        assert_eq!(init.blendshape_weights.get("smile").copied(), Some(0.0));
        assert_eq!(
            init.node_visibility.get("Armature/Jacket").copied(),
            Some(true)
        );

        // Toggle Smile via menu -> Equals 1 -> Smile state with BlendShape curve.
        assert!(fx.trigger_menu_item("Smile"));
        assert_eq!(fx.get_param("FaceEmotion"), Some(ParamValue::Int(1)));
        fx.step(0.016); // triggers transition
        let out = fx.step(0.25); // transition completes & curve reaches 1.0
        assert!(
            out.blendshape_weights.get("smile").copied().unwrap_or(0.0) > 0.95,
            "expected smile blendshape near 1.0, got {:?}",
            out.blendshape_weights
        );

        // Toggle Jacket via menu -> IfNot -> JacketHidden GameObjectActive = false.
        assert!(fx.trigger_menu_item("Jacket"));
        assert_eq!(fx.get_param("JacketOn"), Some(ParamValue::Bool(false)));
        let out2 = fx.step(0.016);
        assert_eq!(
            out2.node_visibility.get("Armature/Jacket").copied(),
            Some(false)
        );

        // Toggle Jacket back ON -> If -> JacketVisible GameObjectActive = true.
        assert!(fx.trigger_menu_item("Jacket"));
        let out3 = fx.step(0.016);
        assert_eq!(
            out3.node_visibility.get("Armature/Jacket").copied(),
            Some(true)
        );

        // Trigger AnyState via Greater > 0.5 -> BlushOverride.
        fx.set_float("BlushAmount", 0.8);
        let out4 = fx.step(0.016);
        assert_eq!(out4.blendshape_weights.get("blush").copied(), Some(1.0));
    }

    #[test]
    fn emotion_tag_drives_matching_fx_parameter() {
        let mut fx = FxStateMachine::new(&sample_expressions());
        fx.set_emotion(&EmotionTag::Happy);
        assert_eq!(fx.get_param("FaceEmotion"), Some(ParamValue::Int(1)));

        fx.step(0.016);
        let out = fx.step(0.25);
        assert!(out.blendshape_weights.get("smile").copied().unwrap_or(0.0) > 0.9);
    }

    #[test]
    fn has_exit_time_and_material_float_curves_evaluate_cleanly() {
        let def = ExpressionsDef {
            parameters: vec![ExpressionParameterDef {
                name: "GlowLevel".to_owned(),
                param_type: ExpressionParamType::Float,
                default: 0.2,
                saved: false,
            }],
            menu: vec![ExpressionMenuItem {
                name: "GlowDial".to_owned(),
                item_type: ExpressionMenuItemType::RadialPuppet,
                parameter: Some("GlowLevel".to_owned()),
                value: 0.7,
                sub_menu: vec![],
            }],
            fx_layers: vec![FxLayerDef {
                name: "MaterialLayer".to_owned(),
                default_weight: 1.0,
                states: vec![
                    FxStateDef {
                        name: "Idle".to_owned(),
                        is_default: true,
                        curves: vec![AnimationCurveDef {
                            node_path: "Body".to_owned(),
                            property: CurveProperty::MaterialFloat("_EmissionStrength".to_owned()),
                            keyframes: vec![(0.0, 0.0), (0.5, 0.5)],
                        }],
                    },
                    FxStateDef {
                        name: "Glowing".to_owned(),
                        is_default: false,
                        curves: vec![AnimationCurveDef {
                            node_path: "Body".to_owned(),
                            property: CurveProperty::MaterialFloat("_EmissionStrength".to_owned()),
                            keyframes: vec![(0.0, 2.0)],
                        }],
                    },
                ],
                transitions: vec![FxTransitionDef {
                    from_state: Some("Idle".to_owned()),
                    to_state: "Glowing".to_owned(),
                    has_exit_time: true,
                    exit_time: 1.0,
                    duration: 0.0,
                    conditions: vec![FxConditionDef {
                        parameter: "GlowLevel".to_owned(),
                        mode: FxConditionMode::Greater,
                        threshold: 0.5,
                    }],
                }],
            }],
        };

        let mut fx = FxStateMachine::new(&def);
        assert!(fx.trigger_menu_item_with_value("GlowDial", 0.9));
        // Before exit_time (0.5s), must stay in Idle.
        let mid = fx.step(0.25);
        assert_eq!(fx.active_states(), vec![("MaterialLayer", Some("Idle"))]);
        assert!(
            (mid.material_floats
                .get(&("Body".to_owned(), "_EmissionStrength".to_owned()))
                .copied()
                .unwrap_or(-1.0)
                - 0.25)
                .abs()
                < 1e-4
        );

        // Passing 0.5s fulfills has_exit_time=true, exit_time=1.0 -> Glowing.
        fx.step(0.25);
        let done = fx.step(0.016);
        assert_eq!(fx.active_states(), vec![("MaterialLayer", Some("Glowing"))]);
        assert_eq!(
            done.material_floats
                .get(&("Body".to_owned(), "_EmissionStrength".to_owned()))
                .copied(),
            Some(2.0)
        );
    }
}
