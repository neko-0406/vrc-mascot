use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::SchemaError;
use crate::runtime::VisemeKey;

/// Current semantic version of the `mascot_manifest.json` schema.
pub const SCHEMA_VERSION: &str = "1.0.0";

fn default_schema_version() -> String {
    SCHEMA_VERSION.to_owned()
}

fn default_true() -> bool {
    true
}

fn default_one() -> f32 {
    1.0
}

fn default_pull() -> f32 {
    0.2
}

fn default_spring() -> f32 {
    0.2
}

fn default_stiffness() -> f32 {
    0.2
}

fn default_physbone_radius() -> f32 {
    0.02
}

fn default_collider_radius() -> f32 {
    0.1
}

fn default_identity_quat() -> [f32; 4] {
    [0.0, 0.0, 0.0, 1.0]
}

fn default_transition_duration() -> f32 {
    0.15
}

fn default_render_queue() -> i32 {
    2000
}

fn default_rgba_white() -> [f32; 4] {
    [1.0, 1.0, 1.0, 1.0]
}

fn default_shadow_color_1() -> [f32; 4] {
    [0.85, 0.85, 0.9, 1.0]
}

fn default_shadow_color_2() -> [f32; 4] {
    [0.7, 0.7, 0.8, 1.0]
}

fn default_shadow_border() -> f32 {
    0.5
}

fn default_shadow_blur() -> f32 {
    0.1
}

fn default_shadow_2nd_border() -> f32 {
    0.2
}

fn default_rim_border() -> f32 {
    0.5
}

fn default_rim_fresnel_power() -> f32 {
    3.0
}

fn default_emission_color() -> [f32; 4] {
    [0.0, 0.0, 0.0, 1.0]
}

fn default_outline_width() -> f32 {
    0.002
}

fn default_outline_color() -> [f32; 4] {
    [0.1, 0.1, 0.1, 1.0]
}

fn default_fur_length() -> f32 {
    0.01
}

fn default_fur_layers() -> u32 {
    8
}

fn default_fur_gravity() -> f32 {
    0.25
}

fn default_alpha_cutoff() -> f32 {
    0.5
}

/// Top-level metadata manifest (`mascot_manifest.json`) paired with `avatar.glb`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MascotManifest {
    #[serde(default = "default_schema_version")]
    pub schema_version: String,
    pub avatar_name: String,
    pub glb_file: String,
    #[serde(default)]
    pub humanoid_bones: BTreeMap<String, String>,
    #[serde(default)]
    pub viseme_map: VisemeMap,
    #[serde(default)]
    pub phys_bones: Vec<PhysBoneDef>,
    #[serde(default)]
    pub colliders: Vec<ColliderDef>,
    #[serde(default)]
    pub expressions: ExpressionsDef,
    #[serde(default)]
    pub materials: Vec<MaterialDef>,
}

impl Default for MascotManifest {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION.to_owned(),
            avatar_name: String::new(),
            glb_file: "avatar.glb".to_owned(),
            humanoid_bones: BTreeMap::new(),
            viseme_map: VisemeMap::default(),
            phys_bones: Vec::new(),
            colliders: Vec::new(),
            expressions: ExpressionsDef::default(),
            materials: Vec::new(),
        }
    }
}

impl MascotManifest {
    /// Parses a `MascotManifest` from a JSON string and validates it.
    pub fn from_json_str(json: &str) -> Result<Self, SchemaError> {
        let manifest: Self = serde_json::from_str(json)?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Serializes this manifest to pretty-printed JSON.
    pub fn to_json_string_pretty(&self) -> Result<String, SchemaError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Loads and validates a `mascot_manifest.json` file from disk.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, SchemaError> {
        let contents = fs::read_to_string(path)?;
        Self::from_json_str(&contents)
    }

    /// Validates and writes this manifest as pretty JSON to disk, creating parent directories if needed.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), SchemaError> {
        self.validate()?;
        let path = path.as_ref();
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        let json = self.to_json_string_pretty()?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Validates semantic consistency of the manifest.
    pub fn validate(&self) -> Result<(), SchemaError> {
        let expected_major = SCHEMA_VERSION.split('.').next().unwrap_or("1");
        let found_major = self.schema_version.split('.').next().unwrap_or("");
        if self.schema_version.trim().is_empty() || found_major != expected_major {
            return Err(SchemaError::UnsupportedSchemaVersion {
                expected: SCHEMA_VERSION.to_owned(),
                found: self.schema_version.clone(),
            });
        }

        if self.avatar_name.trim().is_empty() {
            return Err(SchemaError::Validation(
                "avatar_name must not be empty".to_owned(),
            ));
        }
        if self.glb_file.trim().is_empty() {
            return Err(SchemaError::Validation(
                "glb_file must not be empty".to_owned(),
            ));
        }

        let mut collider_ids = HashSet::with_capacity(self.colliders.len());
        for col in &self.colliders {
            if col.id.trim().is_empty() {
                return Err(SchemaError::Validation(
                    "collider id must not be empty".to_owned(),
                ));
            }
            if !collider_ids.insert(col.id.as_str()) {
                return Err(SchemaError::Validation(format!(
                    "duplicate collider id: {}",
                    col.id
                )));
            }
            if !col.radius.is_finite() || col.radius < 0.0 {
                return Err(SchemaError::Validation(format!(
                    "collider '{}' has invalid radius: {}",
                    col.id, col.radius
                )));
            }
        }

        for pb in &self.phys_bones {
            if pb.root_node.trim().is_empty() {
                return Err(SchemaError::Validation(
                    "phys_bone root_node must not be empty".to_owned(),
                ));
            }
            if !pb.radius.is_finite() || pb.radius < 0.0 {
                return Err(SchemaError::Validation(format!(
                    "phys_bone '{}' has invalid radius: {}",
                    pb.root_node, pb.radius
                )));
            }
            for col_ref in &pb.collider_refs {
                if !collider_ids.contains(col_ref.as_str()) {
                    return Err(SchemaError::Validation(format!(
                        "phys_bone '{}' references unknown collider '{}'",
                        pb.root_node, col_ref
                    )));
                }
            }
        }

        for layer in &self.expressions.fx_layers {
            if layer.name.trim().is_empty() {
                return Err(SchemaError::Validation(
                    "fx_layer name must not be empty".to_owned(),
                ));
            }
            let state_names: HashSet<&str> = layer.states.iter().map(|s| s.name.as_str()).collect();
            for tr in &layer.transitions {
                if let Some(from) = &tr.from_state {
                    if !state_names.contains(from.as_str()) {
                        return Err(SchemaError::Validation(format!(
                            "fx_layer '{}' transition references unknown from_state '{}'",
                            layer.name, from
                        )));
                    }
                }
                if !state_names.contains(tr.to_state.as_str()) {
                    return Err(SchemaError::Validation(format!(
                        "fx_layer '{}' transition references unknown to_state '{}'",
                        layer.name, tr.to_state
                    )));
                }
            }
        }

        for mat in &self.materials {
            if mat.params.fur_enable && mat.params.fur_layers == 0 {
                return Err(SchemaError::Validation(format!(
                    "material '{}' has fur enabled with fur_layers == 0",
                    mat.name
                )));
            }
        }

        Ok(())
    }
}

/// Mapping from VRChat lip-sync viseme slots to target mesh BlendShapes.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct VisemeMap {
    pub sil: Option<BlendShapeBinding>,
    pub aa: Option<BlendShapeBinding>,
    pub ih: Option<BlendShapeBinding>,
    pub ou: Option<BlendShapeBinding>,
    pub ee: Option<BlendShapeBinding>,
    pub oh: Option<BlendShapeBinding>,
}

impl VisemeMap {
    /// Returns the BlendShape binding associated with `key`, if configured.
    pub fn get(&self, key: VisemeKey) -> Option<&BlendShapeBinding> {
        match key {
            VisemeKey::Sil => self.sil.as_ref(),
            VisemeKey::Aa => self.aa.as_ref(),
            VisemeKey::Ih => self.ih.as_ref(),
            VisemeKey::Ou => self.ou.as_ref(),
            VisemeKey::Ee => self.ee.as_ref(),
            VisemeKey::Oh => self.oh.as_ref(),
        }
    }
}

/// Reference to a BlendShape channel on a specific mesh node in `avatar.glb`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlendShapeBinding {
    pub mesh: String,
    pub blendshape: String,
}

/// Angle/hinge/polar limit mode matching `VRCPhysBone.LimitType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum PhysBoneLimitType {
    #[default]
    None,
    Angle,
    Hinge,
    Polar,
}

/// Definition of a `VRCPhysBone` simulation chain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhysBoneDef {
    pub root_node: String,
    #[serde(default)]
    pub ignore_nodes: Vec<String>,
    #[serde(default = "default_pull")]
    pub pull: f32,
    #[serde(default = "default_spring")]
    pub spring: f32,
    #[serde(default = "default_stiffness")]
    pub stiffness: f32,
    #[serde(default)]
    pub immovable: f32,
    #[serde(default)]
    pub gravity: f32,
    #[serde(default)]
    pub gravity_falloff: f32,
    #[serde(default = "default_physbone_radius")]
    pub radius: f32,
    #[serde(default)]
    pub limit_type: PhysBoneLimitType,
    #[serde(default)]
    pub max_angle_x: f32,
    #[serde(default)]
    pub max_angle_z: f32,
    #[serde(default)]
    pub collider_refs: Vec<String>,
    #[serde(default = "default_true")]
    pub allow_grabbing: bool,
    #[serde(default = "default_true")]
    pub allow_posing: bool,
}

impl Default for PhysBoneDef {
    fn default() -> Self {
        Self {
            root_node: String::new(),
            ignore_nodes: Vec::new(),
            pull: default_pull(),
            spring: default_spring(),
            stiffness: default_stiffness(),
            immovable: 0.0,
            gravity: 0.0,
            gravity_falloff: 0.0,
            radius: default_physbone_radius(),
            limit_type: PhysBoneLimitType::None,
            max_angle_x: 0.0,
            max_angle_z: 0.0,
            collider_refs: Vec::new(),
            allow_grabbing: true,
            allow_posing: true,
        }
    }
}

/// Geometric shape of a `VRCPhysBoneCollider`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ColliderShape {
    #[default]
    Sphere,
    Capsule,
    Plane,
}

/// Definition of a `VRCPhysBoneCollider` attached to a skeleton node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColliderDef {
    pub id: String,
    pub parent_node: String,
    #[serde(default)]
    pub shape: ColliderShape,
    #[serde(default = "default_collider_radius")]
    pub radius: f32,
    #[serde(default)]
    pub height: f32,
    #[serde(default)]
    pub position: [f32; 3],
    #[serde(default = "default_identity_quat")]
    pub rotation: [f32; 4],
    #[serde(default)]
    pub inside_bounds: bool,
}

impl Default for ColliderDef {
    fn default() -> Self {
        Self {
            id: String::new(),
            parent_node: String::new(),
            shape: ColliderShape::Sphere,
            radius: default_collider_radius(),
            height: 0.0,
            position: [0.0; 3],
            rotation: default_identity_quat(),
            inside_bounds: false,
        }
    }
}

/// Extracted VRChat expression parameters, menu hierarchy, and FX animator layers.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ExpressionsDef {
    pub parameters: Vec<ExpressionParameterDef>,
    pub menu: Vec<ExpressionMenuItem>,
    pub fx_layers: Vec<FxLayerDef>,
}

/// Parameter value type in `VRCExpressionParameters`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ExpressionParamType {
    #[default]
    Bool,
    Int,
    Float,
}

/// Single entry in `VRCExpressionParameters`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpressionParameterDef {
    pub name: String,
    #[serde(rename = "type")]
    pub param_type: ExpressionParamType,
    #[serde(default)]
    pub default: f32,
    #[serde(default = "default_true")]
    pub saved: bool,
}

/// Control type in `VRCExpressionsMenu`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ExpressionMenuItemType {
    #[default]
    Toggle,
    Button,
    RadialPuppet,
    SubMenu,
}

/// Item in `VRCExpressionsMenu`, supporting nested `SubMenu` trees.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpressionMenuItem {
    pub name: String,
    #[serde(rename = "type")]
    pub item_type: ExpressionMenuItemType,
    #[serde(default)]
    pub parameter: Option<String>,
    #[serde(default)]
    pub value: f32,
    #[serde(default)]
    pub sub_menu: Vec<ExpressionMenuItem>,
}

/// Animator controller layer extracted from the avatar FX controller.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FxLayerDef {
    pub name: String,
    #[serde(default = "default_one")]
    pub default_weight: f32,
    #[serde(default)]
    pub states: Vec<FxStateDef>,
    #[serde(default)]
    pub transitions: Vec<FxTransitionDef>,
}

/// Single state inside an FX Animator layer.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct FxStateDef {
    pub name: String,
    #[serde(default)]
    pub is_default: bool,
    #[serde(default)]
    pub curves: Vec<AnimationCurveDef>,
}

/// State transition in an FX layer (`from_state == None` denotes an `AnyState` transition).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FxTransitionDef {
    #[serde(default)]
    pub from_state: Option<String>,
    pub to_state: String,
    #[serde(default)]
    pub has_exit_time: bool,
    #[serde(default = "default_one")]
    pub exit_time: f32,
    #[serde(default = "default_transition_duration")]
    pub duration: f32,
    #[serde(default)]
    pub conditions: Vec<FxConditionDef>,
}

/// Comparison operator for an FX transition condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FxConditionMode {
    If,
    IfNot,
    Greater,
    Less,
    Equals,
    NotEqual,
}

/// Condition guarding an `FxTransitionDef`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FxConditionDef {
    pub parameter: String,
    pub mode: FxConditionMode,
    #[serde(default)]
    pub threshold: f32,
}

/// Target animated property in an `AnimationCurveDef`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurveProperty {
    BlendShape(String),
    GameObjectActive,
    MaterialFloat(String),
}

/// Extracted Unity animation curve with `(time_seconds, value)` keyframes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnimationCurveDef {
    pub node_path: String,
    pub property: CurveProperty,
    #[serde(default)]
    pub keyframes: Vec<(f32, f32)>,
}

/// Recognized VRChat toon shader family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ShaderFamily {
    #[serde(alias = "lilToon")]
    LilToon,
    #[serde(alias = "poiyomi")]
    Poiyomi,
    #[serde(alias = "uts", alias = "UTS")]
    Uts,
    #[default]
    #[serde(alias = "standard")]
    Standard,
}

/// Alpha blending / clipping mode for a material.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum AlphaMode {
    #[default]
    Opaque,
    Mask,
    Blend,
}

/// Face culling mode for a material.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CullMode {
    #[default]
    Back,
    Front,
    None,
}

/// Per-material VRChat shader metadata linked to a glTF material index.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialDef {
    pub gltf_material_index: usize,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub shader_family: ShaderFamily,
    #[serde(default = "default_render_queue")]
    pub render_queue: i32,
    #[serde(default)]
    pub alpha_mode: AlphaMode,
    #[serde(default)]
    pub params: MaterialParams,
}

/// Unified shader uniform parameters mapped from lilToon, Poiyomi, UTS, or Standard materials.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MaterialParams {
    #[serde(default = "default_rgba_white")]
    pub main_color: [f32; 4],
    #[serde(default = "default_shadow_color_1")]
    pub shadow_color_1: [f32; 4],
    #[serde(default = "default_shadow_color_2")]
    pub shadow_color_2: [f32; 4],
    #[serde(default = "default_shadow_border")]
    pub shadow_border: f32,
    #[serde(default = "default_shadow_blur")]
    pub shadow_blur: f32,
    #[serde(default = "default_shadow_2nd_border")]
    pub shadow_2nd_border: f32,
    #[serde(default = "default_shadow_blur")]
    pub shadow_2nd_blur: f32,
    pub rim_enable: bool,
    #[serde(default = "default_rgba_white")]
    pub rim_color: [f32; 4],
    #[serde(default = "default_rim_border")]
    pub rim_border: f32,
    #[serde(default = "default_shadow_blur")]
    pub rim_blur: f32,
    #[serde(default = "default_rim_fresnel_power")]
    pub rim_fresnel_power: f32,
    pub matcap_enable: bool,
    #[serde(default = "default_rgba_white")]
    pub matcap_color: [f32; 4],
    pub matcap_blend_mode: u32,
    pub matcap_texture_index: Option<usize>,
    pub emission_enable: bool,
    #[serde(default = "default_emission_color")]
    pub emission_color: [f32; 4],
    pub emission_texture_index: Option<usize>,
    pub outline_enable: bool,
    #[serde(default = "default_outline_width")]
    pub outline_width: f32,
    #[serde(default = "default_outline_color")]
    pub outline_color: [f32; 4],
    pub fur_enable: bool,
    #[serde(default = "default_fur_length")]
    pub fur_length: f32,
    #[serde(default = "default_fur_layers")]
    pub fur_layers: u32,
    #[serde(default = "default_fur_gravity")]
    pub fur_gravity: f32,
    pub fur_mask_texture_index: Option<usize>,
    pub audiolink_enable: bool,
    pub audiolink_band: u32,
    #[serde(default = "default_one")]
    pub audiolink_strength: f32,
    #[serde(default = "default_alpha_cutoff")]
    pub alpha_cutoff: f32,
    pub cull_mode: CullMode,
}

impl Default for MaterialParams {
    fn default() -> Self {
        Self {
            main_color: default_rgba_white(),
            shadow_color_1: default_shadow_color_1(),
            shadow_color_2: default_shadow_color_2(),
            shadow_border: default_shadow_border(),
            shadow_blur: default_shadow_blur(),
            shadow_2nd_border: default_shadow_2nd_border(),
            shadow_2nd_blur: default_shadow_blur(),
            rim_enable: false,
            rim_color: default_rgba_white(),
            rim_border: default_rim_border(),
            rim_blur: default_shadow_blur(),
            rim_fresnel_power: default_rim_fresnel_power(),
            matcap_enable: false,
            matcap_color: default_rgba_white(),
            matcap_blend_mode: 0,
            matcap_texture_index: None,
            emission_enable: false,
            emission_color: default_emission_color(),
            emission_texture_index: None,
            outline_enable: false,
            outline_width: default_outline_width(),
            outline_color: default_outline_color(),
            fur_enable: false,
            fur_length: default_fur_length(),
            fur_layers: default_fur_layers(),
            fur_gravity: default_fur_gravity(),
            fur_mask_texture_index: None,
            audiolink_enable: false,
            audiolink_band: 0,
            audiolink_strength: 1.0,
            alpha_cutoff: default_alpha_cutoff(),
            cull_mode: CullMode::Back,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn sample_manifest() -> MascotManifest {
        let mut humanoid_bones = BTreeMap::new();
        humanoid_bones.insert("Hips".to_owned(), "Armature/Hips".to_owned());
        humanoid_bones.insert("Head".to_owned(), "Armature/Hips/Spine/Head".to_owned());

        MascotManifest {
            schema_version: SCHEMA_VERSION.to_owned(),
            avatar_name: "SampleAvatar".to_owned(),
            glb_file: "avatar.glb".to_owned(),
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
            phys_bones: vec![PhysBoneDef {
                root_node: "Armature/Hips/Spine/Head/Hair_L".to_owned(),
                collider_refs: vec!["col_head".to_owned()],
                limit_type: PhysBoneLimitType::Angle,
                max_angle_x: 45.0,
                ..Default::default()
            }],
            colliders: vec![ColliderDef {
                id: "col_head".to_owned(),
                parent_node: "Armature/Hips/Spine/Head".to_owned(),
                shape: ColliderShape::Capsule,
                radius: 0.1,
                height: 0.25,
                position: [0.0, 0.08, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                inside_bounds: false,
            }],
            expressions: ExpressionsDef {
                parameters: vec![
                    ExpressionParameterDef {
                        name: "FaceEmotion".to_owned(),
                        param_type: ExpressionParamType::Int,
                        default: 0.0,
                        saved: true,
                    },
                    ExpressionParameterDef {
                        name: "JacketToggle".to_owned(),
                        param_type: ExpressionParamType::Bool,
                        default: 1.0,
                        saved: true,
                    },
                ],
                menu: vec![ExpressionMenuItem {
                    name: "Expressions".to_owned(),
                    item_type: ExpressionMenuItemType::SubMenu,
                    parameter: None,
                    value: 0.0,
                    sub_menu: vec![ExpressionMenuItem {
                        name: "Smile".to_owned(),
                        item_type: ExpressionMenuItemType::Toggle,
                        parameter: Some("FaceEmotion".to_owned()),
                        value: 1.0,
                        sub_menu: vec![],
                    }],
                }],
                fx_layers: vec![FxLayerDef {
                    name: "FaceLayer".to_owned(),
                    default_weight: 1.0,
                    states: vec![
                        FxStateDef {
                            name: "Neutral".to_owned(),
                            is_default: true,
                            curves: vec![],
                        },
                        FxStateDef {
                            name: "Smile".to_owned(),
                            is_default: false,
                            curves: vec![AnimationCurveDef {
                                node_path: "Body".to_owned(),
                                property: CurveProperty::BlendShape("eye_smile".to_owned()),
                                keyframes: vec![(0.0, 0.0), (0.2, 100.0)],
                            }],
                        },
                    ],
                    transitions: vec![FxTransitionDef {
                        from_state: None,
                        to_state: "Smile".to_owned(),
                        has_exit_time: false,
                        exit_time: 1.0,
                        duration: 0.1,
                        conditions: vec![FxConditionDef {
                            parameter: "FaceEmotion".to_owned(),
                            mode: FxConditionMode::Equals,
                            threshold: 1.0,
                        }],
                    }],
                }],
            },
            materials: vec![MaterialDef {
                gltf_material_index: 0,
                name: "Body_Mat".to_owned(),
                shader_family: ShaderFamily::LilToon,
                render_queue: 2000,
                alpha_mode: AlphaMode::Opaque,
                params: MaterialParams {
                    rim_enable: true,
                    outline_enable: true,
                    ..Default::default()
                },
            }],
        }
    }

    #[test]
    fn manifest_round_trip_file_and_json() {
        let dir = tempdir().expect("create tempdir");
        let path = dir.path().join("nested").join("mascot_manifest.json");
        let original = sample_manifest();
        original.save(&path).expect("save manifest");
        let loaded = MascotManifest::load(&path).expect("load manifest");
        assert_eq!(original, loaded);
        assert_eq!(
            loaded
                .viseme_map
                .get(VisemeKey::Aa)
                .map(|b| b.blendshape.as_str()),
            Some("vrc.v_aa")
        );
    }

    #[test]
    fn manifest_validation_rejects_bad_references_and_versions() {
        let mut m = sample_manifest();
        m.schema_version = "2.0.0".to_owned();
        assert!(matches!(
            m.validate(),
            Err(SchemaError::UnsupportedSchemaVersion { .. })
        ));

        let mut m = sample_manifest();
        m.phys_bones[0].collider_refs = vec!["missing_col".to_owned()];
        assert!(matches!(m.validate(), Err(SchemaError::Validation(_))));

        let mut m = sample_manifest();
        m.expressions.fx_layers[0].transitions[0].to_state = "MissingState".to_owned();
        assert!(matches!(m.validate(), Err(SchemaError::Validation(_))));
    }

    #[test]
    fn partial_json_deserializes_with_defaults() {
        let json = r#"{
            "schema_version": "1.0.0",
            "avatar_name": "Minimal",
            "glb_file": "avatar.glb",
            "materials": [
                {
                    "gltf_material_index": 0,
                    "shader_family": "lilToon"
                }
            ]
        }"#;
        let m = MascotManifest::from_json_str(json).expect("parse minimal json");
        assert_eq!(m.materials[0].shader_family, ShaderFamily::LilToon);
        assert_eq!(m.materials[0].render_queue, 2000);
        assert_eq!(m.materials[0].params.alpha_cutoff, 0.5);
    }
}
