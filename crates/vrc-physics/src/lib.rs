//! `VRCPhysBone` Verlet/PBD physics, autonomous mascot motion, and FX expression runtime.

pub mod expressions;
pub mod motion;
pub mod physbone;
pub mod skeleton;

use std::collections::HashMap;

use glam::{Mat4, Quat, Vec2, Vec3};
use vrc_schema::{
    ColliderDef, EmotionTag, MascotManifest, MascotState, MotionConfig, PhysBoneDef,
    PhysicsShaderConfig, VisemeKey, VisemeMap, VisemeWeights,
};

pub use expressions::{FxOutput, FxStateMachine, ParamValue};
pub use motion::{AutonomousMotion, MotionFrameOutput, MotionInput};
pub use physbone::{PhysBoneGrab, PhysBoneWorld, FIXED_DT};
pub use skeleton::{SkeletonNode, SkeletonPose};

/// Legacy minimal physics world kept for compatibility with early skeleton linkage.
#[derive(Debug, Default)]
pub struct PhysicsWorld {
    pub phys_bones: Vec<PhysBoneDef>,
    pub colliders: Vec<ColliderDef>,
    pub gravity: Vec3,
}

impl PhysicsWorld {
    /// Creates a `PhysicsWorld` with standard downward gravity.
    pub fn new(phys_bones: Vec<PhysBoneDef>, colliders: Vec<ColliderDef>) -> Self {
        Self {
            phys_bones,
            colliders,
            gravity: Vec3::new(0.0, -9.81, 0.0),
        }
    }
}

/// Per-frame external inputs driving [`MascotSimulation::step`].
#[derive(Debug, Clone, Default)]
pub struct SimInput {
    /// Inertial acceleration (m/s^2) induced by desktop window dragging or landing impact.
    pub window_accel: Vec3,
    /// Current mascot window state (`Idle`, `Dragged`, `Falling`, `SeatedOnTaskbar`, `SeatedOnWindow`).
    pub mascot_state: MascotState,
    /// Mouse cursor position normalized to `[-1, 1]` around the avatar viewport for gaze tracking.
    pub cursor_ndc: Option<Vec2>,
    /// Projected 2D screen-space position of the avatar's `Head` node (pixels).
    pub head_screen_pos: Option<Vec2>,
    /// Current 2D screen-space cursor position (pixels) for petting detection.
    pub cursor_screen_pos: Option<Vec2>,
    /// Active lip-sync vowel weights from VOICEVOX playback.
    pub viseme_weights: VisemeWeights,
}

/// Evaluated simulation frame outputs passed to `vrc-renderer` and `AppState`.
#[derive(Debug, Clone, Default)]
pub struct SimOutput {
    pub local_transforms: Vec<(Vec3, Quat, Vec3)>,
    pub world_matrices: Vec<Mat4>,
    pub blendshape_weights: HashMap<String, f32>,
    pub node_visibility: HashMap<String, bool>,
    pub material_floats: HashMap<(String, String), f32>,
    pub petting_triggered: bool,
    pub pet_count_recent: u32,
}

/// Unified desktop mascot simulation facade combining skeleton pose, PhysBone PBD,
/// autonomous motion, petting detection, and VRChat FX expression state machine.
#[derive(Debug, Clone)]
pub struct MascotSimulation {
    pose: SkeletonPose,
    phys_bones: PhysBoneWorld,
    motion: AutonomousMotion,
    expressions: FxStateMachine,
    viseme_map: VisemeMap,
}

impl MascotSimulation {
    /// Constructs a new [`MascotSimulation`] from an avatar manifest and rest skeleton nodes.
    pub fn new(manifest: &MascotManifest, rest_nodes: Vec<SkeletonNode>) -> Self {
        let pose = SkeletonPose::from_nodes(rest_nodes);
        let phys_bones = PhysBoneWorld::new(&manifest.phys_bones, &manifest.colliders, &pose);
        let motion = AutonomousMotion::new(&manifest.humanoid_bones, &pose);
        let expressions = FxStateMachine::new(&manifest.expressions);

        Self {
            pose,
            phys_bones,
            motion,
            expressions,
            viseme_map: manifest.viseme_map.clone(),
        }
    }

    /// Advances all mascot subsystems by `dt` seconds and returns updated transforms and BlendShapes.
    pub fn step(
        &mut self,
        dt: f32,
        input: &SimInput,
        motion_cfg: &MotionConfig,
        phys_cfg: &PhysicsShaderConfig,
    ) -> SimOutput {
        // 1. Start from clean rest pose each frame so procedural offsets never drift.
        self.pose.reset_to_rest();

        // 2. Evaluate autonomous idle breathing, postures, gaze tracking, blinking, and petting.
        let motion_input = MotionInput {
            mascot_state: input.mascot_state,
            cursor_ndc: input.cursor_ndc,
            head_screen_pos: input.head_screen_pos,
            cursor_screen_pos: input.cursor_screen_pos,
        };
        let motion_out = self
            .motion
            .step(dt, &mut self.pose, &motion_input, motion_cfg);

        // 3. Propagate animated parent bone transforms before PhysBone & collider evaluation.
        self.pose.update_world_transforms();

        // 4. Step Verlet/PBD PhysBone chains and write resulting rotations onto the skeleton.
        self.phys_bones
            .step(dt, &mut self.pose, input.window_accel, phys_cfg);

        // 5. Step FX expression state machine and evaluate BlendShape / Active / Material curves.
        let fx_out = self.expressions.step(dt);

        // 6. Merge BlendShape weights: FX expressions + autonomous blink + VOICEVOX lip-sync visemes.
        let mut blendshape_weights = fx_out.blendshape_weights;
        for (name, weight) in motion_out.blendshape_weights {
            blendshape_weights
                .entry(name)
                .and_modify(|existing| *existing = existing.max(weight))
                .or_insert(weight);
        }

        let visemes = input.viseme_weights.clamped();
        for key in [
            VisemeKey::Aa,
            VisemeKey::Ih,
            VisemeKey::Ou,
            VisemeKey::Ee,
            VisemeKey::Oh,
        ] {
            if let Some(binding) = self.viseme_map.get(key) {
                let w = visemes.get(key);
                blendshape_weights
                    .entry(binding.blendshape.clone())
                    .and_modify(|existing| *existing = existing.max(w))
                    .or_insert(w);
            }
        }

        SimOutput {
            local_transforms: self.pose.local_transforms().to_vec(),
            world_matrices: self.pose.world_matrices().to_vec(),
            blendshape_weights,
            node_visibility: fx_out.node_visibility,
            material_floats: fx_out.material_floats,
            petting_triggered: motion_out.petting_triggered,
            pet_count_recent: motion_out.pet_count_recent,
        }
    }

    /// Sets an FX expression float parameter.
    pub fn set_param_float(&mut self, name: &str, value: f32) {
        self.expressions.set_float(name, value);
    }

    /// Sets an FX expression bool parameter.
    pub fn set_param_bool(&mut self, name: &str, value: bool) {
        self.expressions.set_bool(name, value);
    }

    /// Sets an FX expression integer parameter.
    pub fn set_param_int(&mut self, name: &str, value: i32) {
        self.expressions.set_int(name, value);
    }

    /// Reads an FX expression parameter value.
    pub fn get_param(&self, name: &str) -> Option<ParamValue> {
        self.expressions.get_param(name)
    }

    /// Triggers an expression menu control (`Toggle`, `Button`, `RadialPuppet`) by name.
    pub fn trigger_menu_item(&mut self, item_name: &str) -> bool {
        self.expressions.trigger_menu_item(item_name)
    }

    /// Drives expression parameters or fallback BlendShapes from an AI [`EmotionTag`].
    pub fn set_emotion(&mut self, emotion: EmotionTag) {
        self.expressions.set_emotion(&emotion);
    }

    /// Attempts to grab the nearest grabbable PhysBone particle within `max_dist`.
    pub fn grab_physbone_near(&mut self, world_pos: Vec3, max_dist: f32) -> Option<(usize, usize)> {
        self.phys_bones.grab_near(world_pos, max_dist)
    }

    /// Moves the currently grabbed PhysBone target to `target_world_pos`.
    pub fn update_grab(&mut self, target_world_pos: Vec3) {
        self.phys_bones.update_grab(target_world_pos);
    }

    /// Releases the currently grabbed PhysBone particle.
    pub fn release_grab(&mut self) {
        self.phys_bones.release_grab();
    }

    /// Injects an instantaneous velocity impulse (m/s) into all PhysBone chains.
    pub fn inject_impulse(&mut self, impulse: Vec3) {
        self.phys_bones.inject_impulse(impulse);
    }

    /// Returns the number of petting triggers recorded in the last 60 seconds.
    pub fn pet_count_recent(&self) -> u32 {
        self.motion.pet_count_recent()
    }

    /// Resets skeleton pose, PhysBone particle positions, and transient motion state to rest.
    pub fn reset_pose(&mut self) {
        self.pose.reset_to_rest();
        self.phys_bones.reset(&self.pose);
        self.motion.reset();
    }

    /// Returns immutable access to the FX expression state machine.
    pub fn expressions_state(&self) -> &FxStateMachine {
        &self.expressions
    }

    /// Returns immutable access to the current skeleton pose.
    pub fn skeleton_pose(&self) -> &SkeletonPose {
        &self.pose
    }

    /// Returns immutable access to the PhysBone simulation world.
    pub fn phys_bones(&self) -> &PhysBoneWorld {
        &self.phys_bones
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use vrc_schema::{
        AnimationCurveDef, BlendShapeBinding, CurveProperty, ExpressionMenuItem,
        ExpressionMenuItemType, ExpressionParamType, ExpressionParameterDef, ExpressionsDef,
        FxConditionDef, FxConditionMode, FxLayerDef, FxStateDef, FxTransitionDef, SCHEMA_VERSION,
    };

    fn test_manifest_and_nodes() -> (MascotManifest, Vec<SkeletonNode>) {
        let nodes = vec![
            SkeletonNode::new(
                "Hips",
                None,
                vec![1, 6],
                Vec3::new(0.0, 0.8, 0.0),
                Quat::IDENTITY,
                Vec3::ONE,
            ),
            SkeletonNode::new(
                "Spine",
                Some(0),
                vec![2],
                Vec3::new(0.0, 0.2, 0.0),
                Quat::IDENTITY,
                Vec3::ONE,
            ),
            SkeletonNode::new(
                "Head",
                Some(1),
                vec![3],
                Vec3::new(0.0, 0.3, 0.0),
                Quat::IDENTITY,
                Vec3::ONE,
            ),
            SkeletonNode::new(
                "Hair_0",
                Some(2),
                vec![4],
                Vec3::new(0.0, 0.1, 0.0),
                Quat::IDENTITY,
                Vec3::ONE,
            ),
            SkeletonNode::new(
                "Hair_1",
                Some(3),
                vec![5],
                Vec3::new(0.0, -0.25, 0.0),
                Quat::IDENTITY,
                Vec3::ONE,
            ),
            SkeletonNode::new(
                "Hair_2",
                Some(4),
                vec![],
                Vec3::new(0.0, -0.25, 0.0),
                Quat::IDENTITY,
                Vec3::ONE,
            ),
            SkeletonNode::new(
                "LeftUpperLeg",
                Some(0),
                vec![7],
                Vec3::new(-0.1, -0.05, 0.0),
                Quat::IDENTITY,
                Vec3::ONE,
            ),
            SkeletonNode::new(
                "LeftLowerLeg",
                Some(6),
                vec![],
                Vec3::new(0.0, -0.4, 0.0),
                Quat::IDENTITY,
                Vec3::ONE,
            ),
        ];

        let mut humanoid_bones = BTreeMap::new();
        humanoid_bones.insert("Hips".to_owned(), "Hips".to_owned());
        humanoid_bones.insert("Spine".to_owned(), "Spine".to_owned());
        humanoid_bones.insert("Head".to_owned(), "Head".to_owned());
        humanoid_bones.insert("LeftUpperLeg".to_owned(), "LeftUpperLeg".to_owned());
        humanoid_bones.insert("LeftLowerLeg".to_owned(), "LeftLowerLeg".to_owned());

        let manifest = MascotManifest {
            schema_version: SCHEMA_VERSION.to_owned(),
            avatar_name: "TestAvatar".to_owned(),
            glb_file: "avatar.glb".to_owned(),
            humanoid_bones,
            viseme_map: VisemeMap {
                aa: Some(BlendShapeBinding {
                    mesh: "Body".to_owned(),
                    blendshape: "vrc.v_aa".to_owned(),
                }),
                ..Default::default()
            },
            phys_bones: vec![PhysBoneDef {
                root_node: "Hair_0".to_owned(),
                pull: 0.25,
                spring: 0.4,
                gravity: 0.2,
                ..Default::default()
            }],
            colliders: vec![],
            expressions: ExpressionsDef {
                parameters: vec![ExpressionParameterDef {
                    name: "FaceEmotion".to_owned(),
                    param_type: ExpressionParamType::Int,
                    default: 0.0,
                    saved: true,
                }],
                menu: vec![ExpressionMenuItem {
                    name: "Smile".to_owned(),
                    item_type: ExpressionMenuItemType::Toggle,
                    parameter: Some("FaceEmotion".to_owned()),
                    value: 1.0,
                    sub_menu: vec![],
                }],
                fx_layers: vec![FxLayerDef {
                    name: "Face".to_owned(),
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
                                property: CurveProperty::BlendShape("smile".to_owned()),
                                keyframes: vec![(0.0, 1.0)],
                            }],
                        },
                    ],
                    transitions: vec![FxTransitionDef {
                        from_state: Some("Neutral".to_owned()),
                        to_state: "Smile".to_owned(),
                        has_exit_time: false,
                        exit_time: 1.0,
                        duration: 0.0,
                        conditions: vec![FxConditionDef {
                            parameter: "FaceEmotion".to_owned(),
                            mode: FxConditionMode::Equals,
                            threshold: 1.0,
                        }],
                    }],
                }],
            },
            materials: vec![],
        };

        (manifest, nodes)
    }

    #[test]
    fn simulation_facade_integrates_motion_physbone_viseme_and_fx() {
        let (manifest, nodes) = test_manifest_and_nodes();
        let mut sim = MascotSimulation::new(&manifest, nodes);
        let motion_cfg = MotionConfig::default();
        let phys_cfg = PhysicsShaderConfig::default();

        sim.set_emotion(EmotionTag::Happy);
        let input = SimInput {
            window_accel: Vec3::new(5.0, 0.0, 0.0),
            mascot_state: MascotState::SeatedOnTaskbar,
            cursor_ndc: Some(Vec2::new(0.5, 0.2)),
            viseme_weights: VisemeWeights::from_key(VisemeKey::Aa, 0.75),
            ..Default::default()
        };

        let out = sim.step(1.0 / 60.0, &input, &motion_cfg, &phys_cfg);
        assert_eq!(out.local_transforms.len(), 8);
        assert_eq!(out.world_matrices.len(), 8);
        assert_eq!(out.blendshape_weights.get("smile").copied(), Some(1.0));
        assert_eq!(out.blendshape_weights.get("vrc.v_aa").copied(), Some(0.75));

        sim.reset_pose();
        assert_eq!(sim.pet_count_recent(), 0);
    }
}
