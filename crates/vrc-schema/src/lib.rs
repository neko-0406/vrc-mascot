//! Shared intermediate schema, configuration, registry, and runtime types for `vrc-mascot`.

pub mod config;
pub mod error;
pub mod manifest;
pub mod registry;
pub mod runtime;

pub use config::{
    config_path, resolve_extract_output_dir, AiConfig, AppConfig, AvatarImportConfig,
    DisplayConfig, MotionConfig, PhysicsShaderConfig, QualityPreset, VoiceConfig,
};
pub use error::SchemaError;
pub use manifest::{
    AlphaMode, AnimationCurveDef, BlendShapeBinding, ColliderDef, ColliderShape, CullMode,
    CurveProperty, ExpressionMenuItem, ExpressionMenuItemType, ExpressionParamType,
    ExpressionParameterDef, ExpressionsDef, FxConditionDef, FxConditionMode, FxLayerDef,
    FxStateDef, FxTransitionDef, MascotManifest, MaterialDef, MaterialParams, PhysBoneDef,
    PhysBoneLimitType, ShaderFamily, VisemeMap, SCHEMA_VERSION,
};
pub use registry::{AvatarEntry, AvatarRegistry, MANIFEST_FILE_NAME};
pub use runtime::{
    AudioLinkData, ChatMessage, ChatRole, DecisionAction, DecisionContext, DecisionLogEntry,
    DecisionOutput, EmotionTag, ImportJobStatus, MascotState, VisemeKey, VisemeWeights,
};
