use std::path::PathBuf;

use thiserror::Error;

/// Errors returned by Unity project indexing, parsing, Modular Avatar resolution, FBX loading, and export.
#[derive(Debug, Error)]
pub enum ExtractorError {
    /// Underlying filesystem I/O failure.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Schema validation or serialization error from `vrc-schema`.
    #[error("Schema error: {0}")]
    Schema(#[from] vrc_schema::SchemaError),

    /// JSON serialization failure.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Invalid options or caller input.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Unity YAML parsing error.
    #[error("YAML parse error in {path}: {message}")]
    YamlParse { path: PathBuf, message: String },

    /// Referenced Unity asset GUID could not be resolved.
    #[error("Asset not found for GUID '{guid}' ({context})")]
    AssetNotFound { guid: String, context: String },

    /// Requested avatar GameObject was not found in the target scene.
    #[error("Avatar '{avatar_name}' not found in scene '{scene_path}'")]
    AvatarNotFound {
        avatar_name: String,
        scene_path: PathBuf,
    },

    /// PrefabInstance or Prefab Variant resolution failure.
    #[error("Prefab resolution error: {message}")]
    PrefabResolution { message: String },

    /// FBX loading or geometry extraction error.
    #[error("FBX load error in {path}: {message}")]
    FbxLoad { path: PathBuf, message: String },

    /// Image decoding, resizing, or encoding error.
    #[error("Image error for {path}: {message}")]
    ImageError { path: PathBuf, message: String },

    /// glTF 2.0 / GLB 2.0 export error.
    #[error("GLB export error: {0}")]
    GltfExport(String),
}
