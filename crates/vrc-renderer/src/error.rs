use thiserror::Error;

/// Errors returned by `vrc-renderer` model loading, shader compilation, pipeline setup, and AudioLink capture.
#[derive(Debug, Error)]
pub enum RendererError {
    /// Underlying filesystem I/O failure.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON parsing error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Schema validation or manifest loading error.
    #[error("Schema error: {0}")]
    Schema(#[from] vrc_schema::SchemaError),

    /// glTF / GLB decoding error.
    #[error("glTF load error: {0}")]
    Gltf(#[from] gltf::Error),

    /// Texture image decoding error.
    #[error("Image decode error: {0}")]
    Image(#[from] image::ImageError),

    /// Model geometry, skeleton, or manifest binding inconsistency.
    #[error("Invalid avatar model: {0}")]
    InvalidModel(String),

    /// Missing asset referenced by manifest or model.
    #[error("Missing asset: {0}")]
    MissingAsset(String),

    /// WGSL shader validation or module creation failure.
    #[error("Shader error: {0}")]
    ShaderCompile(String),

    /// Graphics pipeline creation failure.
    #[error("Pipeline error: {0}")]
    Pipeline(String),

    /// AudioLink capture or FFT initialization failure.
    #[error("AudioLink error: {0}")]
    Audio(String),
}
