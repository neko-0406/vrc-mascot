use thiserror::Error;

/// Errors returned by schema loading, saving, validation, configuration, and registry operations.
#[derive(Debug, Error)]
pub enum SchemaError {
    /// Underlying filesystem I/O failure.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization or deserialization failure.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// TOML deserialization failure.
    #[error("TOML parse error: {0}")]
    TomlDeserialize(#[from] toml::de::Error),

    /// TOML serialization failure.
    #[error("TOML serialize error: {0}")]
    TomlSerialize(#[from] toml::ser::Error),

    /// Manifest or configuration semantic validation error.
    #[error("Validation error: {0}")]
    Validation(String),

    /// Incompatible `schema_version` found in a manifest file.
    #[error(
        "Unsupported schema version: expected major version matching {expected}, found {found}"
    )]
    UnsupportedSchemaVersion { expected: String, found: String },

    /// Failed to resolve a standard user directory (e.g. `%APPDATA%`).
    #[error("Could not determine directory: {0}")]
    DirectoryUnavailable(String),
}
