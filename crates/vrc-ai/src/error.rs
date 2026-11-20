//! Error types returned by AI decision, dialogue, VOICEVOX, and audio playback operations.

use thiserror::Error;

/// Unified error type for `vrc-ai`.
#[derive(Debug, Error)]
pub enum AiError {
    /// Underlying HTTP transport or client error.
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    /// Non-successful HTTP status code returned by a remote or local API.
    #[error("API returned HTTP {status}: {body}")]
    ApiStatus { status: u16, body: String },

    /// JSON serialization or deserialization failure.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Required API key was missing or empty.
    #[error("missing API key for {provider}")]
    MissingApiKey { provider: String },

    /// VOICEVOX Engine could not be reached or is offline.
    #[error("VOICEVOX Engine is offline at {url}: {reason}")]
    VoicevoxOffline { url: String, reason: String },

    /// Audio device initialization, decoding, or playback failure.
    #[error("audio output error: {0}")]
    AudioOutput(String),

    /// Unexpected or malformed payload returned by an external service.
    #[error("invalid response from {source_name}: {reason}")]
    InvalidResponse { source_name: String, reason: String },
}
