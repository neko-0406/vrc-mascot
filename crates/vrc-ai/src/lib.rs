//! Gemini Decisions API, Dialogue LLM, and VOICEVOX Engine client + lip-sync timing.

use vrc_schema::{AiConfig, DecisionContext, DecisionOutput, VisemeWeights, VoiceConfig};

/// Coordinator handle for AI decision making, dialogue generation, and VOICEVOX TTS.
#[derive(Debug, Clone)]
pub struct AiCoordinator {
    pub ai_config: AiConfig,
    pub voice_config: VoiceConfig,
    client: reqwest::Client,
}

impl AiCoordinator {
    /// Builds a new `AiCoordinator` with a shared `reqwest::Client`.
    pub fn new(ai_config: AiConfig, voice_config: VoiceConfig) -> anyhow::Result<Self> {
        let client = reqwest::Client::builder().build()?;
        let _ = rodio::DeviceSinkBuilder::open_default_sink;
        Ok(Self {
            ai_config,
            voice_config,
            client,
        })
    }

    /// Returns the underlying HTTP client reference.
    pub fn http_client(&self) -> &reqwest::Client {
        &self.client
    }

    /// Placeholder context evaluation returning `Idle`.
    pub fn default_decision(_ctx: &DecisionContext) -> DecisionOutput {
        let _ = VisemeWeights::default();
        DecisionOutput::default()
    }
}
