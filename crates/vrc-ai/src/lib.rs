//! Gemini Decisions API, Dialogue LLM, privacy filtering, VOICEVOX TTS, and lip-sync timing.

pub mod audio_player;
pub mod decisions;
pub mod dialogue;
pub mod error;
mod gemini;
pub mod lipsync;
pub mod privacy;
#[cfg(test)]
mod test_support;
pub mod voicevox;

pub use audio_player::SpeechPlayer;
pub use decisions::DecisionsClient;
pub use dialogue::{parse_dialogue_response, parse_emotion_tag, DialogueClient, DialogueResponse};
pub use error::AiError;
pub use lipsync::{vowel_to_viseme, LipSyncSegment, LipSyncTimeline};
pub use privacy::{categorize_process, is_excluded_app, sanitize_decision_context};
pub use voicevox::{
    AccentPhrase, AudioQuery, Mora, SynthesizedSpeech, VoicevoxClient, VoicevoxSpeaker,
    VoicevoxSpeakerStyle,
};

use vrc_schema::{AiConfig, ChatMessage, DecisionContext, DecisionOutput, VoiceConfig};

/// High-level service combining [`DecisionsClient`], [`DialogueClient`], and [`VoicevoxClient`]
/// around a shared HTTP client and configuration.
#[derive(Debug, Clone)]
pub struct AiService {
    pub ai_config: AiConfig,
    pub voice_config: VoiceConfig,
    client: reqwest::Client,
    decisions: DecisionsClient,
    dialogue: DialogueClient,
    voicevox: VoicevoxClient,
}

impl AiService {
    /// Builds a new [`AiService`] with a shared `reqwest::Client`.
    pub fn new(ai_config: AiConfig, voice_config: VoiceConfig) -> Result<Self, AiError> {
        let client = reqwest::Client::builder().build()?;
        Ok(Self::with_http_client(ai_config, voice_config, client))
    }

    /// Builds an [`AiService`] using an existing `reqwest::Client`.
    pub fn with_http_client(
        ai_config: AiConfig,
        voice_config: VoiceConfig,
        client: reqwest::Client,
    ) -> Self {
        Self {
            ai_config,
            voice_config,
            decisions: DecisionsClient::with_client(client.clone()),
            dialogue: DialogueClient::with_client(client.clone()),
            voicevox: VoicevoxClient::with_client(client.clone()),
            client,
        }
    }

    /// Overrides the dialogue Gemini endpoint (useful for local mock server testing).
    pub fn with_dialogue_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.dialogue.set_endpoint(endpoint);
        self
    }

    /// Returns the underlying HTTP client reference.
    pub fn http_client(&self) -> &reqwest::Client {
        &self.client
    }

    /// Evaluates an autonomous decision snapshot via [`DecisionsClient::decide`].
    pub async fn decide(&self, ctx: &DecisionContext) -> Result<DecisionOutput, AiError> {
        self.decisions.decide(ctx, &self.ai_config).await
    }

    /// Sends a user chat turn via [`DialogueClient::chat`].
    pub async fn chat(&mut self, user_input: &str) -> Result<DialogueResponse, AiError> {
        self.dialogue.chat(user_input, &self.ai_config).await
    }

    /// Generates an autonomous spoken line from a `speech_intent` via
    /// [`DialogueClient::speak_from_intent`].
    pub async fn speak_from_intent(
        &mut self,
        speech_intent: &str,
    ) -> Result<DialogueResponse, AiError> {
        self.dialogue
            .speak_from_intent(speech_intent, &self.ai_config)
            .await
    }

    /// Synthesizes WAV audio and its aligned [`LipSyncTimeline`] via
    /// [`VoicevoxClient::synthesize_with_timeline`].
    pub async fn synthesize(&self, text: &str) -> Result<SynthesizedSpeech, AiError> {
        self.voicevox
            .synthesize_with_timeline(text, &self.voice_config)
            .await
    }

    /// Checks VOICEVOX Engine availability and returns its version string.
    pub async fn check_voicevox(&self) -> Result<String, AiError> {
        self.voicevox
            .check_health(&self.voice_config.voicevox_url)
            .await
    }

    /// Lists available VOICEVOX speakers and styles.
    pub async fn list_speakers(&self) -> Result<Vec<VoicevoxSpeaker>, AiError> {
        self.voicevox
            .list_speakers(&self.voice_config.voicevox_url)
            .await
    }

    /// Returns the current dialogue history.
    pub fn history(&self) -> &[ChatMessage] {
        self.dialogue.history()
    }

    /// Clears the dialogue history.
    pub fn clear_history(&mut self) {
        self.dialogue.clear_history();
    }

    /// Returns a reference to the inner [`DialogueClient`].
    pub fn dialogue(&self) -> &DialogueClient {
        &self.dialogue
    }

    /// Returns a mutable reference to the inner [`DialogueClient`].
    pub fn dialogue_mut(&mut self) -> &mut DialogueClient {
        &mut self.dialogue
    }
}

/// Backwards-compatible coordinator handle wrapping [`AiService`].
#[derive(Debug, Clone)]
pub struct AiCoordinator {
    pub service: AiService,
}

impl AiCoordinator {
    /// Builds a new `AiCoordinator` with a shared `reqwest::Client`.
    pub fn new(ai_config: AiConfig, voice_config: VoiceConfig) -> anyhow::Result<Self> {
        let service = AiService::new(ai_config, voice_config)?;
        Ok(Self { service })
    }

    /// Returns the underlying HTTP client reference.
    pub fn http_client(&self) -> &reqwest::Client {
        self.service.http_client()
    }

    /// Placeholder context evaluation returning `Idle`.
    pub fn default_decision(_ctx: &DecisionContext) -> DecisionOutput {
        DecisionOutput::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{MockHttpServer, MockResponse};
    use serde_json::json;
    use vrc_schema::{DecisionAction, EmotionTag, VisemeKey};

    #[tokio::test]
    async fn ai_service_end_to_end_decide_speak_and_synthesize() {
        let decision_envelope = json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": r#"{"action":"SPEAK","target_expression":"Smile","speech_intent":"Encourage the user","reason":"idle"}"#
                    }]
                }
            }]
        })
        .to_string();

        let dialogue_envelope = json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": r#"{"reply_text":"お疲れさま！少し休憩しよう？","emotion":"happy"}"#
                    }]
                }
            }]
        })
        .to_string();

        let aq_json = json!({
            "accent_phrases": [{
                "moras": [
                    { "text": "オ", "vowel": "o", "vowel_length": 0.15, "pitch": 5.0 },
                    { "text": "ツ", "consonant": "ts", "consonant_length": 0.05, "vowel": "u", "vowel_length": 0.12, "pitch": 5.0 }
                ],
                "accent": 1
            }],
            "speedScale": 1.0,
            "pitchScale": 0.0,
            "intonationScale": 1.0,
            "volumeScale": 1.0,
            "prePhonemeLength": 0.05,
            "postPhonemeLength": 0.05
        })
        .to_string();

        let server = MockHttpServer::start(move |req| {
            if req.target.contains("gemini-2.5-flash-lite") {
                MockResponse::json(200, decision_envelope.clone())
            } else if req.target.contains("gemini-2.5-flash") {
                MockResponse::json(200, dialogue_envelope.clone())
            } else if req.target == "/version" {
                MockResponse::json(200, "\"0.15.0\"")
            } else if req.target == "/speakers" {
                MockResponse::json(
                    200,
                    r#"[{"name":"ずんだもん","styles":[{"name":"ノーマル","id":3}]}]"#,
                )
            } else if req.target.starts_with("/audio_query?") {
                MockResponse::json(200, aq_json.clone())
            } else if req.target.starts_with("/synthesis?") {
                MockResponse::bytes(200, "audio/wav", b"RIFF\x24\x00\x00\x00WAVEfmt ".to_vec())
            } else {
                MockResponse::json(404, "{}")
            }
        })
        .await;

        let ai_cfg = AiConfig {
            chat_api_key: "shared-key".to_owned(),
            decisions_endpoint: server.base_url(),
            ..Default::default()
        };

        let voice_cfg = VoiceConfig {
            voicevox_url: server.base_url(),
            speaker_id: 3,
            ..Default::default()
        };

        let mut service = AiService::new(ai_cfg, voice_cfg)
            .expect("build service")
            .with_dialogue_endpoint(server.base_url());

        let ctx = sanitize_decision_context(
            60.0,
            0,
            Some("code.exe"),
            Some("secret.rs"),
            "evening",
            EmotionTag::Neutral,
            &service.ai_config,
            &[],
        );
        assert_eq!(ctx.active_window_title, None);

        let decision = service.decide(&ctx).await.expect("decide");
        assert_eq!(decision.action, DecisionAction::Speak);

        let speech_line = service
            .speak_from_intent(decision.speech_intent.as_deref().unwrap_or("greet"))
            .await
            .expect("speak_from_intent");
        assert_eq!(speech_line.emotion, EmotionTag::Happy);

        assert_eq!(service.check_voicevox().await.expect("version"), "0.15.0");
        assert_eq!(service.list_speakers().await.expect("speakers").len(), 1);

        let synth = service
            .synthesize(&speech_line.reply_text)
            .await
            .expect("synthesize");
        assert_eq!(synth.timeline.current_viseme(0.10), VisemeKey::Oh);
        assert!(!synth.wav_bytes.is_empty());
    }
}
