//! Gemini Decisions API client for low-latency structured autonomous action selection.

use serde_json::{json, Value};
use vrc_schema::{AiConfig, DecisionContext, DecisionOutput};

use crate::error::AiError;
use crate::gemini::{build_generate_content_url, post_generate_content, strip_markdown_code_fence};

const DECISION_SYSTEM_INSTRUCTION: &str = "\
You are the autonomous behavior decision engine for a Windows 3D desktop mascot. \
Given a JSON environment snapshot (idle_seconds, pet_count_recent, active_window_category, \
optional active_window_title, time_of_day, current_mood), choose exactly one next action \
matching the JSON schema: SPEAK, MOVE_TO_TASKBAR, SIT_ON_WINDOW, CHANGE_EXPRESSION, or IDLE. \
Prefer IDLE when the user is actively working or nothing notable happened. \
When choosing SPEAK, provide a concise speech_intent. \
When choosing CHANGE_EXPRESSION, provide target_expression.";

/// Client for calling the Gemini Decisions `generateContent` endpoint.
#[derive(Debug, Clone)]
pub struct DecisionsClient {
    client: reqwest::Client,
}

impl Default for DecisionsClient {
    fn default() -> Self {
        Self::with_client(reqwest::Client::new())
    }
}

impl DecisionsClient {
    /// Creates a new `DecisionsClient` with a default `reqwest::Client`.
    pub fn new() -> Result<Self, AiError> {
        let client = reqwest::Client::builder().build()?;
        Ok(Self { client })
    }

    /// Creates a `DecisionsClient` reusing an existing `reqwest::Client`.
    pub fn with_client(client: reqwest::Client) -> Self {
        Self { client }
    }

    /// Evaluates `ctx` against the configured Gemini Decisions endpoint and returns a structured
    /// [`DecisionOutput`].
    ///
    /// If `ai_cfg.decisions_enabled` is `false`, returns `DecisionOutput::default()` (`IDLE`)
    /// without making any network call.
    pub async fn decide(
        &self,
        ctx: &DecisionContext,
        ai_cfg: &AiConfig,
    ) -> Result<DecisionOutput, AiError> {
        if !ai_cfg.decisions_enabled {
            return Ok(DecisionOutput::default());
        }

        let api_key = resolve_decisions_api_key(ai_cfg)?;
        let url = build_generate_content_url(
            &ai_cfg.decisions_endpoint,
            &ai_cfg.decisions_model,
            "gemini_decisions",
        )?;
        let payload = build_decision_request_payload(ctx)?;

        let raw_text =
            post_generate_content(&self.client, &url, api_key, &payload, "gemini_decisions")
                .await?;
        parse_decision_output(&raw_text)
    }
}

fn resolve_decisions_api_key(ai_cfg: &AiConfig) -> Result<&str, AiError> {
    let primary = ai_cfg.decisions_api_key.trim();
    if !primary.is_empty() {
        return Ok(primary);
    }
    let fallback = ai_cfg.chat_api_key.trim();
    if !fallback.is_empty() {
        return Ok(fallback);
    }
    Err(AiError::MissingApiKey {
        provider: "gemini_decisions".to_owned(),
    })
}

fn build_decision_request_payload(ctx: &DecisionContext) -> Result<Value, AiError> {
    let ctx_json = serde_json::to_string(ctx)?;
    Ok(json!({
        "systemInstruction": {
            "parts": [{ "text": DECISION_SYSTEM_INSTRUCTION }]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{ "text": ctx_json }]
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "action": {
                        "type": "STRING",
                        "enum": [
                            "SPEAK",
                            "MOVE_TO_TASKBAR",
                            "SIT_ON_WINDOW",
                            "CHANGE_EXPRESSION",
                            "IDLE"
                        ]
                    },
                    "target_expression": {
                        "type": "STRING",
                        "nullable": true
                    },
                    "speech_intent": {
                        "type": "STRING",
                        "nullable": true
                    },
                    "reason": {
                        "type": "STRING",
                        "nullable": true
                    }
                },
                "required": ["action"]
            }
        }
    }))
}

fn parse_decision_output(raw: &str) -> Result<DecisionOutput, AiError> {
    let cleaned = strip_markdown_code_fence(raw);
    let output: DecisionOutput = serde_json::from_str(cleaned)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{MockHttpServer, MockResponse};
    use vrc_schema::{DecisionAction, EmotionTag};

    #[tokio::test]
    async fn decisions_client_parses_all_five_actions_and_falls_back_to_chat_key() {
        let cases = [
            (
                DecisionAction::Speak,
                r#"{"action":"SPEAK","target_expression":"Smile","speech_intent":"労う","reason":"long idle"}"#,
            ),
            (
                DecisionAction::MoveToTaskbar,
                r#"{"action":"MOVE_TO_TASKBAR","target_expression":null,"speech_intent":null,"reason":"stretch"}"#,
            ),
            (
                DecisionAction::SitOnWindow,
                r#"{"action":"SIT_ON_WINDOW","target_expression":"Happy","speech_intent":null,"reason":"editor active"}"#,
            ),
            (
                DecisionAction::ChangeExpression,
                r#"{"action":"CHANGE_EXPRESSION","target_expression":"Sleepy","speech_intent":null,"reason":"late night"}"#,
            ),
            (
                DecisionAction::Idle,
                r#"{"action":"IDLE","target_expression":null,"speech_intent":null,"reason":"user busy"}"#,
            ),
        ];

        for (expected_action, inner_json) in cases {
            let envelope = json!({
                "candidates": [{
                    "content": {
                        "parts": [{ "text": inner_json }]
                    }
                }]
            })
            .to_string();

            let server =
                MockHttpServer::start(move |_| MockResponse::json(200, envelope.clone())).await;

            let cfg = AiConfig {
                decisions_endpoint: server.base_url(),
                decisions_model: "gemini-2.5-flash-lite".to_owned(),
                decisions_api_key: String::new(),
                chat_api_key: "fallback-chat-key".to_owned(),
                ..Default::default()
            };

            let ctx = DecisionContext {
                idle_seconds: 120.0,
                pet_count_recent: 1,
                active_window_category: "editor".to_owned(),
                active_window_title: None,
                time_of_day: "night".to_owned(),
                current_mood: EmotionTag::Neutral,
            };

            let client = DecisionsClient::new().expect("build decisions client");
            let out = client.decide(&ctx, &cfg).await.expect("decide succeeds");
            assert_eq!(out.action, expected_action);

            let reqs = server.requests();
            assert_eq!(reqs.len(), 1);
            assert_eq!(reqs[0].method, "POST");
            assert_eq!(
                reqs[0].target,
                "/v1beta/models/gemini-2.5-flash-lite:generateContent"
            );
            assert_eq!(
                reqs[0].headers.get("x-goog-api-key").map(String::as_str),
                Some("fallback-chat-key")
            );

            let sent: Value = serde_json::from_str(&reqs[0].body).expect("valid request JSON");
            assert_eq!(
                sent["generationConfig"]["responseMimeType"],
                "application/json"
            );
            let enum_vals = sent["generationConfig"]["responseSchema"]["properties"]["action"]
                ["enum"]
                .as_array()
                .expect("enum array");
            assert_eq!(enum_vals.len(), 5);
        }
    }

    #[tokio::test]
    async fn decisions_client_errors_when_no_api_key_and_skips_when_disabled() {
        let client = DecisionsClient::new().expect("build client");
        let ctx = DecisionContext::default();

        let mut cfg = AiConfig::default();
        cfg.decisions_api_key.clear();
        cfg.chat_api_key.clear();
        let err = client.decide(&ctx, &cfg).await.expect_err("missing key");
        assert!(matches!(err, AiError::MissingApiKey { .. }));

        cfg.decisions_enabled = false;
        let out = client
            .decide(&ctx, &cfg)
            .await
            .expect("disabled returns idle");
        assert_eq!(out.action, DecisionAction::Idle);
    }
}
