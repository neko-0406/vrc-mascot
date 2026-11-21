//! Multi-provider dialogue LLM client (Gemini & OpenAI-compatible) with bounded history
//! and structured/fallback emotion parsing.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use vrc_schema::{AiConfig, ChatMessage, ChatRole, EmotionTag};

use crate::error::AiError;
use crate::gemini::{
    build_generate_content_url, post_generate_content, strip_markdown_code_fence,
    DEFAULT_GEMINI_ENDPOINT,
};
use crate::openai::{
    build_chat_completions_url, build_openai_dialogue_payload, post_chat_completions,
};

const DIALOGUE_FORMAT_DIRECTIVE: &str = "\
Respond ONLY with a JSON object matching {\"reply_text\": string, \"emotion\": \"neutral|happy|sad|angry|surprised|shy|sleepy\"}.";

/// Structured dialogue output returned by [`DialogueClient`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DialogueResponse {
    pub reply_text: String,
    pub emotion: EmotionTag,
}

#[derive(Debug, Deserialize)]
struct RawDialogueJson {
    #[serde(default)]
    reply_text: Option<String>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    reply: Option<String>,
    #[serde(default)]
    emotion: Option<String>,
}

/// Stateful dialogue chat client maintaining bounded conversation turns.
#[derive(Debug, Clone)]
pub struct DialogueClient {
    client: reqwest::Client,
    endpoint: String,
    endpoint_override: Option<String>,
    history: Vec<ChatMessage>,
}

impl Default for DialogueClient {
    fn default() -> Self {
        Self::with_client(reqwest::Client::new())
    }
}

impl DialogueClient {
    /// Creates a new `DialogueClient` pointing at the official Gemini API endpoint.
    pub fn new() -> Result<Self, AiError> {
        let client = reqwest::Client::builder().build()?;
        Ok(Self::with_client(client))
    }

    /// Creates a `DialogueClient` reusing an existing `reqwest::Client`.
    pub fn with_client(client: reqwest::Client) -> Self {
        Self {
            client,
            endpoint: DEFAULT_GEMINI_ENDPOINT.to_owned(),
            endpoint_override: None,
            history: Vec::new(),
        }
    }

    /// Overrides the dialogue API base URL (for example, to point at a local mock server in tests).
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        let ep = endpoint.into();
        self.endpoint = ep.clone();
        self.endpoint_override = Some(ep);
        self
    }

    /// Sets the dialogue API base URL override in place.
    pub fn set_endpoint(&mut self, endpoint: impl Into<String>) {
        let ep = endpoint.into();
        self.endpoint = ep.clone();
        self.endpoint_override = Some(ep);
    }

    /// Returns the currently configured base URL.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Returns the stored conversation history messages.
    pub fn history(&self) -> &[ChatMessage] {
        &self.history
    }

    /// Clears all stored conversation history.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Sends a user message to the dialogue LLM, records the turn, and prunes old history
    /// according to `ai_cfg.max_history_turns`.
    pub async fn chat(
        &mut self,
        user_input: &str,
        ai_cfg: &AiConfig,
    ) -> Result<DialogueResponse, AiError> {
        let user_text = user_input.trim();
        if user_text.is_empty() {
            return Err(AiError::InvalidResponse {
                source_name: "gemini_dialogue".to_owned(),
                reason: "user input must not be empty".to_owned(),
            });
        }

        let response = self.send_turn(user_text, ai_cfg).await?;

        let now_ms = current_epoch_ms();
        self.history.push(ChatMessage {
            role: ChatRole::User,
            content: user_text.to_owned(),
            emotion: None,
            timestamp_epoch_ms: now_ms,
        });
        self.history.push(ChatMessage {
            role: ChatRole::Assistant,
            content: response.reply_text.clone(),
            emotion: Some(response.emotion.clone()),
            timestamp_epoch_ms: now_ms,
        });
        self.prune_history(ai_cfg.max_history_turns);

        Ok(response)
    }

    /// Generates an autonomous spoken line from a `speech_intent` produced by the decision model,
    /// recording the assistant utterance in history.
    pub async fn speak_from_intent(
        &mut self,
        speech_intent: &str,
        ai_cfg: &AiConfig,
    ) -> Result<DialogueResponse, AiError> {
        let prompt = format!(
            "[Autonomous Speech Intent] Speak naturally to the user based on this intent: {}",
            speech_intent.trim()
        );
        let response = self.send_turn(&prompt, ai_cfg).await?;

        self.history.push(ChatMessage {
            role: ChatRole::Assistant,
            content: response.reply_text.clone(),
            emotion: Some(response.emotion.clone()),
            timestamp_epoch_ms: current_epoch_ms(),
        });
        self.prune_history(ai_cfg.max_history_turns);

        Ok(response)
    }

    async fn send_turn(
        &mut self,
        current_user_prompt: &str,
        ai_cfg: &AiConfig,
    ) -> Result<DialogueResponse, AiError> {
        self.prune_history(ai_cfg.max_history_turns);

        let provider = ai_cfg.chat_provider.trim().to_ascii_lowercase();
        let configured_endpoint = ai_cfg.chat_endpoint.trim();
        let effective_endpoint = self
            .endpoint_override
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .or_else(|| (!configured_endpoint.is_empty()).then_some(configured_endpoint))
            .unwrap_or(self.endpoint.as_str());

        match provider.as_str() {
            "openai_compatible" | "openai" => {
                let url = build_chat_completions_url(effective_endpoint, "openai_dialogue")?;
                let payload = build_openai_dialogue_payload(
                    &self.history,
                    current_user_prompt,
                    ai_cfg,
                    "openai_dialogue",
                )?;
                let raw_text = post_chat_completions(
                    &self.client,
                    &url,
                    ai_cfg.chat_api_key.trim(),
                    &payload,
                    "openai_dialogue",
                )
                .await?;
                parse_dialogue_response(&raw_text)
            }
            "gemini" | "" => {
                let api_key = ai_cfg.chat_api_key.trim();
                if api_key.is_empty() {
                    return Err(AiError::MissingApiKey {
                        provider: if ai_cfg.chat_provider.trim().is_empty() {
                            "gemini".to_owned()
                        } else {
                            ai_cfg.chat_provider.clone()
                        },
                    });
                }

                let url = build_generate_content_url(
                    effective_endpoint,
                    &ai_cfg.chat_model,
                    "gemini_dialogue",
                )?;
                let payload = self.build_request_payload(current_user_prompt, ai_cfg);

                let raw_text =
                    post_generate_content(&self.client, &url, api_key, &payload, "gemini_dialogue")
                        .await?;
                parse_dialogue_response(&raw_text)
            }
            other => Err(AiError::InvalidResponse {
                source_name: "dialogue".to_owned(),
                reason: format!("unsupported chat_provider '{other}'"),
            }),
        }
    }

    fn build_request_payload(&self, current_user_prompt: &str, ai_cfg: &AiConfig) -> Value {
        let system_text = if ai_cfg.system_prompt.trim().is_empty() {
            DIALOGUE_FORMAT_DIRECTIVE.to_owned()
        } else {
            format!(
                "{}\n\n{}",
                ai_cfg.system_prompt.trim(),
                DIALOGUE_FORMAT_DIRECTIVE
            )
        };

        let mut contents = Vec::with_capacity(self.history.len() + 2);
        if ai_cfg.max_history_turns > 0 {
            for msg in &self.history {
                let role = match msg.role {
                    ChatRole::User | ChatRole::System => "user",
                    ChatRole::Assistant => "model",
                };
                if contents.is_empty() && role == "model" {
                    contents.push(json!({
                        "role": "user",
                        "parts": [{ "text": "(previous mascot utterance)" }]
                    }));
                }
                contents.push(json!({
                    "role": role,
                    "parts": [{ "text": msg.content }]
                }));
            }
        }

        contents.push(json!({
            "role": "user",
            "parts": [{ "text": current_user_prompt }]
        }));

        json!({
            "systemInstruction": {
                "parts": [{ "text": system_text }]
            },
            "contents": contents,
            "generationConfig": {
                "temperature": 0.7,
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "reply_text": { "type": "STRING" },
                        "emotion": {
                            "type": "STRING",
                            "enum": [
                                "neutral",
                                "happy",
                                "sad",
                                "angry",
                                "surprised",
                                "shy",
                                "sleepy"
                            ]
                        }
                    },
                    "required": ["reply_text", "emotion"]
                }
            }
        })
    }

    fn prune_history(&mut self, max_history_turns: usize) {
        let max_messages = max_history_turns.saturating_mul(2);
        if self.history.len() > max_messages {
            let excess = self.history.len() - max_messages;
            self.history.drain(..excess);
        }
    }
}

/// Parses a model response into [`DialogueResponse`], supporting structured JSON as well as
/// `[emotion] text` prefixes and plain-text fallbacks.
pub fn parse_dialogue_response(raw: &str) -> Result<DialogueResponse, AiError> {
    let cleaned = strip_markdown_code_fence(raw);

    if let Ok(parsed) = serde_json::from_str::<RawDialogueJson>(cleaned) {
        let text = parsed
            .reply_text
            .or(parsed.text)
            .or(parsed.reply)
            .unwrap_or_default();
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            let emotion = parsed
                .emotion
                .as_deref()
                .map(parse_emotion_tag)
                .unwrap_or(EmotionTag::Neutral);
            return Ok(DialogueResponse {
                reply_text: trimmed.to_owned(),
                emotion,
            });
        }
    }

    // Fallback: `[emotion] reply text` or plain text.
    let trimmed = cleaned.trim();
    if trimmed.is_empty() {
        return Err(AiError::InvalidResponse {
            source_name: "gemini_dialogue".to_owned(),
            reason: "empty dialogue response".to_owned(),
        });
    }

    if let Some(rest) = trimmed.strip_prefix('[') {
        if let Some(close_idx) = rest.find(']') {
            let tag_str = &rest[..close_idx];
            if !tag_str.contains('\n') && tag_str.len() <= 32 {
                let reply = rest[close_idx + 1..].trim();
                if !reply.is_empty() {
                    return Ok(DialogueResponse {
                        reply_text: reply.to_owned(),
                        emotion: parse_emotion_tag(tag_str),
                    });
                }
            }
        }
    }

    Ok(DialogueResponse {
        reply_text: trimmed.to_owned(),
        emotion: EmotionTag::Neutral,
    })
}

/// Maps an emotion string (case-insensitive) to [`EmotionTag`].
pub fn parse_emotion_tag(raw: &str) -> EmotionTag {
    let trimmed = raw.trim();
    match trimmed.to_ascii_lowercase().as_str() {
        "" | "neutral" | "none" => EmotionTag::Neutral,
        "happy" | "joy" | "smile" => EmotionTag::Happy,
        "sad" | "sorrow" => EmotionTag::Sad,
        "angry" | "anger" => EmotionTag::Angry,
        "surprised" | "surprise" => EmotionTag::Surprised,
        "shy" | "embarrassed" => EmotionTag::Shy,
        "sleepy" | "tired" => EmotionTag::Sleepy,
        _ => EmotionTag::Custom(trimmed.to_owned()),
    }
}

fn current_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{MockHttpServer, MockResponse};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn parses_json_fenced_prefix_and_plain_text_fallbacks() {
        let json_resp = parse_dialogue_response(r#"{"reply_text":"おかえり！","emotion":"happy"}"#)
            .expect("json");
        assert_eq!(json_resp.reply_text, "おかえり！");
        assert_eq!(json_resp.emotion, EmotionTag::Happy);

        let fenced = parse_dialogue_response(
            "```json\n{\"reply_text\":\"眠いよ…\",\"emotion\":\"SLEEPY\"}\n```",
        )
        .expect("fenced json");
        assert_eq!(fenced.reply_text, "眠いよ…");
        assert_eq!(fenced.emotion, EmotionTag::Sleepy);

        let prefixed =
            parse_dialogue_response("[shy] えへへ、ありがとう！").expect("prefix fallback");
        assert_eq!(prefixed.reply_text, "えへへ、ありがとう！");
        assert_eq!(prefixed.emotion, EmotionTag::Shy);

        let plain = parse_dialogue_response("こんにちは！").expect("plain fallback");
        assert_eq!(plain.reply_text, "こんにちは！");
        assert_eq!(plain.emotion, EmotionTag::Neutral);
    }

    #[tokio::test]
    async fn dialogue_client_prunes_history_and_sends_bounded_turns() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        let server = MockHttpServer::start(move |_| {
            let idx = counter_clone.fetch_add(1, Ordering::SeqCst) + 1;
            let text = match idx {
                1 => r#"{"reply_text":"reply-1","emotion":"happy"}"#.to_owned(),
                2 => "[surprised] reply-2".to_owned(),
                _ => "reply-3".to_owned(),
            };
            let envelope = json!({
                "candidates": [{
                    "content": { "parts": [{ "text": text }] }
                }]
            })
            .to_string();
            MockResponse::json(200, envelope)
        })
        .await;

        let cfg = AiConfig {
            chat_api_key: "test-chat-key".to_owned(),
            chat_model: "gemini-2.5-flash".to_owned(),
            max_history_turns: 2,
            ..Default::default()
        };

        let mut client = DialogueClient::new()
            .expect("build client")
            .with_endpoint(server.base_url());

        let r1 = client.chat("hello 1", &cfg).await.expect("turn 1");
        assert_eq!(r1.emotion, EmotionTag::Happy);
        assert_eq!(client.history().len(), 2);

        let r2 = client.chat("hello 2", &cfg).await.expect("turn 2");
        assert_eq!(r2.emotion, EmotionTag::Surprised);
        assert_eq!(client.history().len(), 4);

        let r3 = client.chat("hello 3", &cfg).await.expect("turn 3");
        assert_eq!(r3.reply_text, "reply-3");
        assert_eq!(r3.emotion, EmotionTag::Neutral);
        // Bounded to 2 turns (4 messages), oldest turn pruned
        assert_eq!(client.history().len(), 4);
        assert_eq!(client.history()[0].content, "hello 2");
        assert_eq!(client.history()[1].content, "reply-2");
        assert_eq!(client.history()[2].content, "hello 3");
        assert_eq!(client.history()[3].content, "reply-3");

        let reqs = server.requests();
        assert_eq!(reqs.len(), 3);
        let req3_json: Value = serde_json::from_str(&reqs[2].body).expect("req 3 json");
        // Turn 3 sent history (4 msgs) + current user msg = 5 contents entries
        assert_eq!(req3_json["contents"].as_array().expect("contents").len(), 5);
        assert!(req3_json["systemInstruction"]["parts"][0]["text"]
            .as_str()
            .expect("system prompt")
            .contains(&cfg.system_prompt));

        client.clear_history();
        assert!(client.history().is_empty());
    }

    #[tokio::test]
    async fn dialogue_client_supports_openai_compatible_with_optional_api_key() {
        let server = MockHttpServer::start(move |req| {
            if req.target == "/v1/chat/completions" {
                let envelope = json!({
                    "choices": [{
                        "message": {
                            "content": r#"{"reply_text":"ローカルLLMからこんにちは！","emotion":"happy"}"#
                        }
                    }]
                })
                .to_string();
                MockResponse::json(200, envelope)
            } else {
                MockResponse::json(404, "{}")
            }
        })
        .await;

        let mut cfg = AiConfig {
            chat_provider: "openai_compatible".to_owned(),
            chat_endpoint: server.base_url(),
            chat_model: "qwen2.5:7b".to_owned(),
            chat_api_key: String::new(),
            ..Default::default()
        };

        let mut client = DialogueClient::new().expect("build client");
        let resp = client
            .chat("やあ", &cfg)
            .await
            .expect("local chat succeeds");
        assert_eq!(resp.reply_text, "ローカルLLMからこんにちは！");
        assert_eq!(resp.emotion, EmotionTag::Happy);
        assert!(!server.requests()[0].headers.contains_key("authorization"));

        cfg.chat_api_key = "sk-test-token".to_owned();
        let _ = client
            .chat("元気？", &cfg)
            .await
            .expect("auth chat succeeds");
        assert_eq!(
            server.requests()[1]
                .headers
                .get("authorization")
                .map(String::as_str),
            Some("Bearer sk-test-token")
        );
    }
}
