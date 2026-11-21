//! OpenAI Chat Completions compatible client for OpenAI, Ollama, LM Studio, OpenRouter, etc.

use serde::Deserialize;
use serde_json::{json, Value};
use vrc_schema::{AiConfig, ChatMessage, ChatRole, DecisionContext};

use crate::error::AiError;
use crate::gemini::strip_markdown_code_fence;

const OPENAI_DECISION_SYSTEM_PROMPT: &str = "\
You are the autonomous behavior decision engine for a Windows 3D desktop mascot. \
Given a JSON environment snapshot (idle_seconds, pet_count_recent, active_window_category, \
optional active_window_title, time_of_day, current_mood), respond ONLY with a JSON object matching: \
{\"action\":\"SPEAK|MOVE_TO_TASKBAR|SIT_ON_WINDOW|CHANGE_EXPRESSION|IDLE\",\"target_expression\":string|null,\"speech_intent\":string|null,\"reason\":string|null}. \
Prefer IDLE when the user is actively working or nothing notable happened.";

const OPENAI_DIALOGUE_FORMAT_DIRECTIVE: &str = "\
Respond ONLY with a JSON object matching {\"reply_text\": string, \"emotion\": \"neutral|happy|sad|angry|surprised|shy|sleepy\"}.";

#[derive(Debug, Deserialize)]
struct ChatCompletionsResponse {
    #[serde(default)]
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    #[serde(default)]
    message: Option<ChatChoiceMessage>,
}

#[derive(Debug, Deserialize)]
struct ChatChoiceMessage {
    #[serde(default)]
    content: Option<Value>,
}

/// Normalizes an OpenAI-compatible base URL to `.../v1/chat/completions`.
pub(crate) fn build_chat_completions_url(
    endpoint: &str,
    source_name: &str,
) -> Result<String, AiError> {
    let base = endpoint.trim().trim_end_matches('/');
    if base.is_empty() {
        return Err(AiError::InvalidResponse {
            source_name: source_name.to_owned(),
            reason: "OpenAI-compatible endpoint URL must not be empty".to_owned(),
        });
    }
    if base.ends_with("/chat/completions") {
        return Ok(base.to_owned());
    }
    if base.ends_with("/v1") {
        return Ok(format!("{base}/chat/completions"));
    }
    Ok(format!("{base}/v1/chat/completions"))
}

pub(crate) fn build_openai_decision_payload(
    ctx: &DecisionContext,
    model: &str,
    source_name: &str,
) -> Result<Value, AiError> {
    let clean_model = model.trim();
    if clean_model.is_empty() {
        return Err(AiError::InvalidResponse {
            source_name: source_name.to_owned(),
            reason: "model name must not be empty".to_owned(),
        });
    }
    let ctx_json = serde_json::to_string(ctx)?;
    Ok(json!({
        "model": clean_model,
        "temperature": 0.3,
        "messages": [
            { "role": "system", "content": OPENAI_DECISION_SYSTEM_PROMPT },
            { "role": "user", "content": ctx_json }
        ]
    }))
}

pub(crate) fn build_openai_dialogue_payload(
    history: &[ChatMessage],
    current_user_prompt: &str,
    ai_cfg: &AiConfig,
    source_name: &str,
) -> Result<Value, AiError> {
    let clean_model = ai_cfg.chat_model.trim();
    if clean_model.is_empty() {
        return Err(AiError::InvalidResponse {
            source_name: source_name.to_owned(),
            reason: "model name must not be empty".to_owned(),
        });
    }

    let system_text = if ai_cfg.system_prompt.trim().is_empty() {
        OPENAI_DIALOGUE_FORMAT_DIRECTIVE.to_owned()
    } else {
        format!(
            "{}\n\n{}",
            ai_cfg.system_prompt.trim(),
            OPENAI_DIALOGUE_FORMAT_DIRECTIVE
        )
    };

    let mut messages = Vec::with_capacity(history.len() + 2);
    messages.push(json!({
        "role": "system",
        "content": system_text
    }));

    if ai_cfg.max_history_turns > 0 {
        for msg in history {
            let role = match msg.role {
                ChatRole::User => "user",
                ChatRole::Assistant => "assistant",
                ChatRole::System => "system",
            };
            messages.push(json!({
                "role": role,
                "content": msg.content
            }));
        }
    }

    messages.push(json!({
        "role": "user",
        "content": current_user_prompt
    }));

    Ok(json!({
        "model": clean_model,
        "temperature": 0.7,
        "messages": messages
    }))
}

pub(crate) async fn post_chat_completions(
    client: &reqwest::Client,
    url: &str,
    api_key: &str,
    payload: &Value,
    source_name: &str,
) -> Result<String, AiError> {
    let mut req = client.post(url).json(payload);
    let trimmed_key = api_key.trim();
    if !trimmed_key.is_empty() {
        req = req.bearer_auth(trimmed_key);
    }

    let response = req.send().await?;
    let status = response.status();
    let body = response.text().await?;
    if !status.is_success() {
        return Err(AiError::ApiStatus {
            status: status.as_u16(),
            body,
        });
    }

    extract_choice_text(&body, source_name)
}

fn extract_choice_text(body: &str, source_name: &str) -> Result<String, AiError> {
    let parsed: ChatCompletionsResponse = serde_json::from_str(body)?;
    let content = parsed
        .choices
        .first()
        .and_then(|c| c.message.as_ref())
        .and_then(|m| m.content.as_ref())
        .ok_or_else(|| AiError::InvalidResponse {
            source_name: source_name.to_owned(),
            reason: "no choices/message content returned".to_owned(),
        })?;

    let raw_text = match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        _ => {
            return Err(AiError::InvalidResponse {
                source_name: source_name.to_owned(),
                reason: "unsupported message content type".to_owned(),
            });
        }
    };

    let cleaned = strip_markdown_code_fence(&raw_text).trim();
    if cleaned.is_empty() {
        return Err(AiError::InvalidResponse {
            source_name: source_name.to_owned(),
            reason: "choice content was empty".to_owned(),
        });
    }
    Ok(cleaned.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_openai_compatible_endpoints() {
        assert_eq!(
            build_chat_completions_url("http://127.0.0.1:11434", "t").unwrap(),
            "http://127.0.0.1:11434/v1/chat/completions"
        );
        assert_eq!(
            build_chat_completions_url("http://127.0.0.1:1234/v1/", "t").unwrap(),
            "http://127.0.0.1:1234/v1/chat/completions"
        );
        assert_eq!(
            build_chat_completions_url("https://api.openai.com/v1/chat/completions", "t").unwrap(),
            "https://api.openai.com/v1/chat/completions"
        );
    }
}
