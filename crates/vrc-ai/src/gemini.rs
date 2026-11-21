//! Shared Gemini `generateContent` HTTP and response-parsing helpers.

use serde::Deserialize;
use serde_json::Value;

use crate::error::AiError;

pub(crate) const DEFAULT_GEMINI_ENDPOINT: &str = "https://generativelanguage.googleapis.com";

#[derive(Debug, Deserialize)]
struct GenerateContentResponse {
    #[serde(default)]
    candidates: Vec<Candidate>,
    #[serde(rename = "promptFeedback", default)]
    prompt_feedback: Option<PromptFeedback>,
}

#[derive(Debug, Deserialize)]
struct PromptFeedback {
    #[serde(rename = "blockReason", default)]
    block_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Candidate {
    #[serde(default)]
    content: Option<CandidateContent>,
    #[serde(rename = "finishReason", default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CandidateContent {
    #[serde(default)]
    parts: Vec<CandidatePart>,
}

#[derive(Debug, Deserialize)]
struct CandidatePart {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    thought: Option<bool>,
}

pub(crate) fn build_generate_content_url(
    endpoint: &str,
    model: &str,
    source_name: &str,
) -> Result<String, AiError> {
    let mut base = endpoint.trim().trim_end_matches('/');
    if base.is_empty() {
        base = DEFAULT_GEMINI_ENDPOINT;
    }

    if base.ends_with(":generateContent") {
        return Ok(base.to_owned());
    }

    for suffix in ["/v1beta/decisions", "/v1beta/models", "/v1beta"] {
        if let Some(stripped) = base.strip_suffix(suffix) {
            base = stripped.trim_end_matches('/');
            break;
        }
    }
    if base.is_empty() {
        base = DEFAULT_GEMINI_ENDPOINT;
    }

    let clean_model = model.trim().trim_start_matches("models/");
    if clean_model.is_empty() {
        return Err(AiError::InvalidResponse {
            source_name: source_name.to_owned(),
            reason: "model name must not be empty".to_owned(),
        });
    }
    Ok(format!(
        "{base}/v1beta/models/{clean_model}:generateContent"
    ))
}

pub(crate) async fn post_generate_content(
    client: &reqwest::Client,
    url: &str,
    api_key: &str,
    payload: &Value,
    source_name: &str,
) -> Result<String, AiError> {
    let response = client
        .post(url)
        .header("x-goog-api-key", api_key)
        .json(payload)
        .send()
        .await?;

    let status = response.status();
    let body = response.text().await?;
    if !status.is_success() {
        return Err(AiError::ApiStatus {
            status: status.as_u16(),
            body,
        });
    }

    extract_candidate_text(&body, source_name)
}

pub(crate) fn extract_candidate_text(body: &str, source_name: &str) -> Result<String, AiError> {
    let parsed: GenerateContentResponse = serde_json::from_str(body)?;

    if parsed.candidates.is_empty() {
        let reason = parsed
            .prompt_feedback
            .and_then(|f| f.block_reason)
            .map(|b| format!("response blocked: {b}"))
            .unwrap_or_else(|| "no candidates returned".to_owned());
        return Err(AiError::InvalidResponse {
            source_name: source_name.to_owned(),
            reason,
        });
    }

    let candidate = &parsed.candidates[0];
    let parts = candidate
        .content
        .as_ref()
        .map(|c| c.parts.as_slice())
        .unwrap_or(&[]);

    // Prefer non-thought text parts (Gemini 2.5 models may emit thought parts first).
    let non_thought: String = parts
        .iter()
        .filter(|p| !p.thought.unwrap_or(false))
        .filter_map(|p| p.text.as_deref())
        .collect::<Vec<_>>()
        .join("");

    if !non_thought.trim().is_empty() {
        return Ok(non_thought);
    }

    let fallback: String = parts
        .iter()
        .filter_map(|p| p.text.as_deref())
        .collect::<Vec<_>>()
        .join("");

    if !fallback.trim().is_empty() {
        return Ok(fallback);
    }

    let finish = candidate.finish_reason.as_deref().unwrap_or("unknown");
    Err(AiError::InvalidResponse {
        source_name: source_name.to_owned(),
        reason: format!("candidate contained no text parts (finishReason: {finish})"),
    })
}

pub(crate) fn strip_markdown_code_fence(raw: &str) -> &str {
    let trimmed = raw.trim();
    let Some(after_open) = trimmed.strip_prefix("```") else {
        return trimmed;
    };
    let after_lang = match after_open.find('\n') {
        Some(idx) => &after_open[idx + 1..],
        None => return trimmed,
    };
    if let Some(before_close) = after_lang.strip_suffix("```") {
        before_close.trim()
    } else {
        trimmed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_generate_content_url_normalizes_legacy_and_full_endpoints() {
        assert_eq!(
            build_generate_content_url(
                "https://generativelanguage.googleapis.com",
                "gemini-2.5-flash",
                "t"
            )
            .unwrap(),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        );
        assert_eq!(
            build_generate_content_url(
                "https://generativelanguage.googleapis.com/v1beta/decisions/",
                "models/gemini-2.5-flash-lite",
                "t"
            )
            .unwrap(),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"
        );
        assert_eq!(
            build_generate_content_url(
                "https://generativelanguage.googleapis.com/v1beta",
                "gemini-2.0-flash",
                "t"
            )
            .unwrap(),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        );
        assert_eq!(
            build_generate_content_url(
                "https://generativelanguage.googleapis.com/v1beta/models/custom:generateContent",
                "ignored",
                "t"
            )
            .unwrap(),
            "https://generativelanguage.googleapis.com/v1beta/models/custom:generateContent"
        );
    }
}
