//! VOICEVOX Engine REST client for health checks, speaker discovery, audio queries, and WAV synthesis.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use vrc_schema::VoiceConfig;

use crate::error::AiError;
use crate::lipsync::LipSyncTimeline;

fn default_one() -> f32 {
    1.0
}

/// Speaker style entry returned by VOICEVOX `GET /speakers`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VoicevoxSpeakerStyle {
    pub name: String,
    pub id: u32,
    #[serde(rename = "type", default)]
    pub style_type: Option<String>,
}

/// Speaker metadata returned by VOICEVOX `GET /speakers`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VoicevoxSpeaker {
    pub name: String,
    #[serde(default)]
    pub speaker_uuid: String,
    #[serde(default)]
    pub styles: Vec<VoicevoxSpeakerStyle>,
    #[serde(default)]
    pub version: Option<String>,
}

/// Single mora timing and pitch entry inside an [`AccentPhrase`].
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Mora {
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub consonant: Option<String>,
    #[serde(default)]
    pub consonant_length: Option<f32>,
    #[serde(default)]
    pub vowel: String,
    #[serde(default)]
    pub vowel_length: f32,
    #[serde(default)]
    pub pitch: f32,
}

/// Accent phrase returned by VOICEVOX `POST /audio_query`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct AccentPhrase {
    #[serde(default)]
    pub moras: Vec<Mora>,
    #[serde(default)]
    pub accent: usize,
    #[serde(default)]
    pub pause_mora: Option<Mora>,
    #[serde(default)]
    pub is_interrogative: Option<bool>,
}

/// VOICEVOX `AudioQuery` payload used for synthesis and lip-sync timeline extraction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioQuery {
    #[serde(default)]
    pub accent_phrases: Vec<AccentPhrase>,
    #[serde(rename = "speedScale", default = "default_one")]
    pub speed_scale: f32,
    #[serde(rename = "pitchScale", default)]
    pub pitch_scale: f32,
    #[serde(rename = "intonationScale", default = "default_one")]
    pub intonation_scale: f32,
    #[serde(rename = "volumeScale", default = "default_one")]
    pub volume_scale: f32,
    #[serde(rename = "prePhonemeLength", default)]
    pub pre_phoneme_length: f32,
    #[serde(rename = "postPhonemeLength", default)]
    pub post_phoneme_length: f32,
    #[serde(rename = "outputSamplingRate", default)]
    pub output_sampling_rate: Option<u32>,
    #[serde(rename = "outputStereo", default)]
    pub output_stereo: Option<bool>,
    #[serde(default)]
    pub kana: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl Default for AudioQuery {
    fn default() -> Self {
        Self {
            accent_phrases: Vec::new(),
            speed_scale: 1.0,
            pitch_scale: 0.0,
            intonation_scale: 1.0,
            volume_scale: 1.0,
            pre_phoneme_length: 0.0,
            post_phoneme_length: 0.0,
            output_sampling_rate: None,
            output_stereo: None,
            kana: None,
            extra: BTreeMap::new(),
        }
    }
}

/// Combined output of VOICEVOX synthesis and its aligned [`LipSyncTimeline`].
#[derive(Debug, Clone, PartialEq)]
pub struct SynthesizedSpeech {
    pub text: String,
    pub audio_query: AudioQuery,
    pub wav_bytes: Vec<u8>,
    pub timeline: LipSyncTimeline,
}

/// Async client for the local/LAN VOICEVOX Engine REST API.
#[derive(Debug, Clone)]
pub struct VoicevoxClient {
    client: reqwest::Client,
}

impl Default for VoicevoxClient {
    fn default() -> Self {
        Self::with_client(reqwest::Client::new())
    }
}

impl VoicevoxClient {
    /// Creates a new `VoicevoxClient` with a default `reqwest::Client`.
    pub fn new() -> Result<Self, AiError> {
        let client = reqwest::Client::builder().build()?;
        Ok(Self { client })
    }

    /// Creates a `VoicevoxClient` reusing an existing `reqwest::Client`.
    pub fn with_client(client: reqwest::Client) -> Self {
        Self { client }
    }

    /// Checks VOICEVOX Engine health via `GET {base_url}/version`, returning the engine version.
    pub async fn check_health(&self, base_url: &str) -> Result<String, AiError> {
        let base = normalize_base_url(base_url)?;
        let url = format!("{base}/version");

        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|err| map_voicevox_transport_error(base, err))?;

        let status = resp.status();
        let body = resp.text().await?;
        if !status.is_success() {
            return Err(AiError::ApiStatus {
                status: status.as_u16(),
                body,
            });
        }

        if let Ok(version) = serde_json::from_str::<String>(&body) {
            return Ok(version);
        }
        let trimmed = body.trim();
        if trimmed.is_empty() {
            return Err(AiError::InvalidResponse {
                source_name: "voicevox_version".to_owned(),
                reason: "empty /version response".to_owned(),
            });
        }
        Ok(trimmed.to_owned())
    }

    /// Lists available speakers and styles via `GET {base_url}/speakers`.
    pub async fn list_speakers(&self, base_url: &str) -> Result<Vec<VoicevoxSpeaker>, AiError> {
        let base = normalize_base_url(base_url)?;
        let url = format!("{base}/speakers");

        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|err| map_voicevox_transport_error(base, err))?;

        let status = resp.status();
        let body = resp.text().await?;
        if !status.is_success() {
            return Err(AiError::ApiStatus {
                status: status.as_u16(),
                body,
            });
        }

        let speakers: Vec<VoicevoxSpeaker> = serde_json::from_str(&body)?;
        Ok(speakers)
    }

    /// Queries mora timings for `text` via `POST {url}/audio_query?text=...&speaker=...`,
    /// applying the scale parameters from `cfg`.
    pub async fn audio_query(&self, text: &str, cfg: &VoiceConfig) -> Result<AudioQuery, AiError> {
        let base = normalize_base_url(&cfg.voicevox_url)?;
        let encoded_text = percent_encode_query(text);
        let url = format!(
            "{base}/audio_query?text={encoded_text}&speaker={}",
            cfg.speaker_id
        );

        let resp = self
            .client
            .post(&url)
            .send()
            .await
            .map_err(|err| map_voicevox_transport_error(base, err))?;

        let status = resp.status();
        let body = resp.text().await?;
        if !status.is_success() {
            return Err(AiError::ApiStatus {
                status: status.as_u16(),
                body,
            });
        }

        let mut query: AudioQuery = serde_json::from_str(&body)?;
        query.speed_scale = cfg.speed_scale;
        query.pitch_scale = cfg.pitch_scale;
        query.intonation_scale = cfg.intonation_scale;
        query.volume_scale = cfg.volume_scale;
        Ok(query)
    }

    /// Synthesizes WAV audio bytes from an [`AudioQuery`] via `POST {url}/synthesis?speaker=...`.
    pub async fn synthesis(
        &self,
        query: &AudioQuery,
        cfg: &VoiceConfig,
    ) -> Result<Vec<u8>, AiError> {
        let base = normalize_base_url(&cfg.voicevox_url)?;
        let url = format!("{base}/synthesis?speaker={}", cfg.speaker_id);

        let resp = self
            .client
            .post(&url)
            .json(query)
            .send()
            .await
            .map_err(|err| map_voicevox_transport_error(base, err))?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(AiError::ApiStatus {
                status: status.as_u16(),
                body,
            });
        }

        let bytes = resp.bytes().await?.to_vec();
        if bytes.is_empty() {
            return Err(AiError::InvalidResponse {
                source_name: "voicevox_synthesis".to_owned(),
                reason: "received empty WAV payload".to_owned(),
            });
        }
        Ok(bytes)
    }

    /// Runs `audio_query`, builds a [`LipSyncTimeline`], and calls `synthesis` in one convenience step.
    pub async fn synthesize_with_timeline(
        &self,
        text: &str,
        cfg: &VoiceConfig,
    ) -> Result<SynthesizedSpeech, AiError> {
        let query = self.audio_query(text, cfg).await?;
        let timeline = LipSyncTimeline::from_audio_query(&query);
        let wav_bytes = self.synthesis(&query, cfg).await?;
        Ok(SynthesizedSpeech {
            text: text.to_owned(),
            audio_query: query,
            wav_bytes,
            timeline,
        })
    }
}

fn normalize_base_url(raw: &str) -> Result<&str, AiError> {
    let trimmed = raw.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return Err(AiError::VoicevoxOffline {
            url: String::new(),
            reason: "VOICEVOX URL is empty".to_owned(),
        });
    }
    Ok(trimmed)
}

fn percent_encode_query(input: &str) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(input.len());
    for byte in input.as_bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            out.push(*byte as char);
        } else {
            let _ = write!(out, "%{byte:02X}");
        }
    }
    out
}

fn map_voicevox_transport_error(base_url: &str, err: reqwest::Error) -> AiError {
    if err.is_connect() || err.is_timeout() || err.is_request() {
        AiError::VoicevoxOffline {
            url: base_url.to_owned(),
            reason: err.to_string(),
        }
    } else {
        AiError::Http(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{MockHttpServer, MockResponse};
    use serde_json::json;
    use vrc_schema::VisemeKey;

    #[tokio::test]
    async fn voicevox_health_speakers_and_synthesize_with_timeline() {
        let query_json = json!({
            "accent_phrases": [{
                "moras": [
                    {
                        "text": "ア",
                        "consonant": null,
                        "consonant_length": null,
                        "vowel": "a",
                        "vowel_length": 0.2,
                        "pitch": 5.0
                    },
                    {
                        "text": "オ",
                        "consonant": null,
                        "consonant_length": null,
                        "vowel": "o",
                        "vowel_length": 0.2,
                        "pitch": 5.0
                    }
                ],
                "accent": 1,
                "pause_mora": null,
                "is_interrogative": false
            }],
            "speedScale": 1.0,
            "pitchScale": 0.0,
            "intonationScale": 1.0,
            "volumeScale": 1.0,
            "prePhonemeLength": 0.05,
            "postPhonemeLength": 0.05,
            "outputSamplingRate": 24000,
            "outputStereo": false
        })
        .to_string();

        let fake_wav = b"RIFF\x24\x00\x00\x00WAVEfmt ".to_vec();
        let fake_wav_resp = fake_wav.clone();

        let server = MockHttpServer::start(move |req| {
            if req.method == "GET" && req.target == "/version" {
                MockResponse::json(200, "\"0.14.10\"")
            } else if req.method == "GET" && req.target == "/speakers" {
                MockResponse::json(
                    200,
                    r#"[{"name":"四国めたん","speaker_uuid":"uuid-1","styles":[{"name":"ノーマル","id":2}]}]"#,
                )
            } else if req.method == "POST" && req.target.starts_with("/audio_query?") {
                MockResponse::json(200, query_json.clone())
            } else if req.method == "POST" && req.target.starts_with("/synthesis?") {
                MockResponse::bytes(200, "audio/wav", fake_wav_resp.clone())
            } else {
                MockResponse::json(404, "{}")
            }
        })
        .await;

        let client = VoicevoxClient::new().expect("build voicevox client");
        let version = client
            .check_health(&server.base_url())
            .await
            .expect("health ok");
        assert_eq!(version, "0.14.10");

        let speakers = client
            .list_speakers(&server.base_url())
            .await
            .expect("speakers ok");
        assert_eq!(speakers.len(), 1);
        assert_eq!(speakers[0].styles[0].id, 2);

        let cfg = VoiceConfig {
            voicevox_url: server.base_url(),
            speaker_id: 2,
            speed_scale: 1.25,
            pitch_scale: 0.05,
            intonation_scale: 1.1,
            volume_scale: 0.9,
            ..Default::default()
        };

        let synth = client
            .synthesize_with_timeline("あお", &cfg)
            .await
            .expect("synthesize ok");
        assert_eq!(synth.wav_bytes, fake_wav);
        assert_eq!(synth.audio_query.speed_scale, 1.25);
        assert_eq!(synth.audio_query.pitch_scale, 0.05);
        assert_eq!(synth.audio_query.intonation_scale, 1.1);
        assert_eq!(synth.audio_query.volume_scale, 0.9);

        // Total raw duration = 0.05 + 0.2 + 0.2 + 0.05 = 0.50; at 1.25x -> 0.40s
        assert!((synth.timeline.duration_sec() - 0.40).abs() < 1e-5);
        assert_eq!(synth.timeline.current_viseme(0.10), VisemeKey::Aa);

        let reqs = server.requests();
        let aq_req = reqs
            .iter()
            .find(|r| r.target.starts_with("/audio_query?"))
            .expect("audio_query request");
        assert!(aq_req.target.contains("speaker=2"));

        let syn_req = reqs
            .iter()
            .find(|r| r.target.starts_with("/synthesis?"))
            .expect("synthesis request");
        assert!(syn_req.target.contains("speaker=2"));
        let sent_query: AudioQuery = serde_json::from_str(&syn_req.body).expect("sent query json");
        assert_eq!(sent_query.speed_scale, 1.25);
    }

    #[tokio::test]
    async fn voicevox_offline_returns_offline_error() {
        // Bind and drop immediately so the port refuses connections.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let port = listener.local_addr().expect("addr").port();
        drop(listener);

        let client = VoicevoxClient::new().expect("build client");
        let err = client
            .check_health(&format!("http://127.0.0.1:{port}"))
            .await
            .expect_err("offline error");
        assert!(matches!(err, AiError::VoicevoxOffline { .. }));
    }
}
