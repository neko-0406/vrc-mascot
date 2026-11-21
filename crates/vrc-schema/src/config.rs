use std::fs;
use std::path::{Path, PathBuf};

use directories::BaseDirs;
use serde::{Deserialize, Serialize};

use crate::error::SchemaError;

/// Subdirectory under `%APPDATA%` (`BaseDirs::config_dir`) where user configuration lives.
pub const APP_CONFIG_DIR_NAME: &str = "vrc-mascot";
/// Filename of the persistent configuration file.
pub const CONFIG_FILE_NAME: &str = "config.toml";

/// Returns `%APPDATA%/vrc-mascot/config.toml` (via `directories::BaseDirs::config_dir`).
pub fn config_path() -> Result<PathBuf, SchemaError> {
    let base_dirs = BaseDirs::new().ok_or_else(|| {
        SchemaError::DirectoryUnavailable("failed to resolve user BaseDirs".to_owned())
    })?;
    Ok(base_dirs
        .config_dir()
        .join(APP_CONFIG_DIR_NAME)
        .join(CONFIG_FILE_NAME))
}

/// Expands `%APPDATA%` in `raw` and returns a normalized `PathBuf`.
pub fn resolve_extract_output_dir(raw: &str) -> PathBuf {
    let appdata = BaseDirs::new()
        .map(|b| b.config_dir().to_path_buf())
        .or_else(|| std::env::var_os("APPDATA").map(PathBuf::from));

    resolve_extract_output_dir_with(raw, appdata.as_deref())
}

fn resolve_extract_output_dir_with(raw: &str, appdata_dir: Option<&Path>) -> PathBuf {
    let trimmed = raw.trim();
    if let Some(base) = appdata_dir {
        let rest_opt = trimmed
            .strip_prefix("%APPDATA%/")
            .or_else(|| trimmed.strip_prefix("%APPDATA%\\"))
            .or_else(|| trimmed.strip_prefix("%appdata%/"))
            .or_else(|| trimmed.strip_prefix("%appdata%\\"));

        if let Some(rest) = rest_opt {
            return rest
                .split(['/', '\\'])
                .filter(|seg| !seg.is_empty())
                .fold(base.to_path_buf(), |acc, seg| acc.join(seg));
        }

        if trimmed.eq_ignore_ascii_case("%APPDATA%") {
            return base.to_path_buf();
        }
    }

    PathBuf::from(trimmed)
}

/// Root persistent application configuration stored in `%APPDATA%/vrc-mascot/config.toml`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub motion: MotionConfig,
    pub ai: AiConfig,
    pub voice: VoiceConfig,
    pub display: DisplayConfig,
    pub physics_shader: PhysicsShaderConfig,
    pub avatar_import: AvatarImportConfig,
}

impl AppConfig {
    /// Loads configuration from [`config_path()`], or returns `AppConfig::default()` if the file does not exist.
    pub fn load_or_default() -> Result<Self, SchemaError> {
        let path = config_path()?;
        Self::load_or_default_from(&path)
    }

    /// Loads configuration from `path`, or returns `AppConfig::default()` if `path` does not exist.
    pub fn load_or_default_from(path: impl AsRef<Path>) -> Result<Self, SchemaError> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = fs::read_to_string(path)?;
        let cfg: Self = toml::from_str(&raw)?;
        Ok(cfg)
    }

    /// Persists this configuration to [`config_path()`].
    pub fn save(&self) -> Result<(), SchemaError> {
        let path = config_path()?;
        self.save_to(&path)
    }

    /// Persists this configuration to `path`, creating parent directories as needed.
    pub fn save_to(&self, path: impl AsRef<Path>) -> Result<(), SchemaError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        let encoded = toml::to_string_pretty(self)?;
        fs::write(path, encoded)?;
        Ok(())
    }

    /// Loads `.env` if present (without failing if absent) and fills only empty string fields
    /// in `self` from environment variables.
    pub fn apply_env_fallbacks(&mut self) {
        let _ = dotenvy::dotenv();
        self.apply_env_fallbacks_with(|key| std::env::var(key).ok());
    }

    /// Testable helper that resolves fallback variables via the supplied lookup closure.
    pub fn apply_env_fallbacks_with<F>(&mut self, lookup: F)
    where
        F: Fn(&str) -> Option<String>,
    {
        let non_empty = |key: &str| {
            lookup(key).and_then(|v| {
                let trimmed = v.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_owned())
                }
            })
        };

        if let Some(api_key) = non_empty("GEMINI_API_KEY") {
            if self.ai.chat_api_key.trim().is_empty() {
                self.ai.chat_api_key = api_key.clone();
            }
            if self.ai.decisions_api_key.trim().is_empty() {
                self.ai.decisions_api_key = api_key;
            }
        }

        if self.ai.chat_model.trim().is_empty() {
            if let Some(model) = non_empty("GEMINI_MODEL_DIALOGUE") {
                self.ai.chat_model = model;
            }
        }

        if self.ai.decisions_model.trim().is_empty() {
            if let Some(model) = non_empty("GEMINI_MODEL_DECISION") {
                self.ai.decisions_model = model;
            }
        }

        if self.voice.voicevox_url.trim().is_empty() {
            if let Some(url) = non_empty("VOICEVOX_BASE_URL") {
                self.voice.voicevox_url = url;
            }
        }
    }

    /// Resolves `avatar_import.extract_output_dir` with `%APPDATA%` expansion.
    pub fn resolved_extract_output_dir(&self) -> PathBuf {
        resolve_extract_output_dir(&self.avatar_import.extract_output_dir)
    }
}

/// Autonomous motion, physics-interaction, and window-perching settings (`CF-01`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MotionConfig {
    pub autonomous_enabled: bool,
    pub idle_enabled: bool,
    pub blink_enabled: bool,
    pub gaze_tracking_enabled: bool,
    pub window_perch_enabled: bool,
    pub gravity_enabled: bool,
    pub interaction_enabled: bool,
    pub move_speed_scale: f32,
    pub perch_exclude_apps: Vec<String>,
    pub pause_on_fullscreen: bool,
    pub hide_on_fullscreen: bool,
}

impl Default for MotionConfig {
    fn default() -> Self {
        Self {
            autonomous_enabled: true,
            idle_enabled: true,
            blink_enabled: true,
            gaze_tracking_enabled: true,
            window_perch_enabled: true,
            gravity_enabled: true,
            interaction_enabled: true,
            move_speed_scale: 1.0,
            perch_exclude_apps: vec![
                "VALORANT-Win64-Shipping.exe".to_owned(),
                "obs64.exe".to_owned(),
            ],
            pause_on_fullscreen: true,
            hide_on_fullscreen: false,
        }
    }
}

fn default_chat_endpoint() -> String {
    "https://generativelanguage.googleapis.com".to_owned()
}

fn default_decisions_provider() -> String {
    "gemini".to_owned()
}

/// Dialogue LLM and Decisions AI configuration (`CF-02`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct AiConfig {
    pub chat_provider: String,
    #[serde(default = "default_chat_endpoint")]
    pub chat_endpoint: String,
    pub chat_model: String,
    pub chat_api_key: String,
    pub decisions_enabled: bool,
    #[serde(default = "default_decisions_provider")]
    pub decisions_provider: String,
    pub decisions_endpoint: String,
    pub decisions_model: String,
    pub decisions_api_key: String,
    pub decision_interval_sec: u64,
    pub max_history_turns: usize,
    pub system_prompt: String,
    pub send_active_window_title: bool,
}

impl Default for AiConfig {
    fn default() -> Self {
        Self {
            chat_provider: "gemini".to_owned(),
            chat_endpoint: default_chat_endpoint(),
            chat_model: "gemini-2.5-flash".to_owned(),
            chat_api_key: String::new(),
            decisions_enabled: true,
            decisions_provider: default_decisions_provider(),
            decisions_endpoint: "https://generativelanguage.googleapis.com".to_owned(),
            decisions_model: "gemini-2.5-flash-lite".to_owned(),
            decisions_api_key: String::new(),
            decision_interval_sec: 30,
            max_history_turns: 20,
            system_prompt: "あなたは親しみやすいデスクトップマスコットです。".to_owned(),
            send_active_window_title: false,
        }
    }
}

/// VOICEVOX Engine speech synthesis and lip-sync parameters (`CF-03`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VoiceConfig {
    pub voicevox_url: String,
    pub speaker_id: u32,
    pub speed_scale: f32,
    pub pitch_scale: f32,
    pub intonation_scale: f32,
    pub volume_scale: f32,
    pub muted: bool,
    pub lip_sync_enabled: bool,
    pub lip_sync_sensitivity: f32,
}

impl Default for VoiceConfig {
    fn default() -> Self {
        Self {
            voicevox_url: "http://127.0.0.1:50021".to_owned(),
            speaker_id: 1,
            speed_scale: 1.0,
            pitch_scale: 0.0,
            intonation_scale: 1.0,
            volume_scale: 1.0,
            muted: false,
            lip_sync_enabled: true,
            lip_sync_sensitivity: 1.0,
        }
    }
}

/// Avatar overlay window and OS startup settings (`CF-04`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DisplayConfig {
    pub scale: f32,
    pub always_on_top: bool,
    pub click_through_locked: bool,
    pub max_fps: u32,
    pub auto_start: bool,
    pub show_window_on_startup: bool,
}

impl Default for DisplayConfig {
    fn default() -> Self {
        Self {
            scale: 1.0,
            always_on_top: true,
            click_through_locked: false,
            max_fps: 60,
            auto_start: false,
            show_window_on_startup: false,
        }
    }
}

/// Rendering quality preset (`CF-05`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QualityPreset {
    Low,
    Medium,
    #[default]
    High,
}

/// PhysBone simulation and shader pass toggles (`CF-05`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PhysicsShaderConfig {
    pub physbone_enabled: bool,
    pub physbone_strength: f32,
    pub physbone_collider_enabled: bool,
    pub outline_enabled: bool,
    pub fur_enabled: bool,
    pub audiolink_enabled: bool,
    pub quality_preset: QualityPreset,
}

impl Default for PhysicsShaderConfig {
    fn default() -> Self {
        Self {
            physbone_enabled: true,
            physbone_strength: 1.0,
            physbone_collider_enabled: true,
            outline_enabled: true,
            fur_enabled: true,
            audiolink_enabled: false,
            quality_preset: QualityPreset::High,
        }
    }
}

/// Default avatar selection and Unity extraction output configuration (`CF-06`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct AvatarImportConfig {
    pub default_avatar_id: String,
    pub extract_output_dir: String,
    pub modular_avatar_enabled: bool,
    pub max_texture_size: u32,
}

impl Default for AvatarImportConfig {
    fn default() -> Self {
        Self {
            default_avatar_id: String::new(),
            extract_output_dir: "%APPDATA%/vrc-mascot/avatars".to_owned(),
            modular_avatar_enabled: true,
            max_texture_size: 2048,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::tempdir;

    #[test]
    fn default_config_round_trip_toml() {
        let dir = tempdir().expect("create tempdir");
        let path = dir.path().join("vrc-mascot").join("config.toml");

        let cfg = AppConfig::default();
        cfg.save_to(&path).expect("save default config");
        let loaded = AppConfig::load_or_default_from(&path).expect("load saved config");
        assert_eq!(cfg, loaded);
    }

    #[test]
    fn load_or_default_from_missing_returns_default() {
        let dir = tempdir().expect("create tempdir");
        let path = dir.path().join("nonexistent.toml");
        let loaded = AppConfig::load_or_default_from(&path).expect("default when missing");
        assert_eq!(loaded, AppConfig::default());
        assert_eq!(loaded.ai.chat_provider, "gemini");
        assert_eq!(loaded.ai.decision_interval_sec, 30);
        assert_eq!(loaded.voice.speaker_id, 1);
        assert_eq!(loaded.physics_shader.quality_preset, QualityPreset::High);
    }

    #[test]
    fn env_fallback_only_fills_empty_fields() {
        let mut cfg = AppConfig::default();
        cfg.ai.chat_api_key = "existing-key".to_owned();
        cfg.voice.voicevox_url = String::new();
        cfg.ai.chat_model = String::new();

        let env = HashMap::from([
            ("GEMINI_API_KEY", "env-key"),
            ("GEMINI_MODEL_DIALOGUE", "gemini-2.5-pro"),
            ("GEMINI_MODEL_DECISION", "gemini-2.5-flash"),
            ("VOICEVOX_BASE_URL", "http://127.0.0.1:50022"),
        ]);

        cfg.apply_env_fallbacks_with(|k| env.get(k).map(|v| (*v).to_owned()));

        assert_eq!(cfg.ai.chat_api_key, "existing-key");
        assert_eq!(cfg.ai.decisions_api_key, "env-key");
        assert_eq!(cfg.ai.chat_model, "gemini-2.5-pro");
        // decisions_model was non-empty default, so it must NOT be overwritten
        assert_eq!(cfg.ai.decisions_model, "gemini-2.5-flash-lite");
        assert_eq!(cfg.voice.voicevox_url, "http://127.0.0.1:50022");
    }

    #[test]
    fn resolve_extract_output_dir_expands_appdata() {
        let base = Path::new("C:\\Users\\Test\\AppData\\Roaming");
        let resolved = resolve_extract_output_dir_with("%APPDATA%/vrc-mascot/avatars", Some(base));
        assert_eq!(
            resolved,
            PathBuf::from("C:\\Users\\Test\\AppData\\Roaming")
                .join("vrc-mascot")
                .join("avatars")
        );
    }

    #[test]
    fn legacy_ai_config_toml_deserializes_with_new_field_defaults() {
        let legacy_toml = r#"
[ai]
chat_provider = "gemini"
chat_model = "gemini-2.5-flash"
decisions_enabled = true
decisions_endpoint = "https://generativelanguage.googleapis.com"
decisions_model = "gemini-2.5-flash-lite"
"#;
        let cfg: AppConfig = toml::from_str(legacy_toml).expect("parse legacy config");
        assert_eq!(cfg.ai.chat_provider, "gemini");
        assert_eq!(
            cfg.ai.chat_endpoint,
            "https://generativelanguage.googleapis.com"
        );
        assert_eq!(cfg.ai.decisions_provider, "gemini");
    }
}
