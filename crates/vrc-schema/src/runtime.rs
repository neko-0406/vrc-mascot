use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Lip-sync viseme keys supported by VRChat avatars and VOICEVOX vowel mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VisemeKey {
    /// Silence / closed mouth (`sil`).
    #[default]
    Sil,
    /// Open vowel "a" (`aa`).
    Aa,
    /// Front vowel "i" (`ih`).
    Ih,
    /// Rounded vowel "u" (`ou`).
    Ou,
    /// Mid-front vowel "e" (`ee`).
    Ee,
    /// Mid-back vowel "o" (`oh`).
    Oh,
}

/// BlendShape weights for the five active lip-sync visemes.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct VisemeWeights {
    pub aa: f32,
    pub ih: f32,
    pub ou: f32,
    pub ee: f32,
    pub oh: f32,
}

impl VisemeWeights {
    /// Returns weights in `[aa, ih, ou, ee, oh]` order.
    pub const fn as_array(&self) -> [f32; 5] {
        [self.aa, self.ih, self.ou, self.ee, self.oh]
    }

    /// Creates weights with a single active viseme set to `weight` (clamped to `[0.0, 1.0]`).
    pub fn from_key(key: VisemeKey, weight: f32) -> Self {
        let mut weights = Self::default();
        weights.set(key, weight);
        weights
    }

    /// Reads the weight for a specific viseme (`Sil` always returns `0.0`).
    pub const fn get(&self, key: VisemeKey) -> f32 {
        match key {
            VisemeKey::Sil => 0.0,
            VisemeKey::Aa => self.aa,
            VisemeKey::Ih => self.ih,
            VisemeKey::Ou => self.ou,
            VisemeKey::Ee => self.ee,
            VisemeKey::Oh => self.oh,
        }
    }

    /// Sets the weight for a specific viseme (clamped to `[0.0, 1.0]`).
    pub fn set(&mut self, key: VisemeKey, weight: f32) {
        let clamped = weight.clamp(0.0, 1.0);
        match key {
            VisemeKey::Sil => {}
            VisemeKey::Aa => self.aa = clamped,
            VisemeKey::Ih => self.ih = clamped,
            VisemeKey::Ou => self.ou = clamped,
            VisemeKey::Ee => self.ee = clamped,
            VisemeKey::Oh => self.oh = clamped,
        }
    }

    /// Returns a copy with all weights clamped to `[0.0, 1.0]`.
    pub fn clamped(self) -> Self {
        Self {
            aa: self.aa.clamp(0.0, 1.0),
            ih: self.ih.clamp(0.0, 1.0),
            ou: self.ou.clamp(0.0, 1.0),
            ee: self.ee.clamp(0.0, 1.0),
            oh: self.oh.clamp(0.0, 1.0),
        }
    }
}

/// Emotional state tags emitted by dialogue AI and used to drive expressions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmotionTag {
    #[default]
    Neutral,
    Happy,
    Sad,
    Angry,
    Surprised,
    Shy,
    Sleepy,
    Custom(String),
}

/// High-level autonomous actions returned by the decision model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DecisionAction {
    Speak,
    MoveToTaskbar,
    SitOnWindow,
    ChangeExpression,
    #[default]
    Idle,
}

/// Environmental and interaction snapshot sent to the decision model.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct DecisionContext {
    pub idle_seconds: f32,
    pub pet_count_recent: u32,
    pub active_window_category: String,
    pub active_window_title: Option<String>,
    pub time_of_day: String,
    pub current_mood: EmotionTag,
}

/// Structured output returned by the autonomous decision model.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct DecisionOutput {
    pub action: DecisionAction,
    pub target_expression: Option<String>,
    pub speech_intent: Option<String>,
    pub reason: Option<String>,
}

/// Entry displayed in the Status & Dialogue UI decision history log.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionLogEntry {
    pub timestamp_epoch_ms: u64,
    pub action: DecisionAction,
    pub target_expression: Option<String>,
    pub speech_intent: Option<String>,
    pub reason: String,
}

/// Participant role in the dialogue history.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatRole {
    User,
    Assistant,
    System,
}

/// Single dialogue message stored in chat history.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    pub emotion: Option<EmotionTag>,
    pub timestamp_epoch_ms: u64,
}

/// FFT audio energy bands and smoothing history passed to the shader uniform.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AudioLinkData {
    /// Normalized amplitudes for `[Bass, LowMid, HighMid, Treble]`.
    pub bands: [f32; 4],
    /// Rolling 32-frame history of the 4 bands (`[0]` = newest).
    pub history: [[f32; 4]; 32],
    /// Whether audio loopback capture is currently active.
    pub active: bool,
}

impl Default for AudioLinkData {
    fn default() -> Self {
        Self {
            bands: [0.0; 4],
            history: [[0.0; 4]; 32],
            active: false,
        }
    }
}

/// Desktop mascot movement and window-perching state machine state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum MascotState {
    #[default]
    Idle,
    Dragged,
    Falling,
    SeatedOnTaskbar,
    SeatedOnWindow,
}

/// Background Unity project extraction status reported to the Import UI tab.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub enum ImportJobStatus {
    #[default]
    Idle,
    Running {
        progress: f32,
        stage: String,
    },
    Succeeded {
        avatar_id: String,
        manifest_path: PathBuf,
    },
    Failed {
        error: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viseme_weights_helpers_and_clamp() {
        let mut weights = VisemeWeights::from_key(VisemeKey::Aa, 1.5);
        assert_eq!(weights.get(VisemeKey::Aa), 1.0);
        assert_eq!(weights.get(VisemeKey::Sil), 0.0);

        weights.set(VisemeKey::Oh, 0.45);
        assert_eq!(weights.as_array(), [1.0, 0.0, 0.0, 0.0, 0.45]);

        let raw = VisemeWeights {
            aa: -0.2,
            ih: 0.5,
            ou: 2.0,
            ee: 0.25,
            oh: 0.75,
        };
        assert_eq!(raw.clamped().as_array(), [0.0, 0.5, 1.0, 0.25, 0.75]);
    }

    #[test]
    fn decision_and_runtime_types_round_trip_json() {
        let output = DecisionOutput {
            action: DecisionAction::MoveToTaskbar,
            target_expression: Some("Smile".to_owned()),
            speech_intent: None,
            reason: Some("idle timeout".to_owned()),
        };
        let json = serde_json::to_string(&output).expect("serialize decision output");
        assert!(json.contains("\"MOVE_TO_TASKBAR\""));
        let decoded: DecisionOutput =
            serde_json::from_str(&json).expect("deserialize decision output");
        assert_eq!(output, decoded);

        let al = AudioLinkData {
            bands: [0.8, 0.5, 0.3, 0.1],
            active: true,
            ..Default::default()
        };
        let al_json = serde_json::to_string(&al).expect("serialize audiolink");
        let al_decoded: AudioLinkData =
            serde_json::from_str(&al_json).expect("deserialize audiolink");
        assert_eq!(al, al_decoded);

        let chat = ChatMessage {
            role: ChatRole::Assistant,
            content: "こんにちは！".to_owned(),
            emotion: Some(EmotionTag::Happy),
            timestamp_epoch_ms: 1_700_000_000_000,
        };
        let chat_json = serde_json::to_string(&chat).expect("serialize chat");
        let chat_decoded: ChatMessage = serde_json::from_str(&chat_json).expect("deserialize chat");
        assert_eq!(chat, chat_decoded);
    }
}
