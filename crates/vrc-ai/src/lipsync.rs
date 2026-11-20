//! Mora-to-viseme lip-sync timeline and trapezoidal Attack/Release envelope evaluation.

use serde::{Deserialize, Serialize};
use vrc_schema::{VisemeKey, VisemeWeights};

use crate::voicevox::AudioQuery;

const DEFAULT_ATTACK_SEC: f32 = 0.025;
const DEFAULT_RELEASE_SEC: f32 = 0.035;

/// Single timed viseme interval on a lip-sync timeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LipSyncSegment {
    pub start_sec: f32,
    pub end_sec: f32,
    pub viseme: VisemeKey,
    pub peak: f32,
}

/// Timeline of viseme segments built from a VOICEVOX [`AudioQuery`].
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct LipSyncTimeline {
    segments: Vec<LipSyncSegment>,
    total_duration_sec: f32,
}

impl LipSyncTimeline {
    /// Builds a [`LipSyncTimeline`] directly from precomputed segments.
    pub fn from_segments(segments: Vec<LipSyncSegment>) -> Self {
        let total_duration_sec = segments
            .iter()
            .map(|s| s.end_sec)
            .fold(0.0_f32, f32::max)
            .max(0.0);
        Self {
            segments,
            total_duration_sec,
        }
    }

    /// Builds a [`LipSyncTimeline`] from a VOICEVOX [`AudioQuery`], scaling all phoneme lengths
    /// inversely by `query.speed_scale`.
    pub fn from_audio_query(query: &AudioQuery) -> Self {
        let speed = if query.speed_scale.is_finite() && query.speed_scale > 0.0 {
            query.speed_scale
        } else {
            1.0
        };

        let mut segments = Vec::new();
        let mut cursor = 0.0_f32;

        let push_segment =
            |segments: &mut Vec<LipSyncSegment>, cursor: &mut f32, dur: f32, viseme: VisemeKey| {
                if !dur.is_finite() || dur <= 0.0 {
                    return;
                }
                let start_sec = *cursor;
                let end_sec = start_sec + dur;
                let peak = if viseme == VisemeKey::Sil { 0.0 } else { 1.0 };
                segments.push(LipSyncSegment {
                    start_sec,
                    end_sec,
                    viseme,
                    peak,
                });
                *cursor = end_sec;
            };

        push_segment(
            &mut segments,
            &mut cursor,
            query.pre_phoneme_length.max(0.0) / speed,
            VisemeKey::Sil,
        );

        for phrase in &query.accent_phrases {
            for mora in &phrase.moras {
                let cons_len = mora.consonant_length.unwrap_or(0.0).max(0.0) / speed;
                push_segment(&mut segments, &mut cursor, cons_len, VisemeKey::Sil);

                let vowel_len = mora.vowel_length.max(0.0) / speed;
                let viseme = vowel_to_viseme(&mora.vowel);
                push_segment(&mut segments, &mut cursor, vowel_len, viseme);
            }

            if let Some(pause) = &phrase.pause_mora {
                let p_cons = pause.consonant_length.unwrap_or(0.0).max(0.0) / speed;
                let p_vowel = pause.vowel_length.max(0.0) / speed;
                let p_viseme = vowel_to_viseme(&pause.vowel);
                push_segment(&mut segments, &mut cursor, p_cons, VisemeKey::Sil);
                push_segment(&mut segments, &mut cursor, p_vowel, p_viseme);
            }
        }

        push_segment(
            &mut segments,
            &mut cursor,
            query.post_phoneme_length.max(0.0) / speed,
            VisemeKey::Sil,
        );

        Self {
            segments,
            total_duration_sec: cursor,
        }
    }

    /// Returns all segments in chronological order.
    pub fn segments(&self) -> &[LipSyncSegment] {
        &self.segments
    }

    /// Returns the total duration of the timeline in seconds.
    pub fn duration_sec(&self) -> f32 {
        self.total_duration_sec
    }

    /// Returns the active [`VisemeKey`] at `elapsed_sec` (or [`VisemeKey::Sil`] outside the range).
    pub fn current_viseme(&self, elapsed_sec: f32) -> VisemeKey {
        self.segment_at(elapsed_sec)
            .map(|s| s.viseme)
            .unwrap_or(VisemeKey::Sil)
    }

    /// Evaluates smooth [`VisemeWeights`] at `elapsed_sec` using a trapezoidal Attack/Release
    /// envelope scaled by `sensitivity` and clamped to `[0.0, 1.0]`.
    pub fn evaluate(&self, elapsed_sec: f32, sensitivity: f32) -> VisemeWeights {
        if !elapsed_sec.is_finite()
            || !sensitivity.is_finite()
            || sensitivity <= 0.0
            || elapsed_sec < 0.0
            || elapsed_sec >= self.total_duration_sec
        {
            return VisemeWeights::default();
        }

        let Some(seg) = self.segment_at(elapsed_sec) else {
            return VisemeWeights::default();
        };
        if seg.viseme == VisemeKey::Sil || seg.peak <= 0.0 {
            return VisemeWeights::default();
        }

        let env = trapezoidal_envelope(elapsed_sec - seg.start_sec, seg.end_sec - seg.start_sec);
        let weight = (env * seg.peak * sensitivity).clamp(0.0, 1.0);
        VisemeWeights::from_key(seg.viseme, weight)
    }

    fn segment_at(&self, elapsed_sec: f32) -> Option<&LipSyncSegment> {
        if !elapsed_sec.is_finite() || elapsed_sec < 0.0 || elapsed_sec >= self.total_duration_sec {
            return None;
        }
        let idx = self
            .segments
            .partition_point(|s| s.start_sec <= elapsed_sec);
        if idx == 0 {
            return None;
        }
        let seg = &self.segments[idx - 1];
        if elapsed_sec >= seg.start_sec && elapsed_sec < seg.end_sec {
            Some(seg)
        } else {
            None
        }
    }
}

/// Maps a VOICEVOX vowel symbol to its corresponding VRChat [`VisemeKey`].
pub fn vowel_to_viseme(vowel: &str) -> VisemeKey {
    match vowel.trim() {
        "a" | "A" => VisemeKey::Aa,
        "i" | "I" => VisemeKey::Ih,
        "u" | "U" => VisemeKey::Ou,
        "e" | "E" => VisemeKey::Ee,
        "o" | "O" => VisemeKey::Oh,
        "N" | "pau" | "cl" | "" => VisemeKey::Sil,
        _ => VisemeKey::Sil,
    }
}

fn trapezoidal_envelope(t: f32, dur: f32) -> f32 {
    if dur <= 0.0 || t < 0.0 || t >= dur {
        return 0.0;
    }
    let scale = (dur / (DEFAULT_ATTACK_SEC + DEFAULT_RELEASE_SEC)).min(1.0);
    let attack = DEFAULT_ATTACK_SEC * scale;
    let release = DEFAULT_RELEASE_SEC * scale;

    if attack > 0.0 && t < attack {
        (t / attack).clamp(0.0, 1.0)
    } else if release > 0.0 && t > dur - release {
        ((dur - t) / release).clamp(0.0, 1.0)
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voicevox::{AccentPhrase, Mora};
    use std::collections::BTreeMap;

    fn sample_query(speed_scale: f32) -> AudioQuery {
        AudioQuery {
            accent_phrases: vec![AccentPhrase {
                moras: vec![
                    Mora {
                        text: "カ".to_owned(),
                        consonant: Some("k".to_owned()),
                        consonant_length: Some(0.04),
                        vowel: "a".to_owned(),
                        vowel_length: 0.16,
                        pitch: 5.0,
                    },
                    Mora {
                        text: "キ".to_owned(),
                        consonant: None,
                        consonant_length: None,
                        vowel: "I".to_owned(),
                        vowel_length: 0.10,
                        pitch: 5.0,
                    },
                    Mora {
                        text: "ン".to_owned(),
                        consonant: None,
                        consonant_length: None,
                        vowel: "N".to_owned(),
                        vowel_length: 0.08,
                        pitch: 5.0,
                    },
                ],
                accent: 1,
                pause_mora: Some(Mora {
                    text: "、".to_owned(),
                    consonant: None,
                    consonant_length: None,
                    vowel: "pau".to_owned(),
                    vowel_length: 0.20,
                    pitch: 0.0,
                }),
                is_interrogative: Some(false),
            }],
            speed_scale,
            pitch_scale: 0.0,
            intonation_scale: 1.0,
            volume_scale: 1.0,
            pre_phoneme_length: 0.10,
            post_phoneme_length: 0.12,
            output_sampling_rate: Some(24000),
            output_stereo: Some(false),
            kana: None,
            extra: BTreeMap::new(),
        }
    }

    #[test]
    fn vowel_mapping_covers_all_documented_vowels() {
        assert_eq!(vowel_to_viseme("a"), VisemeKey::Aa);
        assert_eq!(vowel_to_viseme("A"), VisemeKey::Aa);
        assert_eq!(vowel_to_viseme("i"), VisemeKey::Ih);
        assert_eq!(vowel_to_viseme("I"), VisemeKey::Ih);
        assert_eq!(vowel_to_viseme("u"), VisemeKey::Ou);
        assert_eq!(vowel_to_viseme("U"), VisemeKey::Ou);
        assert_eq!(vowel_to_viseme("e"), VisemeKey::Ee);
        assert_eq!(vowel_to_viseme("E"), VisemeKey::Ee);
        assert_eq!(vowel_to_viseme("o"), VisemeKey::Oh);
        assert_eq!(vowel_to_viseme("O"), VisemeKey::Oh);
        assert_eq!(vowel_to_viseme("N"), VisemeKey::Sil);
        assert_eq!(vowel_to_viseme("pau"), VisemeKey::Sil);
        assert_eq!(vowel_to_viseme("cl"), VisemeKey::Sil);
        assert_eq!(vowel_to_viseme(""), VisemeKey::Sil);
    }

    #[test]
    fn timeline_scales_by_speed_and_evaluates_trapezoidal_envelope() {
        let query_1x = sample_query(1.0);
        let tl_1x = LipSyncTimeline::from_audio_query(&query_1x);
        // 0.10 + 0.04 + 0.16 + 0.10 + 0.08 + 0.20 + 0.12 = 0.80s
        assert!((tl_1x.duration_sec() - 0.80).abs() < 1e-5);

        let query_2x = sample_query(2.0);
        let tl_2x = LipSyncTimeline::from_audio_query(&query_2x);
        assert!((tl_2x.duration_sec() - 0.40).abs() < 1e-5);

        // Inside prePhoneme (0.05s) -> Sil
        assert_eq!(tl_1x.current_viseme(0.05), VisemeKey::Sil);
        assert_eq!(tl_1x.evaluate(0.05, 1.0), VisemeWeights::default());

        // Inside "a" segment [0.14, 0.30] (dur = 0.16s, attack = 0.025, release = 0.035)
        assert_eq!(tl_1x.current_viseme(0.20), VisemeKey::Aa);

        // Attack ramp at t = 0.0125s -> ~0.5
        let w_attack = tl_1x.evaluate(0.14 + 0.0125, 1.0);
        assert!((w_attack.aa - 0.5).abs() < 1e-4, "aa={}", w_attack.aa);

        // Plateau at t = 0.08s -> 1.0 (or clamped with sensitivity > 1.0)
        let w_plateau = tl_1x.evaluate(0.14 + 0.08, 1.0);
        assert!((w_plateau.aa - 1.0).abs() < 1e-5);
        let w_scaled = tl_1x.evaluate(0.14 + 0.08, 0.6);
        assert!((w_scaled.aa - 0.6).abs() < 1e-5);

        // Release ramp at dur - 0.0175s -> ~0.5
        let w_release = tl_1x.evaluate(0.30 - 0.0175, 1.0);
        assert!((w_release.aa - 0.5).abs() < 1e-4, "aa={}", w_release.aa);

        // Past end -> Sil / 0.0
        assert_eq!(tl_1x.current_viseme(0.85), VisemeKey::Sil);
        assert_eq!(tl_1x.evaluate(0.85, 1.0), VisemeWeights::default());
    }
}
