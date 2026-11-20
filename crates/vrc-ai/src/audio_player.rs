//! WAV audio playback via `rodio` synchronized with a [`LipSyncTimeline`] timer.

use std::io::Cursor;
use std::time::Instant;

use rodio::{Decoder, DeviceSinkBuilder, MixerDeviceSink, Player};
use vrc_schema::{VisemeKey, VisemeWeights};

use crate::error::AiError;
use crate::lipsync::LipSyncTimeline;

struct RodioBackend {
    _sink: MixerDeviceSink,
    player: Player,
}

/// Synchronized WAV speech player and lip-sync viseme sampler.
///
/// If no audio output device is available (e.g., headless environments) or when muted,
/// playback degrades gracefully to a timer-driven lip-sync runner.
pub struct SpeechPlayer {
    backend: Option<RodioBackend>,
    timeline: LipSyncTimeline,
    started_at: Option<Instant>,
    volume: f32,
    muted: bool,
}

impl Default for SpeechPlayer {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for SpeechPlayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpeechPlayer")
            .field("has_audio_device", &self.backend.is_some())
            .field("timeline_duration_sec", &self.timeline.duration_sec())
            .field("is_playing", &self.is_playing())
            .field("volume", &self.volume)
            .field("muted", &self.muted)
            .finish()
    }
}

impl SpeechPlayer {
    /// Initializes a `SpeechPlayer`, opening the default OS audio device if available.
    ///
    /// Never fails when an audio device is absent; falls back to headless lip-sync timing.
    pub fn new() -> Self {
        let backend = match DeviceSinkBuilder::open_default_sink() {
            Ok(mut sink) => {
                sink.log_on_drop(false);
                let player = Player::connect_new(sink.mixer());
                Some(RodioBackend {
                    _sink: sink,
                    player,
                })
            }
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    "no default audio output device available; running SpeechPlayer in headless timer mode"
                );
                None
            }
        };

        Self {
            backend,
            timeline: LipSyncTimeline::default(),
            started_at: None,
            volume: 1.0,
            muted: false,
        }
    }

    /// Creates a timer-only `SpeechPlayer` without opening an OS audio device.
    pub fn new_headless() -> Self {
        Self {
            backend: None,
            timeline: LipSyncTimeline::default(),
            started_at: None,
            volume: 1.0,
            muted: false,
        }
    }

    /// Returns `true` if a physical audio output sink was opened.
    pub fn has_audio_device(&self) -> bool {
        self.backend.is_some()
    }

    /// Starts playing `wav_bytes` aligned with `timeline`, applying `volume` and `muted`.
    pub fn play(
        &mut self,
        wav_bytes: Vec<u8>,
        timeline: LipSyncTimeline,
        volume: f32,
        muted: bool,
    ) -> Result<(), AiError> {
        self.stop();

        self.volume = sanitize_volume(volume);
        self.muted = muted;
        self.timeline = timeline;

        if let Some(backend) = &self.backend {
            if !wav_bytes.is_empty() {
                let decoder = Decoder::new_wav(Cursor::new(wav_bytes))
                    .map_err(|err| AiError::AudioOutput(format!("WAV decode failed: {err}")))?;
                backend.player.set_volume(self.effective_volume());
                backend.player.append(decoder);
                backend.player.play();
            }
        }

        self.started_at = Some(Instant::now());
        Ok(())
    }

    /// Stops any active playback and resets the lip-sync timer.
    pub fn stop(&mut self) {
        self.started_at = None;
        if let Some(backend) = &self.backend {
            backend.player.stop();
        }
    }

    /// Updates playback volume (`>= 0.0`) immediately.
    pub fn set_volume(&mut self, volume: f32) {
        self.volume = sanitize_volume(volume);
        self.sync_backend_volume();
    }

    /// Returns the configured non-muted volume.
    pub fn volume(&self) -> f32 {
        self.volume
    }

    /// Enables or disables mute immediately.
    pub fn set_muted(&mut self, muted: bool) {
        self.muted = muted;
        self.sync_backend_volume();
    }

    /// Returns `true` if the player is currently muted.
    pub fn is_muted(&self) -> bool {
        self.muted
    }

    /// Returns `true` while elapsed playback time is within the active [`LipSyncTimeline`].
    pub fn is_playing(&self) -> bool {
        let Some(started_at) = self.started_at else {
            return false;
        };
        started_at.elapsed().as_secs_f32() < self.timeline.duration_sec()
    }

    /// Samples the current lip-sync [`VisemeWeights`] and active [`VisemeKey`] for this frame.
    pub fn sample_viseme(&self, sensitivity: f32) -> (VisemeWeights, VisemeKey) {
        let Some(started_at) = self.started_at else {
            return (VisemeWeights::default(), VisemeKey::Sil);
        };
        let elapsed = started_at.elapsed().as_secs_f32();
        if elapsed >= self.timeline.duration_sec() {
            return (VisemeWeights::default(), VisemeKey::Sil);
        }
        (
            self.timeline.evaluate(elapsed, sensitivity),
            self.timeline.current_viseme(elapsed),
        )
    }

    fn effective_volume(&self) -> f32 {
        if self.muted {
            0.0
        } else {
            self.volume
        }
    }

    fn sync_backend_volume(&self) {
        if let Some(backend) = &self.backend {
            backend.player.set_volume(self.effective_volume());
        }
    }
}

fn sanitize_volume(volume: f32) -> f32 {
    if volume.is_finite() {
        volume.max(0.0)
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lipsync::LipSyncSegment;

    fn silent_pcm_wav(duration_ms: u32, sample_rate: u32) -> Vec<u8> {
        let num_samples = (sample_rate * duration_ms) / 1000;
        let data_bytes = num_samples * 2;
        let riff_size = 36 + data_bytes;

        let mut buf = Vec::with_capacity((44 + data_bytes) as usize);
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&riff_size.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16_u32.to_le_bytes()); // PCM chunk size
        buf.extend_from_slice(&1_u16.to_le_bytes()); // PCM format
        buf.extend_from_slice(&1_u16.to_le_bytes()); // mono
        buf.extend_from_slice(&sample_rate.to_le_bytes());
        buf.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
        buf.extend_from_slice(&2_u16.to_le_bytes()); // block align
        buf.extend_from_slice(&16_u16.to_le_bytes()); // bits per sample
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_bytes.to_le_bytes());
        buf.resize((44 + data_bytes) as usize, 0);
        buf
    }

    #[test]
    fn headless_player_runs_lipsync_timer_and_supports_controls() {
        let timeline = LipSyncTimeline::from_segments(vec![LipSyncSegment {
            start_sec: 0.0,
            end_sec: 0.5,
            viseme: VisemeKey::Aa,
            peak: 1.0,
        }]);

        let mut player = SpeechPlayer::new_headless();
        assert!(!player.has_audio_device());
        assert!(!player.is_playing());

        player
            .play(silent_pcm_wav(500, 24_000), timeline, 0.75, true)
            .expect("headless play");
        assert!(player.is_playing());
        assert!(player.is_muted());
        assert!((player.volume() - 0.75).abs() < 1e-6);

        let (_weights, key) = player.sample_viseme(1.0);
        assert_eq!(key, VisemeKey::Aa);

        player.set_muted(false);
        assert!(!player.is_muted());
        player.set_volume(0.5);
        assert!((player.volume() - 0.5).abs() < 1e-6);

        player.stop();
        assert!(!player.is_playing());
        assert_eq!(
            player.sample_viseme(1.0),
            (VisemeWeights::default(), VisemeKey::Sil)
        );
    }

    #[test]
    fn default_player_initializes_gracefully_and_plays_muted_wav() {
        let timeline = LipSyncTimeline::from_segments(vec![LipSyncSegment {
            start_sec: 0.0,
            end_sec: 0.2,
            viseme: VisemeKey::Oh,
            peak: 1.0,
        }]);

        let mut player = SpeechPlayer::new();
        player
            .play(silent_pcm_wav(200, 24_000), timeline, 0.8, true)
            .expect("play succeeds with or without audio device");
        assert!(player.is_playing());
        let (_weights, key) = player.sample_viseme(1.0);
        assert_eq!(key, VisemeKey::Oh);
        player.stop();
        assert!(!player.is_playing());
    }
}
