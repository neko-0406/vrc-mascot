//! Centralized application state (`AppState`), Tokio background worker (`AppCommand` / `AppEvent`),
//! avatar loading/switching, AI/VOICEVOX orchestration, and Unity import coordination.

use std::path::{Path, PathBuf};
use std::sync::mpsc as std_mpsc;
use std::time::{SystemTime, UNIX_EPOCH};

use tokio::sync::mpsc as tokio_mpsc;
use vrc_ai::{AiService, LipSyncTimeline, SpeechPlayer, VoicevoxSpeaker};
use vrc_extractor::{
    discover_avatars, discover_scenes, extract_avatar, DiscoveredAvatar, ExtractOptions,
    ExtractProgress, ExtractReport,
};
use vrc_physics::{MascotSimulation, SkeletonNode};
use vrc_renderer::{AudioLinkCapture, AvatarModel};
use vrc_schema::{
    AiConfig, AppConfig, AvatarRegistry, ChatMessage, ChatRole, DecisionAction, DecisionContext,
    DecisionLogEntry, DecisionOutput, EmotionTag, ImportJobStatus, MascotManifest, MascotState,
    VisemeKey, VisemeWeights, VoiceConfig,
};

use crate::window_manager::win32;

/// Commands dispatched from the UI / winit loop to the Tokio background worker.
#[derive(Debug, Clone)]
pub enum AppCommand {
    RunDecision(DecisionContext),
    SendChat(String),
    CheckVoicevox,
    DiscoverScenes(PathBuf),
    DiscoverAvatars {
        project_dir: PathBuf,
        scene_path: PathBuf,
    },
    RunImport(ExtractOptions),
    SaveConfig(AppConfig),
    UpdateAiVoiceConfig {
        ai: AiConfig,
        voice: VoiceConfig,
    },
    Shutdown,
}

/// Events emitted from the Tokio background worker back to [`AppState`].
#[derive(Debug, Clone)]
pub enum AppEvent {
    DecisionResult {
        output: DecisionOutput,
        timestamp_epoch_ms: u64,
        error: Option<String>,
    },
    AutonomousSpeak {
        text: String,
        emotion: EmotionTag,
        wav_bytes: Option<Vec<u8>>,
        timeline: Option<LipSyncTimeline>,
        timestamp_epoch_ms: u64,
        voice_error: Option<String>,
    },
    ChatReply {
        text: String,
        emotion: EmotionTag,
        wav_bytes: Option<Vec<u8>>,
        timeline: Option<LipSyncTimeline>,
        timestamp_epoch_ms: u64,
        voice_error: Option<String>,
    },
    ChatError(String),
    VoicevoxStatus {
        online: bool,
        version: Option<String>,
        speakers: Vec<VoicevoxSpeaker>,
        error: Option<String>,
    },
    ScenesDiscovered(Result<Vec<PathBuf>, String>),
    AvatarsDiscovered(Result<Vec<DiscoveredAvatar>, String>),
    ImportProgress(ExtractProgress),
    ImportFinished(Result<ExtractReport, String>),
    ConfigSaved(Result<(), String>),
}

/// Deferred perch movement requested by Decisions AI or user actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PendingPerchAction {
    MoveToTaskbar,
    SitOnWindow,
}

/// UI state for the Unity Avatar Import tab.
#[derive(Debug, Clone)]
pub struct ImportFormState {
    pub project_dir: String,
    pub discovered_scenes: Vec<PathBuf>,
    pub selected_scene_idx: usize,
    pub discovered_avatars: Vec<DiscoveredAvatar>,
    pub selected_avatar_idx: usize,
    pub custom_avatar_name: String,
    pub modular_avatar_enabled: bool,
    pub max_texture_size: u32,
    pub logs: Vec<String>,
}

impl ImportFormState {
    pub fn from_config(cfg: &AppConfig) -> Self {
        Self {
            project_dir: String::new(),
            discovered_scenes: Vec::new(),
            selected_scene_idx: 0,
            discovered_avatars: Vec::new(),
            selected_avatar_idx: 0,
            custom_avatar_name: String::new(),
            modular_avatar_enabled: cfg.avatar_import.modular_avatar_enabled,
            max_texture_size: cfg.avatar_import.max_texture_size,
            logs: Vec::new(),
        }
    }

    pub fn push_log(&mut self, line: impl Into<String>) {
        self.logs.push(line.into());
        if self.logs.len() > 200 {
            let excess = self.logs.len() - 200;
            self.logs.drain(..excess);
        }
    }
}

/// Unified runtime state owned by the single `winit` event loop.
pub struct AppState {
    pub config: AppConfig,
    pub draft_config: AppConfig,
    pub registry: AvatarRegistry,
    pub active_avatar_id: Option<String>,
    pub active_manifest_path: Option<PathBuf>,
    pub manifest: MascotManifest,
    pub simulation: MascotSimulation,
    pub speech_player: SpeechPlayer,
    pub audiolink_capture: AudioLinkCapture,
    pub mascot_state: MascotState,
    pub current_mood: EmotionTag,
    pub chat_history: Vec<ChatMessage>,
    pub chat_input: String,
    pub chat_in_flight: bool,
    pub decision_log: Vec<DecisionLogEntry>,
    pub decision_in_flight: bool,
    pub voicevox_online: bool,
    pub voicevox_version: Option<String>,
    pub voicevox_speakers: Vec<VoicevoxSpeaker>,
    pub voicevox_last_error: Option<String>,
    pub current_viseme: VisemeKey,
    pub current_viseme_weights: VisemeWeights,
    pub fps_estimate: f32,
    pub import_status: ImportJobStatus,
    pub import_form: ImportFormState,
    pub status_notice: Option<String>,
    pub pending_model_reload: Option<(AvatarModel, MascotManifest)>,
    pub pending_perch_action: Option<PendingPerchAction>,
    pub menu_dirty: bool,
    pub status_window_visible: bool,
    cmd_tx: tokio_mpsc::UnboundedSender<AppCommand>,
    event_rx: std_mpsc::Receiver<AppEvent>,
    _runtime: Option<tokio::runtime::Runtime>,
}

impl Drop for AppState {
    fn drop(&mut self) {
        let _ = self.cmd_tx.send(AppCommand::Shutdown);
    }
}

impl AppState {
    /// Initializes runtime state, loads initial avatar, starts AudioLink capture (if enabled),
    /// and spawns the Tokio background worker.
    pub fn initialize(
        config: AppConfig,
        cli_manifest: Option<&Path>,
        show_window_override: bool,
    ) -> anyhow::Result<Self> {
        let extract_dir = config.resolved_extract_output_dir();
        let registry = AvatarRegistry::scan(&extract_dir).unwrap_or_else(|err| {
            tracing::warn!(error = %err, dir = %extract_dir.display(), "AvatarRegistry scan failed");
            AvatarRegistry::new(&extract_dir)
        });

        let (model, manifest, active_avatar_id, active_manifest_path) =
            load_initial_avatar(&config, &registry, cli_manifest);

        let speech_player = SpeechPlayer::new();
        let audiolink_capture = AudioLinkCapture::new(config.physics_shader.audiolink_enabled);

        let mut state = Self::build_with_parts(
            config,
            registry,
            model,
            manifest,
            active_avatar_id,
            active_manifest_path,
            speech_player,
            audiolink_capture,
            show_window_override,
            None,
        )?;
        state.check_voicevox();
        Ok(state)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_with_parts(
        config: AppConfig,
        registry: AvatarRegistry,
        model: AvatarModel,
        manifest: MascotManifest,
        active_avatar_id: Option<String>,
        active_manifest_path: Option<PathBuf>,
        mut speech_player: SpeechPlayer,
        audiolink_capture: AudioLinkCapture,
        show_window_override: bool,
        dialogue_endpoint_override: Option<String>,
    ) -> anyhow::Result<Self> {
        speech_player.set_volume(config.voice.volume_scale);
        speech_player.set_muted(config.voice.muted);

        let rest_nodes = skeleton_nodes_from_model(&model);
        let simulation = MascotSimulation::new(&manifest, rest_nodes);

        let (cmd_tx, cmd_rx) = tokio_mpsc::unbounded_channel();
        let (event_tx, event_rx) = std_mpsc::channel();

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .thread_name("vrc-mascot-tokio")
            .build()?;

        let ai_cfg = config.ai.clone();
        let voice_cfg = config.voice.clone();
        runtime.spawn(async move {
            run_background_worker(
                ai_cfg,
                voice_cfg,
                dialogue_endpoint_override,
                cmd_rx,
                event_tx,
            )
            .await;
        });

        let import_form = ImportFormState::from_config(&config);
        let status_window_visible = show_window_override || config.display.show_window_on_startup;

        Ok(Self {
            draft_config: config.clone(),
            config,
            registry,
            active_avatar_id,
            active_manifest_path,
            manifest: manifest.clone(),
            simulation,
            speech_player,
            audiolink_capture,
            mascot_state: MascotState::SeatedOnTaskbar,
            current_mood: EmotionTag::Neutral,
            chat_history: Vec::new(),
            chat_input: String::new(),
            chat_in_flight: false,
            decision_log: Vec::new(),
            decision_in_flight: false,
            voicevox_online: false,
            voicevox_version: None,
            voicevox_speakers: Vec::new(),
            voicevox_last_error: None,
            current_viseme: VisemeKey::Sil,
            current_viseme_weights: VisemeWeights::default(),
            fps_estimate: 60.0,
            import_status: ImportJobStatus::Idle,
            import_form,
            status_notice: None,
            pending_model_reload: Some((model, manifest)),
            pending_perch_action: None,
            menu_dirty: false,
            status_window_visible,
            cmd_tx,
            event_rx,
            _runtime: Some(runtime),
        })
    }

    /// Requests an autonomous decision evaluation on the background worker.
    pub fn trigger_decision(&mut self, ctx: DecisionContext) {
        if self.decision_in_flight {
            return;
        }
        self.decision_in_flight = true;
        let _ = self.cmd_tx.send(AppCommand::RunDecision(ctx));
    }

    /// Sends a user chat message to the background AI worker and records it in chat history.
    pub fn send_chat(&mut self, raw_text: String) {
        let trimmed = raw_text.trim().to_owned();
        if trimmed.is_empty() || self.chat_in_flight {
            return;
        }
        self.chat_history.push(ChatMessage {
            role: ChatRole::User,
            content: trimmed.clone(),
            emotion: None,
            timestamp_epoch_ms: now_epoch_ms(),
        });
        prune_chat_history(&mut self.chat_history, self.config.ai.max_history_turns);
        self.chat_in_flight = true;
        let _ = self.cmd_tx.send(AppCommand::SendChat(trimmed));
    }

    /// Requests a VOICEVOX Engine health & speaker list check.
    pub fn check_voicevox(&mut self) {
        let _ = self.cmd_tx.send(AppCommand::CheckVoicevox);
    }

    /// Triggers asynchronous Unity scene discovery inside `project_dir`.
    pub fn discover_scenes_in(&mut self, project_dir: PathBuf) {
        self.import_form
            .push_log(format!("シーン検索中: {}", project_dir.display()));
        let _ = self.cmd_tx.send(AppCommand::DiscoverScenes(project_dir));
    }

    /// Triggers asynchronous avatar candidate discovery in `scene_path`.
    pub fn discover_avatars_in(&mut self, project_dir: PathBuf, scene_path: PathBuf) {
        self.import_form
            .push_log(format!("アバター検索中: {}", scene_path.display()));
        let _ = self.cmd_tx.send(AppCommand::DiscoverAvatars {
            project_dir,
            scene_path,
        });
    }

    /// Starts background `vrc-extractor::extract_avatar` execution.
    pub fn start_import(&mut self, opts: ExtractOptions) {
        if matches!(self.import_status, ImportJobStatus::Running { .. }) {
            return;
        }
        self.import_status = ImportJobStatus::Running {
            progress: 0.0,
            stage: "Starting import...".to_owned(),
        };
        self.import_form.push_log(format!(
            "抽出開始: {} -> {}",
            opts.avatar_name,
            opts.output_dir.display()
        ));
        let _ = self.cmd_tx.send(AppCommand::RunImport(opts));
    }

    /// Applies `new_config` immediately to runtime systems and persists it to disk + registry.
    pub fn apply_and_save_config(&mut self, mut new_config: AppConfig) {
        new_config.display.scale = sanitize_scale(new_config.display.scale);
        new_config.display.max_fps = new_config.display.max_fps.clamp(15, 240);
        new_config.ai.decision_interval_sec = new_config.ai.decision_interval_sec.max(5);
        new_config.ai.max_history_turns = new_config.ai.max_history_turns.clamp(1, 200);

        let extract_dir_changed =
            new_config.resolved_extract_output_dir() != self.config.resolved_extract_output_dir();

        self.config = new_config.clone();
        self.draft_config = new_config.clone();

        self.speech_player
            .set_volume(self.config.voice.volume_scale);
        self.speech_player.set_muted(self.config.voice.muted);
        self.audiolink_capture
            .set_enabled(self.config.physics_shader.audiolink_enabled);

        if extract_dir_changed {
            let dir = self.config.resolved_extract_output_dir();
            self.registry =
                AvatarRegistry::scan(&dir).unwrap_or_else(|_| AvatarRegistry::new(&dir));
        }

        let _ = self.cmd_tx.send(AppCommand::UpdateAiVoiceConfig {
            ai: self.config.ai.clone(),
            voice: self.config.voice.clone(),
        });
        let _ = self
            .cmd_tx
            .send(AppCommand::SaveConfig(self.config.clone()));
        self.menu_dirty = true;
    }

    /// Switches the active avatar to `avatar_id` (`None` = built-in procedural fallback).
    pub fn switch_avatar(&mut self, avatar_id: Option<&str>) {
        match avatar_id {
            None => {
                let (model, manifest) = AvatarModel::procedural_fallback();
                self.install_loaded_avatar(model, manifest, None, None);
            }
            Some(id) => {
                if self.registry.find_by_id(id).is_none() {
                    let _ = self.registry.refresh();
                }
                let Some(entry) = self.registry.find_by_id(id).cloned() else {
                    self.status_notice = Some(format!("アバター '{id}' が見つかりません"));
                    return;
                };
                match AvatarModel::load_from_manifest(&entry.manifest_path) {
                    Ok((model, manifest)) => {
                        self.install_loaded_avatar(
                            model,
                            manifest,
                            Some(entry.avatar_id),
                            Some(entry.manifest_path),
                        );
                    }
                    Err(err) => {
                        self.status_notice =
                            Some(format!("アバター読込失敗 ({}): {err}", entry.avatar_name));
                    }
                }
            }
        }
    }

    fn install_loaded_avatar(
        &mut self,
        model: AvatarModel,
        manifest: MascotManifest,
        avatar_id: Option<String>,
        manifest_path: Option<PathBuf>,
    ) {
        let rest_nodes = skeleton_nodes_from_model(&model);
        self.simulation = MascotSimulation::new(&manifest, rest_nodes);
        self.manifest = manifest.clone();
        self.active_avatar_id = avatar_id;
        self.active_manifest_path = manifest_path;
        self.pending_model_reload = Some((model, manifest));
        self.menu_dirty = true;
    }

    /// Samples current lip-sync viseme weights (zeroed if lip-sync is disabled).
    pub fn sample_current_visemes(&mut self) -> VisemeWeights {
        if !self.config.voice.lip_sync_enabled {
            self.current_viseme = VisemeKey::Sil;
            self.current_viseme_weights = VisemeWeights::default();
            return VisemeWeights::default();
        }
        let (weights, key) = self
            .speech_player
            .sample_viseme(self.config.voice.lip_sync_sensitivity);
        self.current_viseme = key;
        self.current_viseme_weights = weights;
        weights
    }

    /// Drains pending background worker events and updates state synchronously.
    pub fn poll_events(&mut self) {
        while let Ok(ev) = self.event_rx.try_recv() {
            self.apply_event(ev);
        }
    }

    fn apply_event(&mut self, ev: AppEvent) {
        match ev {
            AppEvent::DecisionResult {
                output,
                timestamp_epoch_ms,
                error,
            } => {
                self.decision_in_flight = false;
                if let Some(err) = error {
                    self.status_notice = Some(format!("Decisions API エラー: {err}"));
                    return;
                }
                let reason = output
                    .reason
                    .clone()
                    .unwrap_or_else(|| "autonomous tick".to_owned());
                self.decision_log.push(DecisionLogEntry {
                    timestamp_epoch_ms,
                    action: output.action,
                    target_expression: output.target_expression.clone(),
                    speech_intent: output.speech_intent.clone(),
                    reason,
                });
                if self.decision_log.len() > 100 {
                    let excess = self.decision_log.len() - 100;
                    self.decision_log.drain(..excess);
                }

                match output.action {
                    DecisionAction::Speak => {
                        // Followed by AppEvent::AutonomousSpeak from worker.
                    }
                    DecisionAction::MoveToTaskbar => {
                        self.pending_perch_action = Some(PendingPerchAction::MoveToTaskbar);
                    }
                    DecisionAction::SitOnWindow => {
                        self.pending_perch_action = Some(PendingPerchAction::SitOnWindow);
                    }
                    DecisionAction::ChangeExpression => {
                        if let Some(expr) = output.target_expression.as_deref() {
                            apply_expression_or_emotion(
                                &mut self.simulation,
                                &mut self.current_mood,
                                expr,
                            );
                        }
                    }
                    DecisionAction::Idle => {
                        if let Some(expr) = output.target_expression.as_deref() {
                            if !expr.trim().is_empty() {
                                apply_expression_or_emotion(
                                    &mut self.simulation,
                                    &mut self.current_mood,
                                    expr,
                                );
                            }
                        }
                    }
                }
            }
            AppEvent::AutonomousSpeak {
                text,
                emotion,
                wav_bytes,
                timeline,
                timestamp_epoch_ms,
                voice_error,
            }
            | AppEvent::ChatReply {
                text,
                emotion,
                wav_bytes,
                timeline,
                timestamp_epoch_ms,
                voice_error,
            } => {
                self.chat_in_flight = false;
                self.chat_history.push(ChatMessage {
                    role: ChatRole::Assistant,
                    content: text,
                    emotion: Some(emotion.clone()),
                    timestamp_epoch_ms,
                });
                prune_chat_history(&mut self.chat_history, self.config.ai.max_history_turns);
                self.current_mood = emotion.clone();
                self.simulation.set_emotion(emotion);

                if let Some(err) = voice_error {
                    self.status_notice = Some(format!("音声合成エラー: {err}"));
                }
                if let (Some(wav), Some(tl)) = (wav_bytes, timeline) {
                    if !self.config.voice.muted {
                        if let Err(err) = self.speech_player.play(
                            wav,
                            tl,
                            self.config.voice.volume_scale,
                            self.config.voice.muted,
                        ) {
                            self.status_notice = Some(format!("音声再生エラー: {err}"));
                        }
                    }
                }
            }
            AppEvent::ChatError(err) => {
                self.chat_in_flight = false;
                self.status_notice = Some(format!("チャットエラー: {err}"));
            }
            AppEvent::VoicevoxStatus {
                online,
                version,
                speakers,
                error,
            } => {
                self.voicevox_online = online;
                self.voicevox_version = version;
                self.voicevox_speakers = speakers;
                self.voicevox_last_error = error;
            }
            AppEvent::ScenesDiscovered(res) => match res {
                Ok(scenes) => {
                    self.import_form
                        .push_log(format!("{} 件のシーンを検出", scenes.len()));
                    self.import_form.discovered_scenes = scenes;
                    self.import_form.selected_scene_idx = 0;
                    if let Some(first_scene) = self.import_form.discovered_scenes.first().cloned() {
                        let proj = PathBuf::from(self.import_form.project_dir.trim());
                        self.discover_avatars_in(proj, first_scene);
                    } else {
                        self.import_form.discovered_avatars.clear();
                    }
                }
                Err(err) => {
                    self.import_form.discovered_scenes.clear();
                    self.import_form.discovered_avatars.clear();
                    self.import_form.push_log(format!("シーン検出失敗: {err}"));
                }
            },
            AppEvent::AvatarsDiscovered(res) => match res {
                Ok(avatars) => {
                    self.import_form
                        .push_log(format!("{} 件のアバター候補を検出", avatars.len()));
                    self.import_form.discovered_avatars = avatars;
                    self.import_form.selected_avatar_idx = 0;
                    if let Some(first) = self.import_form.discovered_avatars.first() {
                        self.import_form.custom_avatar_name = first.name.clone();
                    }
                }
                Err(err) => {
                    self.import_form.discovered_avatars.clear();
                    self.import_form
                        .push_log(format!("アバター検出失敗: {err}"));
                }
            },
            AppEvent::ImportProgress(prog) => {
                self.import_form.push_log(format!(
                    "[{:>3}%] {}",
                    (prog.progress * 100.0).round() as i32,
                    prog.stage
                ));
                self.import_status = ImportJobStatus::Running {
                    progress: prog.progress,
                    stage: prog.stage,
                };
            }
            AppEvent::ImportFinished(res) => match res {
                Ok(report) => {
                    for w in &report.warnings {
                        self.import_form.push_log(format!("警告: {w}"));
                    }
                    let avatar_id = report
                        .manifest_path
                        .parent()
                        .and_then(|p| p.file_name())
                        .and_then(|n| n.to_str())
                        .map(str::to_owned)
                        .unwrap_or_else(|| sanitize_avatar_id(&report.avatar_name));

                    if let Err(err) = self
                        .registry
                        .register_manifest(&avatar_id, &report.manifest_path)
                    {
                        self.import_form
                            .push_log(format!("レジストリ登録エラー: {err}"));
                    } else {
                        let _ = self.registry.refresh();
                    }

                    self.import_form.push_log(format!(
                        "インポート完了: {} ({})",
                        report.avatar_name,
                        report.manifest_path.display()
                    ));
                    self.import_status = ImportJobStatus::Succeeded {
                        avatar_id: avatar_id.clone(),
                        manifest_path: report.manifest_path.clone(),
                    };

                    match AvatarModel::load_from_manifest(&report.manifest_path) {
                        Ok((model, manifest)) => {
                            self.install_loaded_avatar(
                                model,
                                manifest,
                                Some(avatar_id),
                                Some(report.manifest_path),
                            );
                        }
                        Err(err) => {
                            self.import_form
                                .push_log(format!("生成アバターの読込失敗: {err}"));
                        }
                    }
                }
                Err(err) => {
                    self.import_form.push_log(format!("インポート失敗: {err}"));
                    self.import_status = ImportJobStatus::Failed { error: err };
                }
            },
            AppEvent::ConfigSaved(res) => match res {
                Ok(()) => {
                    self.status_notice = Some("設定を保存して適用しました".to_owned());
                }
                Err(err) => {
                    self.status_notice = Some(format!("設定保存エラー: {err}"));
                }
            },
        }
    }
}

async fn run_background_worker(
    ai_cfg: AiConfig,
    voice_cfg: VoiceConfig,
    dialogue_endpoint_override: Option<String>,
    mut cmd_rx: tokio_mpsc::UnboundedReceiver<AppCommand>,
    event_tx: std_mpsc::Sender<AppEvent>,
) {
    let mut service = match AiService::new(ai_cfg, voice_cfg) {
        Ok(svc) => svc,
        Err(err) => {
            tracing::error!(error = %err, "Failed to initialize AiService");
            return;
        }
    };
    if let Some(ep) = dialogue_endpoint_override {
        service = service.with_dialogue_endpoint(ep);
    }

    let mut voicevox_online = false;

    while let Some(cmd) = cmd_rx.recv().await {
        match cmd {
            AppCommand::Shutdown => break,
            AppCommand::UpdateAiVoiceConfig { ai, voice } => {
                service.ai_config = ai;
                service.voice_config = voice;
            }
            AppCommand::CheckVoicevox => match service.check_voicevox().await {
                Ok(version) => {
                    let speakers = service.list_speakers().await.unwrap_or_default();
                    voicevox_online = true;
                    let _ = event_tx.send(AppEvent::VoicevoxStatus {
                        online: true,
                        version: Some(version),
                        speakers,
                        error: None,
                    });
                }
                Err(err) => {
                    voicevox_online = false;
                    let _ = event_tx.send(AppEvent::VoicevoxStatus {
                        online: false,
                        version: None,
                        speakers: Vec::new(),
                        error: Some(err.to_string()),
                    });
                }
            },
            AppCommand::RunDecision(ctx) => {
                let ts = now_epoch_ms();
                match service.decide(&ctx).await {
                    Ok(decision) => {
                        let _ = event_tx.send(AppEvent::DecisionResult {
                            output: decision.clone(),
                            timestamp_epoch_ms: ts,
                            error: None,
                        });
                        if decision.action == DecisionAction::Speak {
                            let intent = decision
                                .speech_intent
                                .as_deref()
                                .unwrap_or("ユーザーに優しく一言話しかける");
                            match service.speak_from_intent(intent).await {
                                Ok(reply) => {
                                    let (wav_bytes, timeline, voice_error) = maybe_synthesize(
                                        &service,
                                        &mut voicevox_online,
                                        &reply.reply_text,
                                    )
                                    .await;
                                    let _ = event_tx.send(AppEvent::AutonomousSpeak {
                                        text: reply.reply_text,
                                        emotion: reply.emotion,
                                        wav_bytes,
                                        timeline,
                                        timestamp_epoch_ms: now_epoch_ms(),
                                        voice_error,
                                    });
                                }
                                Err(err) => {
                                    let _ = event_tx.send(AppEvent::ChatError(err.to_string()));
                                }
                            }
                        }
                    }
                    Err(err) => {
                        let _ = event_tx.send(AppEvent::DecisionResult {
                            output: DecisionOutput::default(),
                            timestamp_epoch_ms: ts,
                            error: Some(err.to_string()),
                        });
                    }
                }
            }
            AppCommand::SendChat(text) => match service.chat(&text).await {
                Ok(reply) => {
                    let (wav_bytes, timeline, voice_error) =
                        maybe_synthesize(&service, &mut voicevox_online, &reply.reply_text).await;
                    let _ = event_tx.send(AppEvent::ChatReply {
                        text: reply.reply_text,
                        emotion: reply.emotion,
                        wav_bytes,
                        timeline,
                        timestamp_epoch_ms: now_epoch_ms(),
                        voice_error,
                    });
                }
                Err(err) => {
                    let _ = event_tx.send(AppEvent::ChatError(err.to_string()));
                }
            },
            AppCommand::DiscoverScenes(project_dir) => {
                let res = tokio::task::spawn_blocking(move || {
                    discover_scenes(&project_dir).map_err(|e| e.to_string())
                })
                .await
                .unwrap_or_else(|e| Err(e.to_string()));
                let _ = event_tx.send(AppEvent::ScenesDiscovered(res));
            }
            AppCommand::DiscoverAvatars {
                project_dir,
                scene_path,
            } => {
                let res = tokio::task::spawn_blocking(move || {
                    discover_avatars(&project_dir, &scene_path).map_err(|e| e.to_string())
                })
                .await
                .unwrap_or_else(|e| Err(e.to_string()));
                let _ = event_tx.send(AppEvent::AvatarsDiscovered(res));
            }
            AppCommand::RunImport(opts) => {
                let tx_prog = event_tx.clone();
                let res = tokio::task::spawn_blocking(move || {
                    extract_avatar(&opts, |prog| {
                        let _ = tx_prog.send(AppEvent::ImportProgress(prog));
                    })
                    .map_err(|e| e.to_string())
                })
                .await
                .unwrap_or_else(|e| Err(e.to_string()));
                let _ = event_tx.send(AppEvent::ImportFinished(res));
            }
            AppCommand::SaveConfig(cfg) => {
                let res = tokio::task::spawn_blocking(move || -> Result<(), String> {
                    cfg.save().map_err(|e| e.to_string())?;
                    win32::set_auto_start(cfg.display.auto_start).map_err(|e| e.to_string())?;
                    Ok(())
                })
                .await
                .unwrap_or_else(|e| Err(e.to_string()));
                let _ = event_tx.send(AppEvent::ConfigSaved(res));
            }
        }
    }
}

async fn maybe_synthesize(
    service: &AiService,
    voicevox_online: &mut bool,
    text: &str,
) -> (Option<Vec<u8>>, Option<LipSyncTimeline>, Option<String>) {
    if service.voice_config.muted || !*voicevox_online {
        return (None, None, None);
    }
    match service.synthesize(text).await {
        Ok(synth) => (Some(synth.wav_bytes), Some(synth.timeline), None),
        Err(err) => {
            *voicevox_online = false;
            (None, None, Some(err.to_string()))
        }
    }
}

/// Converts [`AvatarModel::nodes`] into [`SkeletonNode`] list for [`MascotSimulation::new`].
pub fn skeleton_nodes_from_model(model: &AvatarModel) -> Vec<SkeletonNode> {
    model
        .nodes
        .iter()
        .map(|n| {
            SkeletonNode::new(
                n.name.clone(),
                n.parent,
                n.children.clone(),
                n.rest_translation,
                n.rest_rotation,
                n.rest_scale,
            )
        })
        .collect()
}

/// Loads the startup avatar following the priority chain:
/// `--manifest` -> `default_avatar_id` in registry -> first registered avatar -> procedural fallback.
pub fn load_initial_avatar(
    config: &AppConfig,
    registry: &AvatarRegistry,
    cli_manifest: Option<&Path>,
) -> (AvatarModel, MascotManifest, Option<String>, Option<PathBuf>) {
    if let Some(path) = cli_manifest {
        match AvatarModel::load_from_manifest(path) {
            Ok((model, manifest)) => {
                let matched_id = registry
                    .entries()
                    .iter()
                    .find(|e| e.manifest_path == path)
                    .map(|e| e.avatar_id.clone());
                return (model, manifest, matched_id, Some(path.to_path_buf()));
            }
            Err(err) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %err,
                    "Failed to load --manifest avatar; falling back to registry"
                );
            }
        }
    }

    if let Some(entry) = registry.default_or_first(&config.avatar_import.default_avatar_id) {
        match AvatarModel::load_from_manifest(&entry.manifest_path) {
            Ok((model, manifest)) => {
                return (
                    model,
                    manifest,
                    Some(entry.avatar_id.clone()),
                    Some(entry.manifest_path.clone()),
                );
            }
            Err(err) => {
                tracing::warn!(
                    avatar_id = %entry.avatar_id,
                    error = %err,
                    "Failed to load registered avatar; falling back to procedural model"
                );
            }
        }
    }

    let (model, manifest) = AvatarModel::procedural_fallback();
    (model, manifest, None, None)
}

/// Sanitizes an avatar name into a filesystem-safe directory identifier.
pub fn sanitize_avatar_id(name: &str) -> String {
    let cleaned: String = name
        .trim()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if cleaned.trim_matches('_').is_empty() {
        "imported_avatar".to_owned()
    } else {
        cleaned
    }
}

fn apply_expression_or_emotion(
    simulation: &mut MascotSimulation,
    current_mood: &mut EmotionTag,
    expr: &str,
) {
    let triggered = simulation.trigger_menu_item(expr);
    let lower = expr.trim().to_ascii_lowercase();
    let tag = match lower.as_str() {
        "neutral" | "normal" | "通常" => EmotionTag::Neutral,
        "happy" | "smile" | "joy" | "笑顔" => EmotionTag::Happy,
        "sad" | "sorrow" | "悲しい" => EmotionTag::Sad,
        "angry" | "mad" | "怒り" => EmotionTag::Angry,
        "surprised" | "surprise" | "驚き" => EmotionTag::Surprised,
        "shy" | "embarrassed" | "照れ" => EmotionTag::Shy,
        "sleepy" | "tired" | "眠い" => EmotionTag::Sleepy,
        _ => {
            if triggered {
                return;
            }
            EmotionTag::Custom(expr.trim().to_owned())
        }
    };
    *current_mood = tag.clone();
    simulation.set_emotion(tag);
}

fn prune_chat_history(history: &mut Vec<ChatMessage>, max_turns: usize) {
    let max_messages = max_turns.max(1).saturating_mul(2);
    if history.len() > max_messages {
        let excess = history.len() - max_messages;
        history.drain(..excess);
    }
}

fn sanitize_scale(scale: f32) -> f32 {
    if scale.is_finite() {
        scale.clamp(0.5, 2.0)
    } else {
        1.0
    }
}

pub fn now_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Returns coarse time-of-day label (`"morning"`, `"afternoon"`, `"evening"`, `"night"`).
pub fn current_time_of_day_label() -> &'static str {
    // Use Windows local-ish offset approximation or UTC hour bucket cleanly without extra deps.
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Approximate JST (+9h) or local day bucket
    let hour = ((secs / 3600) + 9) % 24;
    match hour {
        5..=10 => "morning",
        11..=16 => "afternoon",
        17..=21 => "evening",
        _ => "night",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;
    use std::time::{Duration, Instant};

    fn spawn_mock_http_server<F>(handler: F) -> (String, thread::JoinHandle<()>)
    where
        F: Fn(&str) -> (u16, &'static str, Vec<u8>) + Send + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock listener");
        let addr = listener.local_addr().expect("local addr");
        let base_url = format!("http://{addr}");

        let handle = thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let mut stream = stream;
                let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
                let mut buf = [0u8; 8192];
                let n = stream.read(&mut buf).unwrap_or(0);
                if n == 0 {
                    continue;
                }
                let req_str = String::from_utf8_lossy(&buf[..n]);
                let target = req_str
                    .lines()
                    .next()
                    .and_then(|line| line.split_whitespace().nth(1))
                    .unwrap_or("/");
                if target == "/__shutdown" {
                    let _ = stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n");
                    break;
                }
                let (status, ctype, body) = handler(target);
                let header = format!(
                    "HTTP/1.1 {status} OK\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(header.as_bytes());
                let _ = stream.write_all(&body);
            }
        });

        (base_url, handle)
    }

    fn silent_wav_bytes() -> Vec<u8> {
        let mut buf = Vec::with_capacity(44);
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&36_u32.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16_u32.to_le_bytes());
        buf.extend_from_slice(&1_u16.to_le_bytes());
        buf.extend_from_slice(&1_u16.to_le_bytes());
        buf.extend_from_slice(&24_000_u32.to_le_bytes());
        buf.extend_from_slice(&48_000_u32.to_le_bytes());
        buf.extend_from_slice(&2_u16.to_le_bytes());
        buf.extend_from_slice(&16_u16.to_le_bytes());
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&0_u32.to_le_bytes());
        buf
    }

    fn wait_until<F: FnMut(&mut AppState) -> bool>(state: &mut AppState, mut pred: F) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            state.poll_events();
            if pred(state) {
                return;
            }
            thread::sleep(Duration::from_millis(10));
        }
        panic!("timed out waiting for AppState condition");
    }

    #[test]
    fn app_state_runs_voicevox_check_decision_and_chat_with_mock_server() {
        let wav = silent_wav_bytes();
        let (base_url, server_thread) = spawn_mock_http_server(move |target| {
            if target == "/version" {
                (200, "application/json", br#""0.15.0""#.to_vec())
            } else if target == "/speakers" {
                (
                    200,
                    "application/json",
                    br#"[{"name":"TestSpeaker","styles":[{"name":"Normal","id":1}]}]"#.to_vec(),
                )
            } else if target.starts_with("/audio_query") {
                let aq = r#"{"accent_phrases":[{"moras":[{"text":"ア","vowel":"a","vowel_length":0.2,"pitch":5.0}],"accent":1}],"speedScale":1.0,"pitchScale":0.0,"intonationScale":1.0,"volumeScale":1.0,"prePhonemeLength":0.0,"postPhonemeLength":0.0}"#;
                (200, "application/json", aq.as_bytes().to_vec())
            } else if target.starts_with("/synthesis") {
                (200, "audio/wav", wav.clone())
            } else if target.contains("gemini-2.5-flash-lite") {
                let dec = r#"{"candidates":[{"content":{"parts":[{"text":"{\"action\":\"SIT_ON_WINDOW\",\"target_expression\":\"Smile\",\"reason\":\"focusing\"}"}]}}]}"#;
                (200, "application/json", dec.as_bytes().to_vec())
            } else {
                let chat = r#"{"candidates":[{"content":{"parts":[{"text":"{\"reply_text\":\"こんにちは！\",\"emotion\":\"happy\"}"}]}}]}"#;
                (200, "application/json", chat.as_bytes().to_vec())
            }
        });

        let mut cfg = AppConfig::default();
        cfg.ai.chat_api_key = "test-key".to_owned();
        cfg.ai.decisions_endpoint = base_url.clone();
        cfg.voice.voicevox_url = base_url.clone();

        let (model, manifest) = AvatarModel::procedural_fallback();
        let mut state = AppState::build_with_parts(
            cfg,
            AvatarRegistry::new(std::env::temp_dir()),
            model,
            manifest,
            None,
            None,
            SpeechPlayer::new_headless(),
            AudioLinkCapture::new(false),
            false,
            Some(base_url.clone()),
        )
        .expect("build AppState");

        state.check_voicevox();
        wait_until(&mut state, |s| s.voicevox_online);
        assert_eq!(state.voicevox_version.as_deref(), Some("0.15.0"));
        assert_eq!(state.voicevox_speakers.len(), 1);

        // Non-SPEAK decision skips LLM & TTS and emits pending SitOnWindow action
        state.trigger_decision(DecisionContext::default());
        wait_until(&mut state, |s| {
            !s.decision_in_flight && !s.decision_log.is_empty()
        });
        assert_eq!(state.decision_log[0].action, DecisionAction::SitOnWindow);
        assert_eq!(
            state.pending_perch_action,
            Some(PendingPerchAction::SitOnWindow)
        );

        // Chat triggers Dialogue LLM -> VOICEVOX synthesis -> SpeechPlayer lip-sync
        state.send_chat("おはよう".to_owned());
        wait_until(&mut state, |s| {
            !s.chat_in_flight && s.chat_history.len() == 2
        });
        assert_eq!(state.chat_history[1].content, "こんにちは！");
        assert_eq!(state.current_mood, EmotionTag::Happy);
        assert!(state.speech_player.is_playing());

        // ChangeExpression decision event updates current_mood without LLM/TTS
        state.apply_event(AppEvent::DecisionResult {
            output: DecisionOutput {
                action: DecisionAction::ChangeExpression,
                target_expression: Some("Surprised".to_owned()),
                speech_intent: None,
                reason: Some("test".to_owned()),
            },
            timestamp_epoch_ms: 1000,
            error: None,
        });
        assert_eq!(state.current_mood, EmotionTag::Surprised);

        drop(state);
        let _ = std::net::TcpStream::connect(base_url.trim_start_matches("http://"))
            .and_then(|mut s| s.write_all(b"GET /__shutdown HTTP/1.1\r\n\r\n"));
        let _ = server_thread.join();
    }

    #[test]
    fn sanitize_avatar_id_and_fallback_loader() {
        assert_eq!(sanitize_avatar_id("My Avatar!"), "My_Avatar_");
        assert_eq!(sanitize_avatar_id("   "), "imported_avatar");

        let cfg = AppConfig::default();
        let reg = AvatarRegistry::new(std::env::temp_dir().join("nonexistent_vrc_mascot_avatars"));
        let (model, manifest, id, path) = load_initial_avatar(&cfg, &reg, None);
        assert!(!model.nodes.is_empty());
        assert!(!manifest.avatar_name.is_empty());
        assert!(id.is_none());
        assert!(path.is_none());
    }

    #[test]
    fn app_state_speak_decision_triggers_dialogue_and_voicevox_lipsync() {
        let wav = silent_wav_bytes();
        let (base_url, server_thread) = spawn_mock_http_server(move |target| {
            if target == "/version" {
                (200, "application/json", br#""0.15.0""#.to_vec())
            } else if target == "/speakers" {
                (200, "application/json", b"[]".to_vec())
            } else if target.starts_with("/audio_query") {
                let aq = r#"{"accent_phrases":[{"moras":[{"text":"オ","vowel":"o","vowel_length":0.25,"pitch":5.0}],"accent":1}],"speedScale":1.0,"pitchScale":0.0,"intonationScale":1.0,"volumeScale":1.0,"prePhonemeLength":0.0,"postPhonemeLength":0.0}"#;
                (200, "application/json", aq.as_bytes().to_vec())
            } else if target.starts_with("/synthesis") {
                (200, "audio/wav", wav.clone())
            } else if target.contains("gemini-2.5-flash-lite") {
                let dec = r#"{"candidates":[{"content":{"parts":[{"text":"{\"action\":\"SPEAK\",\"target_expression\":\"Smile\",\"speech_intent\":\"greet\",\"reason\":\"break time\"}"}]}}]}"#;
                (200, "application/json", dec.as_bytes().to_vec())
            } else {
                let chat = r#"{"candidates":[{"content":{"parts":[{"text":"{\"reply_text\":\"お疲れさま！\",\"emotion\":\"happy\"}"}]}}]}"#;
                (200, "application/json", chat.as_bytes().to_vec())
            }
        });

        let mut cfg = AppConfig::default();
        cfg.ai.chat_api_key = "test-key".to_owned();
        cfg.ai.decisions_endpoint = base_url.clone();
        cfg.voice.voicevox_url = base_url.clone();

        let (model, manifest) = AvatarModel::procedural_fallback();
        let mut state = AppState::build_with_parts(
            cfg,
            AvatarRegistry::new(std::env::temp_dir()),
            model,
            manifest,
            None,
            None,
            SpeechPlayer::new_headless(),
            AudioLinkCapture::new(false),
            false,
            Some(base_url.clone()),
        )
        .expect("build AppState");

        state.check_voicevox();
        wait_until(&mut state, |s| s.voicevox_online);

        state.trigger_decision(DecisionContext::default());
        wait_until(&mut state, |s| {
            !s.decision_in_flight && !s.chat_history.is_empty()
        });
        assert_eq!(state.decision_log[0].action, DecisionAction::Speak);
        assert_eq!(state.chat_history[0].role, ChatRole::Assistant);
        assert_eq!(state.chat_history[0].content, "お疲れさま！");
        assert_eq!(state.current_mood, EmotionTag::Happy);
        assert!(state.speech_player.is_playing());
        let weights = state.sample_current_visemes();
        assert!(weights.oh >= 0.0);

        // MoveToTaskbar event queues PendingPerchAction::MoveToTaskbar
        state.apply_event(AppEvent::DecisionResult {
            output: DecisionOutput {
                action: DecisionAction::MoveToTaskbar,
                target_expression: None,
                speech_intent: None,
                reason: Some("rest".to_owned()),
            },
            timestamp_epoch_ms: 2000,
            error: None,
        });
        assert_eq!(
            state.pending_perch_action,
            Some(PendingPerchAction::MoveToTaskbar)
        );

        drop(state);
        let _ = std::net::TcpStream::connect(base_url.trim_start_matches("http://"))
            .and_then(|mut s| s.write_all(b"GET /__shutdown HTTP/1.1\r\n\r\n"));
        let _ = server_thread.join();
    }
}
