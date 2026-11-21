//! `egui` + `egui-wgpu` + `egui-winit` Status & Settings window UI (4 tabs).

use std::path::{Path, PathBuf};

use vrc_extractor::ExtractOptions;
use vrc_physics::ParamValue;
use vrc_schema::{
    AppConfig, ChatRole, DecisionAction, EmotionTag, ExpressionMenuItem, ExpressionMenuItemType,
    ImportJobStatus, MascotState, QualityPreset,
};

use crate::app_state::{sanitize_avatar_id, AppState, FolderPickerTarget};

/// Active tab in the Status & Settings window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UiTab {
    #[default]
    StatusDialogue,
    AvatarGimmick,
    Import,
    Settings,
}

/// Immediate side effects requested by UI interactions for `window_manager` to apply.
#[derive(Debug, Clone, Default)]
pub struct StatusUiEffects {
    pub trigger_decision: bool,
    pub reset_position: bool,
    pub config_applied: bool,
}

/// Persistent UI-only widget state (selected tab, API key reveal toggles, exclude list draft).
#[derive(Debug, Clone, Default)]
pub struct StatusUiState {
    pub active_tab: UiTab,
    pub reveal_chat_api_key: bool,
    pub reveal_decisions_api_key: bool,
    pub exclude_apps_buf: String,
    pub exclude_apps_synced: bool,
}

impl StatusUiState {
    /// Installs Windows Japanese fonts (Yu Gothic / Meiryo / MS Gothic) if available on disk.
    pub fn configure_fonts(ctx: &egui::Context) {
        let mut fonts = egui::FontDefinitions::default();
        let candidates = [
            r"C:\Windows\Fonts\YuGothM.ttc",
            r"C:\Windows\Fonts\YuGothR.ttc",
            r"C:\Windows\Fonts\meiryo.ttc",
            r"C:\Windows\Fonts\msgothic.ttc",
        ];

        for path_str in candidates {
            let path = Path::new(path_str);
            if path.is_file() {
                if let Ok(bytes) = std::fs::read(path) {
                    let font_name = "windows_jp".to_owned();
                    fonts
                        .font_data
                        .insert(font_name.clone(), egui::FontData::from_owned(bytes).into());
                    if let Some(family) = fonts.families.get_mut(&egui::FontFamily::Proportional) {
                        family.insert(0, font_name.clone());
                    }
                    if let Some(family) = fonts.families.get_mut(&egui::FontFamily::Monospace) {
                        family.push(font_name);
                    }
                    ctx.set_fonts(fonts);
                    return;
                }
            }
        }
    }

    /// Renders the entire Status & Settings window UI and returns immediate window-manager effects.
    pub fn draw(&mut self, ui: &mut egui::Ui, app: &mut AppState) -> StatusUiEffects {
        let mut effects = StatusUiEffects::default();
        if !self.exclude_apps_synced {
            self.exclude_apps_buf = app.draft_config.motion.perch_exclude_apps.join(", ");
            self.exclude_apps_synced = true;
        }

        egui::Panel::top("vrc_mascot_top_tabs").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading("VRC Desktop Mascot");
                ui.separator();
                ui.selectable_value(
                    &mut self.active_tab,
                    UiTab::StatusDialogue,
                    "ステータス・対話",
                );
                ui.selectable_value(
                    &mut self.active_tab,
                    UiTab::AvatarGimmick,
                    "アバター・ギミック",
                );
                ui.selectable_value(&mut self.active_tab, UiTab::Import, "インポート");
                ui.selectable_value(&mut self.active_tab, UiTab::Settings, "設定");
            });

            if let Some(notice) = app.status_notice.clone() {
                ui.horizontal(|ui| {
                    ui.colored_label(egui::Color32::from_rgb(240, 190, 90), &notice);
                    if ui.small_button("閉じる").clicked() {
                        app.status_notice = None;
                    }
                });
            }
        });

        egui::CentralPanel::default().show(ui, |ui| match self.active_tab {
            UiTab::StatusDialogue => self.draw_status_dialogue_tab(ui, app, &mut effects),
            UiTab::AvatarGimmick => self.draw_avatar_gimmick_tab(ui, app),
            UiTab::Import => self.draw_import_tab(ui, app),
            UiTab::Settings => self.draw_settings_tab(ui, app, &mut effects),
        });

        effects
    }

    fn draw_status_dialogue_tab(
        &mut self,
        ui: &mut egui::Ui,
        app: &mut AppState,
        effects: &mut StatusUiEffects,
    ) {
        // Top status summary bar
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(format!("アバター: {}", app.manifest.avatar_name));
                ui.separator();
                ui.label(format!("姿勢: {}", format_mascot_state(app.mascot_state)));
                ui.separator();
                ui.label(format!("感情: {}", format_emotion(&app.current_mood)));
                ui.separator();
                ui.label(format!("FPS: {:.1}", app.fps_estimate));
                ui.separator();

                let (vv_color, vv_text) = if app.voicevox_online {
                    let ver = app.voicevox_version.as_deref().unwrap_or("?");
                    (
                        egui::Color32::from_rgb(90, 210, 120),
                        format!("VOICEVOX: オンライン (v{ver})"),
                    )
                } else {
                    (
                        egui::Color32::from_rgb(230, 100, 100),
                        "VOICEVOX: オフライン".to_owned(),
                    )
                };
                ui.colored_label(vv_color, vv_text);
                if ui.button("接続テスト").clicked() {
                    app.check_voicevox();
                }
            });

            // Viseme bar indicator
            ui.horizontal(|ui| {
                ui.label(format!("Viseme [{:?}]:", app.current_viseme));
                let w = app.current_viseme_weights;
                for (label, val) in [
                    ("aa", w.aa),
                    ("ih", w.ih),
                    ("ou", w.ou),
                    ("ee", w.ee),
                    ("oh", w.oh),
                ] {
                    ui.label(label);
                    ui.add(
                        egui::ProgressBar::new(val.clamp(0.0, 1.0))
                            .desired_width(56.0)
                            .show_percentage(),
                    );
                }
            });
        });

        ui.add_space(6.0);

        let avail = ui.available_size();
        ui.columns(2, |cols| {
            // Left column: Chat history & input
            cols[0].group(|ui| {
                ui.set_min_height((avail.y - 24.0).max(260.0));
                ui.horizontal(|ui| {
                    ui.heading("対話チャット");
                    if ui.small_button("履歴クリア").clicked() {
                        app.clear_chat_history();
                    }
                });
                ui.separator();

                let scroll_h = (ui.available_height() - 42.0).max(140.0);
                egui::ScrollArea::vertical()
                    .id_salt("chat_history_scroll")
                    .max_height(scroll_h)
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        if app.chat_history.is_empty() {
                            ui.weak("まだ会話履歴がありません。下の入力欄から話しかけてください。");
                        }
                        for msg in &app.chat_history {
                            let (prefix, color) = match msg.role {
                                ChatRole::User => {
                                    ("あなた", egui::Color32::from_rgb(130, 195, 255))
                                }
                                ChatRole::Assistant => (
                                    app.manifest.avatar_name.as_str(),
                                    egui::Color32::from_rgb(255, 180, 210),
                                ),
                                ChatRole::System => ("System", egui::Color32::GRAY),
                            };
                            ui.horizontal_wrapped(|ui| {
                                ui.colored_label(color, format!("[{prefix}]"));
                                if let Some(em) = &msg.emotion {
                                    ui.weak(format!("({})", format_emotion(em)));
                                }
                                ui.label(&msg.content);
                            });
                            ui.add_space(2.0);
                        }
                    });

                ui.separator();
                ui.horizontal(|ui| {
                    let input_resp = ui.add_sized(
                        [ui.available_width() - 68.0, 24.0],
                        egui::TextEdit::singleline(&mut app.chat_input)
                            .hint_text("メッセージを入力..."),
                    );
                    let enter_pressed =
                        input_resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    let send_enabled = !app.chat_in_flight && !app.chat_input.trim().is_empty();
                    let send_clicked = ui
                        .add_enabled(send_enabled, egui::Button::new("送信"))
                        .clicked();
                    if (send_clicked || enter_pressed) && send_enabled {
                        let text = std::mem::take(&mut app.chat_input);
                        app.send_chat(text);
                    }
                });
            });

            // Right column: Decisions AI log
            cols[1].group(|ui| {
                ui.set_min_height((avail.y - 24.0).max(260.0));
                ui.horizontal(|ui| {
                    ui.heading("自律行動判断ログ");
                    if ui
                        .add_enabled(!app.decision_in_flight, egui::Button::new("今すぐ判断実行"))
                        .clicked()
                    {
                        effects.trigger_decision = true;
                    }
                    if ui.button("位置リセット").clicked() {
                        effects.reset_position = true;
                    }
                });
                ui.separator();

                egui::ScrollArea::vertical()
                    .id_salt("decisions_log_scroll")
                    .max_height((ui.available_height() - 12.0).max(160.0))
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        if app.decision_log.is_empty() {
                            ui.weak("自律判断ログはまだありません。");
                        }
                        for entry in &app.decision_log {
                            ui.horizontal_wrapped(|ui| {
                                ui.monospace(format_epoch_hms(entry.timestamp_epoch_ms));
                                ui.strong(format_decision_action(entry.action));
                                if let Some(expr) = &entry.target_expression {
                                    ui.weak(format!("[{expr}]"));
                                }
                            });
                            ui.label(format!("  理由: {}", entry.reason));
                            if let Some(intent) = &entry.speech_intent {
                                ui.weak(format!("  発話意図: {intent}"));
                            }
                            ui.separator();
                        }
                    });
            });
        });
    }

    fn draw_avatar_gimmick_tab(&mut self, ui: &mut egui::Ui, app: &mut AppState) {
        ui.columns(2, |cols| {
            // Left: Registered avatars list
            cols[0].group(|ui| {
                ui.horizontal(|ui| {
                    ui.heading("登録アバター一覧");
                    if ui.button("再スキャン").clicked() {
                        let _ = app.registry.refresh();
                        app.menu_dirty = true;
                    }
                });
                ui.weak(format!(
                    "ディレクトリ: {}",
                    app.registry.root_dir().display()
                ));
                ui.separator();

                let is_fallback = app.active_avatar_id.is_none();
                if ui
                    .selectable_label(is_fallback, "ビルトイン (フォールバックモデル)")
                    .clicked()
                {
                    app.switch_avatar(None);
                }
                ui.separator();

                let entries = app.registry.entries().to_vec();
                egui::ScrollArea::vertical()
                    .id_salt("avatar_registry_scroll")
                    .max_height(460.0)
                    .show(ui, |ui| {
                        if entries.is_empty() {
                            ui.weak("抽出済みアバターはまだありません。「インポート」タブから追加できます。");
                        }
                        for entry in &entries {
                            let selected =
                                app.active_avatar_id.as_deref() == Some(entry.avatar_id.as_str());
                            if ui
                                .selectable_label(
                                    selected,
                                    format!("{} ({})", entry.avatar_name, entry.avatar_id),
                                )
                                .clicked()
                            {
                                app.switch_avatar(Some(&entry.avatar_id));
                            }
                        }
                    });
            });

            // Right: VRCExpressionsMenu & live FX parameters
            cols[1].group(|ui| {
                ui.heading("Expressions メニュー & FX パラメータ");
                ui.separator();

                let menu_items = app.manifest.expressions.menu.clone();
                egui::ScrollArea::vertical()
                    .id_salt("expressions_menu_scroll")
                    .max_height(300.0)
                    .show(ui, |ui| {
                        if menu_items.is_empty() {
                            ui.weak("VRCExpressionsMenu 項目がありません。");
                        } else {
                            draw_expression_menu_items(ui, &menu_items, app);
                        }
                    });

                ui.separator();
                ui.strong("FX パラメータ現在値");
                egui::ScrollArea::vertical()
                    .id_salt("fx_params_scroll")
                    .max_height(180.0)
                    .show(ui, |ui| {
                        let mut params: Vec<(String, ParamValue)> = app
                            .simulation
                            .expressions_state()
                            .all_params()
                            .iter()
                            .map(|(k, v)| (k.clone(), *v))
                            .collect();
                        params.sort_by(|a, b| a.0.cmp(&b.0));
                        if params.is_empty() {
                            ui.weak("(パラメータなし)");
                        }
                        for (name, val) in params {
                            ui.horizontal(|ui| {
                                ui.monospace(&name);
                                match val {
                                    ParamValue::Bool(mut b) => {
                                        if ui.checkbox(&mut b, "").changed() {
                                            app.simulation.set_param_bool(&name, b);
                                        }
                                    }
                                    ParamValue::Int(mut i) => {
                                        if ui.add(egui::DragValue::new(&mut i)).changed() {
                                            app.simulation.set_param_int(&name, i);
                                        }
                                    }
                                    ParamValue::Float(mut f) => {
                                        if ui
                                            .add(egui::Slider::new(&mut f, 0.0..=1.0))
                                            .changed()
                                        {
                                            app.simulation.set_param_float(&name, f);
                                        }
                                    }
                                }
                            });
                        }
                    });
            });
        });
    }

    fn draw_import_tab(&mut self, ui: &mut egui::Ui, app: &mut AppState) {
        let is_running = matches!(app.import_status, ImportJobStatus::Running { .. });

        ui.group(|ui| {
            ui.heading("Unity プロジェクト外部解析・インポート");
            ui.separator();

            ui.horizontal(|ui| {
                ui.label("Unity プロジェクト:");
                ui.add(
                    egui::TextEdit::singleline(&mut app.import_form.project_dir)
                        .desired_width(420.0)
                        .hint_text("C:\\Path\\To\\UnityProject"),
                );
                if ui
                    .add_enabled(
                        !app.folder_picker_open,
                        egui::Button::new("フォルダ選択..."),
                    )
                    .clicked()
                {
                    let trimmed = app.import_form.project_dir.trim();
                    let initial = (!trimmed.is_empty()).then(|| PathBuf::from(trimmed));
                    app.open_folder_picker(FolderPickerTarget::ImportProjectDir, initial);
                }
                if ui.button("スキャン").clicked() && !app.import_form.project_dir.trim().is_empty()
                {
                    let dir = PathBuf::from(app.import_form.project_dir.trim());
                    app.discover_scenes_in(dir);
                }
            });

            ui.add_space(4.0);

            // Scene selector
            ui.horizontal(|ui| {
                ui.label("シーン (.unity):");
                let selected_scene_text = app
                    .import_form
                    .discovered_scenes
                    .get(app.import_form.selected_scene_idx)
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "(未検出)".to_owned());

                let mut scene_changed = None;
                egui::ComboBox::from_id_salt("import_scene_combo")
                    .selected_text(&selected_scene_text)
                    .width(380.0)
                    .show_ui(ui, |ui| {
                        for (idx, scene) in app.import_form.discovered_scenes.iter().enumerate() {
                            let label = scene.display().to_string();
                            if ui
                                .selectable_value(
                                    &mut app.import_form.selected_scene_idx,
                                    idx,
                                    label,
                                )
                                .clicked()
                            {
                                scene_changed = Some(scene.clone());
                            }
                        }
                    });
                if let Some(scene) = scene_changed {
                    let dir = PathBuf::from(app.import_form.project_dir.trim());
                    app.discover_avatars_in(dir, scene);
                }
            });

            // Avatar selector
            ui.horizontal(|ui| {
                ui.label("対象アバター:");
                let selected_avatar_text = app
                    .import_form
                    .discovered_avatars
                    .get(app.import_form.selected_avatar_idx)
                    .map(|a| {
                        if a.has_descriptor {
                            format!("{} [VRCAvatarDescriptor]", a.name)
                        } else {
                            a.name.clone()
                        }
                    })
                    .unwrap_or_else(|| "(未検出)".to_owned());

                egui::ComboBox::from_id_salt("import_avatar_combo")
                    .selected_text(&selected_avatar_text)
                    .width(260.0)
                    .show_ui(ui, |ui| {
                        for (idx, av) in app.import_form.discovered_avatars.iter().enumerate() {
                            let label = if av.has_descriptor {
                                format!("{} [Descriptor]", av.name)
                            } else {
                                av.name.clone()
                            };
                            if ui
                                .selectable_value(
                                    &mut app.import_form.selected_avatar_idx,
                                    idx,
                                    label,
                                )
                                .clicked()
                            {
                                app.import_form.custom_avatar_name = av.name.clone();
                            }
                        }
                    });

                ui.label("出力名:");
                ui.add(
                    egui::TextEdit::singleline(&mut app.import_form.custom_avatar_name)
                        .desired_width(180.0),
                );
            });

            ui.horizontal(|ui| {
                ui.checkbox(
                    &mut app.import_form.modular_avatar_enabled,
                    "Modular Avatar 非破壊統合を有効にする",
                );
                ui.separator();
                ui.label("最大テクスチャサイズ:");
                egui::ComboBox::from_id_salt("import_max_tex")
                    .selected_text(format!("{} px", app.import_form.max_texture_size))
                    .show_ui(ui, |ui| {
                        for sz in [512_u32, 1024, 2048, 4096] {
                            ui.selectable_value(
                                &mut app.import_form.max_texture_size,
                                sz,
                                format!("{sz} px"),
                            );
                        }
                    });
            });

            ui.add_space(4.0);
            let can_start = !is_running
                && !app.import_form.project_dir.trim().is_empty()
                && !app.import_form.discovered_scenes.is_empty()
                && !app.import_form.custom_avatar_name.trim().is_empty();

            if ui
                .add_enabled(can_start, egui::Button::new("インポート開始"))
                .clicked()
            {
                let proj = PathBuf::from(app.import_form.project_dir.trim());
                let scene =
                    app.import_form.discovered_scenes[app.import_form.selected_scene_idx].clone();
                let avatar_name = app.import_form.custom_avatar_name.trim().to_owned();
                let avatar_id = sanitize_avatar_id(&avatar_name);
                let output_dir = app.config.resolved_extract_output_dir().join(avatar_id);

                app.start_import(ExtractOptions {
                    project_dir: proj,
                    scene_path: scene,
                    avatar_name,
                    output_dir,
                    modular_avatar_enabled: app.import_form.modular_avatar_enabled,
                    max_texture_size: app.import_form.max_texture_size,
                });
            }
        });

        ui.add_space(6.0);
        ui.group(|ui| {
            ui.heading("進捗・ログ");
            match &app.import_status {
                ImportJobStatus::Idle => {
                    ui.label("状態: 待機中");
                }
                ImportJobStatus::Running { progress, stage } => {
                    ui.add(
                        egui::ProgressBar::new(progress.clamp(0.0, 1.0))
                            .text(stage)
                            .animate(true),
                    );
                }
                ImportJobStatus::Succeeded {
                    avatar_id,
                    manifest_path,
                } => {
                    ui.colored_label(
                        egui::Color32::from_rgb(90, 210, 120),
                        format!("完了: {avatar_id} ({})", manifest_path.display()),
                    );
                }
                ImportJobStatus::Failed { error } => {
                    ui.colored_label(
                        egui::Color32::from_rgb(230, 100, 100),
                        format!("失敗: {error}"),
                    );
                }
            }
            ui.separator();
            egui::ScrollArea::vertical()
                .id_salt("import_logs_scroll")
                .max_height((ui.available_height() - 12.0).max(160.0))
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for line in &app.import_form.logs {
                        ui.monospace(line);
                    }
                });
        });
    }

    fn draw_settings_tab(
        &mut self,
        ui: &mut egui::Ui,
        app: &mut AppState,
        effects: &mut StatusUiEffects,
    ) {
        ui.horizontal(|ui| {
            if ui.button("保存して適用").clicked() {
                app.draft_config.motion.perch_exclude_apps = self
                    .exclude_apps_buf
                    .split([',', '\n'])
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(ToOwned::to_owned)
                    .collect();
                let next = app.draft_config.clone();
                app.apply_and_save_config(next);
                effects.config_applied = true;
            }
            if ui.button("デフォルトに戻す").clicked() {
                app.draft_config = AppConfig::default();
                self.exclude_apps_buf = app.draft_config.motion.perch_exclude_apps.join(", ");
            }
            ui.weak(
                vrc_schema::config_path()
                    .map(|p| format!("保存先: {}", p.display()))
                    .unwrap_or_default(),
            );
        });
        ui.separator();

        egui::ScrollArea::vertical()
            .id_salt("settings_scroll")
            .show(ui, |ui| {
                // [motion] (CF-01)
                egui::CollapsingHeader::new("1. モーション・行動設定 ([motion])")
                    .default_open(true)
                    .show(ui, |ui| {
                        let m = &mut app.draft_config.motion;
                        ui.horizontal_wrapped(|ui| {
                            ui.checkbox(&mut m.autonomous_enabled, "自律行動全体 ON");
                            ui.checkbox(&mut m.idle_enabled, "待機呼吸・揺らぎ");
                            ui.checkbox(&mut m.blink_enabled, "まばたき");
                            ui.checkbox(&mut m.gaze_tracking_enabled, "視線追従");
                            ui.checkbox(&mut m.window_perch_enabled, "ウィンドウ着席");
                            ui.checkbox(&mut m.gravity_enabled, "重力落下");
                            ui.checkbox(&mut m.interaction_enabled, "撫で・クリック反応");
                        });
                        ui.horizontal(|ui| {
                            ui.label("移動速度倍率:");
                            ui.add(egui::Slider::new(&mut m.move_speed_scale, 0.2..=3.0));
                        });
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut m.pause_on_fullscreen, "フルスクリーン時一時停止");
                            ui.checkbox(&mut m.hide_on_fullscreen, "フルスクリーン時非表示");
                        });
                        ui.horizontal(|ui| {
                            ui.label("着席除外アプリ (カンマ区切り):");
                            ui.add(
                                egui::TextEdit::singleline(&mut self.exclude_apps_buf)
                                    .desired_width(420.0),
                            );
                        });
                    });

                // [ai] (CF-02)
                egui::CollapsingHeader::new("2. AI・API設定 ([ai])")
                    .default_open(true)
                    .show(ui, |ui| {
                        let ai = &mut app.draft_config.ai;
                        egui::Grid::new("ai_settings_grid")
                            .num_columns(2)
                            .spacing([12.0, 6.0])
                            .show(ui, |ui| {
                                ui.label("対話プロバイダ:");
                                egui::ComboBox::from_id_salt("chat_provider_combo")
                                    .selected_text(format_ai_provider_label(&ai.chat_provider))
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(
                                            &mut ai.chat_provider,
                                            "gemini".to_owned(),
                                            "gemini (Google Gemini)",
                                        );
                                        ui.selectable_value(
                                            &mut ai.chat_provider,
                                            "openai_compatible".to_owned(),
                                            "openai_compatible (OpenAI / Ollama / LM Studio)",
                                        );
                                    });
                                ui.end_row();

                                ui.label("対話エンドポイント:");
                                ui.add(
                                    egui::TextEdit::singleline(&mut ai.chat_endpoint)
                                        .desired_width(340.0),
                                );
                                ui.end_row();

                                ui.label("対話モデル:");
                                ui.vertical(|ui| {
                                    ui.text_edit_singleline(&mut ai.chat_model);
                                    ui.horizontal_wrapped(|ui| {
                                        if ui.small_button("gemini-2.5-flash").clicked() {
                                            ai.chat_provider = "gemini".to_owned();
                                            ai.chat_endpoint =
                                                "https://generativelanguage.googleapis.com"
                                                    .to_owned();
                                            ai.chat_model = "gemini-2.5-flash".to_owned();
                                        }
                                        if ui.small_button("gpt-4o-mini").clicked() {
                                            ai.chat_provider = "openai_compatible".to_owned();
                                            ai.chat_endpoint = "https://api.openai.com".to_owned();
                                            ai.chat_model = "gpt-4o-mini".to_owned();
                                        }
                                        if ui.small_button("Ollama (localhost:11434)").clicked() {
                                            ai.chat_provider = "openai_compatible".to_owned();
                                            ai.chat_endpoint = "http://127.0.0.1:11434".to_owned();
                                            if ai.chat_model.trim().is_empty()
                                                || ai.chat_model.starts_with("gemini-")
                                            {
                                                ai.chat_model = "llama3.2".to_owned();
                                            }
                                        }
                                    });
                                });
                                ui.end_row();

                                ui.label("対話 API Key:");
                                ui.horizontal(|ui| {
                                    ui.add(
                                        egui::TextEdit::singleline(&mut ai.chat_api_key)
                                            .password(!self.reveal_chat_api_key)
                                            .desired_width(280.0),
                                    );
                                    ui.checkbox(&mut self.reveal_chat_api_key, "表示");
                                    ui.weak("(ローカルLLMは空欄可)");
                                });
                                ui.end_row();

                                ui.label("行動判断AI (Decisions):");
                                ui.checkbox(&mut ai.decisions_enabled, "リモート判断を有効にする");
                                ui.end_row();

                                ui.label("Decisions プロバイダ:");
                                egui::ComboBox::from_id_salt("decisions_provider_combo")
                                    .selected_text(format_ai_provider_label(&ai.decisions_provider))
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(
                                            &mut ai.decisions_provider,
                                            "gemini".to_owned(),
                                            "gemini (Google Gemini)",
                                        );
                                        ui.selectable_value(
                                            &mut ai.decisions_provider,
                                            "openai_compatible".to_owned(),
                                            "openai_compatible (OpenAI / Ollama / LM Studio)",
                                        );
                                        ui.selectable_value(
                                            &mut ai.decisions_provider,
                                            "local".to_owned(),
                                            "local (ローカル自律のみ / API不要)",
                                        );
                                    });
                                ui.end_row();

                                ui.label("Decisions エンドポイント:");
                                ui.add(
                                    egui::TextEdit::singleline(&mut ai.decisions_endpoint)
                                        .desired_width(340.0),
                                );
                                ui.end_row();

                                ui.label("Decisions モデル:");
                                ui.vertical(|ui| {
                                    ui.text_edit_singleline(&mut ai.decisions_model);
                                    ui.horizontal_wrapped(|ui| {
                                        if ui.small_button("gemini-2.5-flash-lite").clicked() {
                                            ai.decisions_provider = "gemini".to_owned();
                                            ai.decisions_endpoint =
                                                "https://generativelanguage.googleapis.com"
                                                    .to_owned();
                                            ai.decisions_model = "gemini-2.5-flash-lite".to_owned();
                                        }
                                        if ui.small_button("gemini-2.0-flash").clicked() {
                                            ai.decisions_provider = "gemini".to_owned();
                                            ai.decisions_endpoint =
                                                "https://generativelanguage.googleapis.com"
                                                    .to_owned();
                                            ai.decisions_model = "gemini-2.0-flash".to_owned();
                                        }
                                        if ui.small_button("gpt-4o-mini").clicked() {
                                            ai.decisions_provider = "openai_compatible".to_owned();
                                            ai.decisions_endpoint =
                                                "https://api.openai.com".to_owned();
                                            ai.decisions_model = "gpt-4o-mini".to_owned();
                                        }
                                        if ui.small_button("Ollama (localhost:11434)").clicked() {
                                            ai.decisions_provider = "openai_compatible".to_owned();
                                            ai.decisions_endpoint =
                                                "http://127.0.0.1:11434".to_owned();
                                            if ai.decisions_model.trim().is_empty()
                                                || ai.decisions_model.starts_with("gemini-")
                                            {
                                                ai.decisions_model = "llama3.2".to_owned();
                                            }
                                        }
                                    });
                                });
                                ui.end_row();

                                ui.label("Decisions API Key:");
                                ui.horizontal(|ui| {
                                    ui.add(
                                        egui::TextEdit::singleline(&mut ai.decisions_api_key)
                                            .password(!self.reveal_decisions_api_key)
                                            .desired_width(280.0),
                                    );
                                    ui.checkbox(&mut self.reveal_decisions_api_key, "表示");
                                    ui.weak("(空欄時は対話Keyを使用)");
                                });
                                ui.end_row();

                                ui.label("判定間隔 (秒):");
                                let mut interval = ai.decision_interval_sec as u32;
                                if ui
                                    .add(egui::DragValue::new(&mut interval).range(5..=600))
                                    .changed()
                                {
                                    ai.decision_interval_sec = u64::from(interval);
                                }
                                ui.end_row();

                                ui.label("保持会話ターン数:");
                                let mut turns = ai.max_history_turns as u32;
                                if ui
                                    .add(egui::DragValue::new(&mut turns).range(1..=100))
                                    .changed()
                                {
                                    ai.max_history_turns = turns as usize;
                                }
                                ui.end_row();

                                ui.label("アクティブタイトル送信:");
                                ui.checkbox(
                                    &mut ai.send_active_window_title,
                                    "許可する (プライバシー注意)",
                                );
                                ui.end_row();
                            });

                        ui.label("システムプロンプト:");
                        ui.add(
                            egui::TextEdit::multiline(&mut ai.system_prompt)
                                .desired_rows(3)
                                .desired_width(f32::INFINITY),
                        );
                    });

                // [voice] (CF-03)
                egui::CollapsingHeader::new("3. 音声合成・リップシンク設定 ([voice])")
                    .default_open(true)
                    .show(ui, |ui| {
                        let mut trigger_vv_check = false;
                        let speakers = app.voicevox_speakers.clone();
                        let v = &mut app.draft_config.voice;
                        ui.horizontal(|ui| {
                            ui.label("VOICEVOX URL:");
                            ui.add(
                                egui::TextEdit::singleline(&mut v.voicevox_url)
                                    .desired_width(260.0),
                            );
                            if ui.button("接続テスト").clicked() {
                                trigger_vv_check = true;
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("話者 (Speaker ID):");
                            if !speakers.is_empty() {
                                let current_label = speakers
                                    .iter()
                                    .flat_map(|sp| sp.styles.iter().map(move |st| (sp, st)))
                                    .find(|(_, st)| st.id == v.speaker_id)
                                    .map(|(sp, st)| {
                                        format!("{} ({}) [{}]", sp.name, st.name, st.id)
                                    })
                                    .unwrap_or_else(|| format!("ID {}", v.speaker_id));
                                egui::ComboBox::from_id_salt("voicevox_speaker_combo")
                                    .selected_text(current_label)
                                    .show_ui(ui, |ui| {
                                        for sp in &speakers {
                                            for st in &sp.styles {
                                                let lbl = format!(
                                                    "{} ({}) [{}]",
                                                    sp.name, st.name, st.id
                                                );
                                                ui.selectable_value(&mut v.speaker_id, st.id, lbl);
                                            }
                                        }
                                    });
                            }
                            ui.add(egui::DragValue::new(&mut v.speaker_id));
                        });
                        ui.horizontal(|ui| {
                            ui.label("話速:");
                            ui.add(egui::Slider::new(&mut v.speed_scale, 0.5..=2.0));
                            ui.label("音高:");
                            ui.add(egui::Slider::new(&mut v.pitch_scale, -0.15..=0.15));
                        });
                        ui.horizontal(|ui| {
                            ui.label("抑揚:");
                            ui.add(egui::Slider::new(&mut v.intonation_scale, 0.0..=2.0));
                            ui.label("音量:");
                            ui.add(egui::Slider::new(&mut v.volume_scale, 0.0..=2.0));
                        });
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut v.muted, "ミュート");
                            ui.checkbox(&mut v.lip_sync_enabled, "リップシンク有効");
                            ui.label("口パク感度:");
                            ui.add(egui::Slider::new(&mut v.lip_sync_sensitivity, 0.2..=2.5));
                        });
                        if trigger_vv_check {
                            app.config.voice.voicevox_url =
                                app.draft_config.voice.voicevox_url.clone();
                            app.check_voicevox();
                        }
                    });

                // [display] (CF-04)
                egui::CollapsingHeader::new("4. 表示・ウィンドウ設定 ([display])")
                    .default_open(true)
                    .show(ui, |ui| {
                        let d = &mut app.draft_config.display;
                        ui.horizontal(|ui| {
                            ui.label("表示スケール:");
                            ui.add(egui::Slider::new(&mut d.scale, 0.5..=2.0));
                            ui.label("上限 FPS:");
                            ui.add(egui::DragValue::new(&mut d.max_fps).range(15..=240));
                        });
                        ui.horizontal_wrapped(|ui| {
                            ui.checkbox(&mut d.always_on_top, "常に最前面表示");
                            ui.checkbox(&mut d.click_through_locked, "クリック透過ロック");
                            ui.checkbox(&mut d.auto_start, "Windows起動時に自動起動");
                            ui.checkbox(
                                &mut d.show_window_on_startup,
                                "起動時にこのウィンドウを開く",
                            );
                        });
                    });

                // [physics_shader] (CF-05)
                egui::CollapsingHeader::new("5. 物理・シェーダー設定 ([physics_shader])")
                    .default_open(true)
                    .show(ui, |ui| {
                        let p = &mut app.draft_config.physics_shader;
                        ui.horizontal_wrapped(|ui| {
                            ui.checkbox(&mut p.physbone_enabled, "PhysBone 物理演算");
                            ui.checkbox(&mut p.physbone_collider_enabled, "コライダー衝突");
                            ui.checkbox(&mut p.outline_enabled, "アウトライン描画");
                            ui.checkbox(&mut p.fur_enabled, "Fur シェル描画");
                            ui.checkbox(&mut p.audiolink_enabled, "AudioLink (WASAPI FFT)");
                        });
                        ui.horizontal(|ui| {
                            ui.label("揺れ強度倍率:");
                            ui.add(egui::Slider::new(&mut p.physbone_strength, 0.0..=2.5));
                            ui.label("品質プリセット:");
                            egui::ComboBox::from_id_salt("quality_preset_combo")
                                .selected_text(format!("{:?}", p.quality_preset))
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut p.quality_preset,
                                        QualityPreset::Low,
                                        "Low",
                                    );
                                    ui.selectable_value(
                                        &mut p.quality_preset,
                                        QualityPreset::Medium,
                                        "Medium",
                                    );
                                    ui.selectable_value(
                                        &mut p.quality_preset,
                                        QualityPreset::High,
                                        "High",
                                    );
                                });
                        });
                    });

                // [avatar_import] (CF-06)
                egui::CollapsingHeader::new("6. アバター・インポート設定 ([avatar_import])")
                    .default_open(true)
                    .show(ui, |ui| {
                        let mut open_extract_picker: Option<Option<PathBuf>> = None;
                        let picker_disabled = app.folder_picker_open;
                        let ai = &mut app.draft_config.avatar_import;
                        ui.horizontal(|ui| {
                            ui.label("デフォルトアバター ID:");
                            ui.text_edit_singleline(&mut ai.default_avatar_id);
                            if !app.registry.entries().is_empty() {
                                egui::ComboBox::from_id_salt("default_avatar_combo")
                                    .selected_text("一覧から選択")
                                    .show_ui(ui, |ui| {
                                        for entry in app.registry.entries() {
                                            ui.selectable_value(
                                                &mut ai.default_avatar_id,
                                                entry.avatar_id.clone(),
                                                format!(
                                                    "{} ({})",
                                                    entry.avatar_name, entry.avatar_id
                                                ),
                                            );
                                        }
                                    });
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("抽出先ディレクトリ:");
                            ui.add(
                                egui::TextEdit::singleline(&mut ai.extract_output_dir)
                                    .desired_width(340.0),
                            );
                            if ui
                                .add_enabled(!picker_disabled, egui::Button::new("選択..."))
                                .clicked()
                            {
                                let trimmed = ai.extract_output_dir.trim();
                                let initial = (!trimmed.is_empty()).then(|| PathBuf::from(trimmed));
                                open_extract_picker = Some(initial);
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.checkbox(
                                &mut ai.modular_avatar_enabled,
                                "Modular Avatar デフォルト有効",
                            );
                            ui.label("最大テクスチャ解像度:");
                            egui::ComboBox::from_id_salt("settings_max_tex")
                                .selected_text(format!("{} px", ai.max_texture_size))
                                .show_ui(ui, |ui| {
                                    for sz in [512_u32, 1024, 2048, 4096] {
                                        ui.selectable_value(
                                            &mut ai.max_texture_size,
                                            sz,
                                            format!("{sz} px"),
                                        );
                                    }
                                });
                        });
                        if let Some(initial) = open_extract_picker {
                            app.open_folder_picker(FolderPickerTarget::ExtractOutputDir, initial);
                        }
                    });
            });
    }
}

fn draw_expression_menu_items(ui: &mut egui::Ui, items: &[ExpressionMenuItem], app: &mut AppState) {
    for item in items {
        match item.item_type {
            ExpressionMenuItemType::SubMenu => {
                egui::CollapsingHeader::new(&item.name)
                    .default_open(false)
                    .show(ui, |ui| {
                        draw_expression_menu_items(ui, &item.sub_menu, app);
                    });
            }
            ExpressionMenuItemType::Toggle => {
                let mut active = false;
                if let Some(param_name) = &item.parameter {
                    if let Some(val) = app.simulation.get_param(param_name) {
                        active = match val {
                            ParamValue::Bool(b) => b,
                            ParamValue::Int(i) => i == item.value.round() as i32,
                            ParamValue::Float(f) => (f - item.value).abs() < 0.05,
                        };
                    }
                }
                if ui.checkbox(&mut active, &item.name).changed() {
                    app.simulation.trigger_menu_item(&item.name);
                }
            }
            ExpressionMenuItemType::Button => {
                if ui.button(&item.name).clicked() {
                    app.simulation.trigger_menu_item(&item.name);
                }
            }
            ExpressionMenuItemType::RadialPuppet => {
                if let Some(param_name) = &item.parameter {
                    let mut val = app
                        .simulation
                        .get_param(param_name)
                        .map(ParamValue::as_f32)
                        .unwrap_or(item.value);
                    ui.horizontal(|ui| {
                        ui.label(&item.name);
                        if ui.add(egui::Slider::new(&mut val, 0.0..=1.0)).changed() {
                            app.simulation.set_param_float(param_name, val);
                        }
                    });
                } else {
                    ui.label(format!("{} (パラメータ未設定)", item.name));
                }
            }
        }
    }
}

fn format_ai_provider_label(provider: &str) -> &str {
    match provider.trim().to_ascii_lowercase().as_str() {
        "gemini" | "" => "gemini (Google Gemini)",
        "openai_compatible" | "openai" => "openai_compatible (OpenAI / Local LLM)",
        "local" => "local (ローカル自律のみ)",
        _ => provider,
    }
}

fn format_mascot_state(state: MascotState) -> &'static str {
    match state {
        MascotState::Idle => "待機",
        MascotState::Dragged => "ドラッグ中",
        MascotState::Falling => "落下中",
        MascotState::SeatedOnTaskbar => "タスクバー着席",
        MascotState::SeatedOnWindow => "ウィンドウ着席",
    }
}

fn format_emotion(tag: &EmotionTag) -> String {
    match tag {
        EmotionTag::Neutral => "通常".to_owned(),
        EmotionTag::Happy => "喜び".to_owned(),
        EmotionTag::Sad => "悲しみ".to_owned(),
        EmotionTag::Angry => "怒り".to_owned(),
        EmotionTag::Surprised => "驚き".to_owned(),
        EmotionTag::Shy => "照れ".to_owned(),
        EmotionTag::Sleepy => "眠い".to_owned(),
        EmotionTag::Custom(s) => s.clone(),
    }
}

fn format_decision_action(action: DecisionAction) -> &'static str {
    match action {
        DecisionAction::Speak => "SPEAK (発話)",
        DecisionAction::MoveToTaskbar => "MOVE_TO_TASKBAR",
        DecisionAction::SitOnWindow => "SIT_ON_WINDOW",
        DecisionAction::ChangeExpression => "CHANGE_EXPRESSION",
        DecisionAction::Idle => "IDLE",
    }
}

fn format_epoch_hms(epoch_ms: u64) -> String {
    let total_secs = (epoch_ms / 1000) + 9 * 3600; // JST display
    let h = (total_secs / 3600) % 24;
    let m = (total_secs / 60) % 60;
    let s = total_secs % 60;
    format!("{h:02}:{m:02}:{s:02}")
}
