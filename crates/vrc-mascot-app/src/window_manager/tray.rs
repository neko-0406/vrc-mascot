//! System tray icon, dynamic avatar/expression submenus, and `TrayAction` dispatch.

use std::collections::HashMap;

use tray_icon::menu::{
    CheckMenuItem, ContextMenu, Menu, MenuId, MenuItem, PredefinedMenuItem, Submenu,
};
use tray_icon::{Icon, TrayIcon, TrayIconBuilder};
use vrc_schema::{AvatarEntry, ExpressionMenuItem, ExpressionMenuItemType};

/// Semantic action triggered from either the system tray menu or the avatar context menu.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrayAction {
    ToggleStatusWindow,
    SwitchAvatar(Option<String>),
    TriggerExpression(String),
    ResetPosition,
    ToggleAutonomous,
    ToggleClickThroughLock,
    Exit,
}

/// Snapshot of runtime state needed to build or refresh the tray/context menus.
#[derive(Debug, Clone)]
pub struct TrayMenuSpec<'a> {
    pub avatars: &'a [AvatarEntry],
    pub active_avatar_id: Option<&'a str>,
    pub expressions: &'a [ExpressionMenuItem],
    pub autonomous_enabled: bool,
    pub click_through_locked: bool,
    pub status_window_visible: bool,
}

/// Pure menu-ID to [`TrayAction`] lookup table + dynamic `muda::Menu` builder.
pub struct TrayMenuModel {
    pub menu: Menu,
    action_map: HashMap<MenuId, TrayAction>,
}

impl TrayMenuModel {
    /// Builds a fresh `Menu` and ID->[`TrayAction`] map from `spec`.
    pub fn build(spec: &TrayMenuSpec<'_>) -> anyhow::Result<Self> {
        let menu = Menu::new();
        let mut action_map = HashMap::new();

        let toggle_label = if spec.status_window_visible {
            "ステータス・設定ウィンドウを隠す"
        } else {
            "ステータス・設定ウィンドウを開く"
        };
        let toggle_item = MenuItem::new(toggle_label, true, None);
        action_map.insert(toggle_item.id().clone(), TrayAction::ToggleStatusWindow);
        menu.append(&toggle_item)?;

        menu.append(&PredefinedMenuItem::separator())?;

        // Avatar switcher submenu
        let avatar_sub = Submenu::new("アバター切替", true);
        let fallback_item = CheckMenuItem::new(
            "ビルトイン (フォールバック)",
            true,
            spec.active_avatar_id.is_none(),
            None,
        );
        action_map.insert(fallback_item.id().clone(), TrayAction::SwitchAvatar(None));
        avatar_sub.append(&fallback_item)?;

        if !spec.avatars.is_empty() {
            avatar_sub.append(&PredefinedMenuItem::separator())?;
            for entry in spec.avatars {
                let is_active = spec.active_avatar_id == Some(entry.avatar_id.as_str());
                let item = CheckMenuItem::new(&entry.avatar_name, true, is_active, None);
                action_map.insert(
                    item.id().clone(),
                    TrayAction::SwitchAvatar(Some(entry.avatar_id.clone())),
                );
                avatar_sub.append(&item)?;
            }
        }
        menu.append(&avatar_sub)?;

        // Dynamic Expressions submenu from MascotManifest
        let expr_sub = Submenu::new("表情・ギミック", true);
        if spec.expressions.is_empty() {
            let empty_item = MenuItem::new("(メニュー項目なし)", false, None);
            expr_sub.append(&empty_item)?;
        } else {
            append_expression_items(&expr_sub, spec.expressions, &mut action_map)?;
        }
        menu.append(&expr_sub)?;

        menu.append(&PredefinedMenuItem::separator())?;

        let reset_item = MenuItem::new("位置をリセット", true, None);
        action_map.insert(reset_item.id().clone(), TrayAction::ResetPosition);
        menu.append(&reset_item)?;

        let auto_item = CheckMenuItem::new("AI自律行動", true, spec.autonomous_enabled, None);
        action_map.insert(auto_item.id().clone(), TrayAction::ToggleAutonomous);
        menu.append(&auto_item)?;

        let lock_item =
            CheckMenuItem::new("クリック透過ロック", true, spec.click_through_locked, None);
        action_map.insert(lock_item.id().clone(), TrayAction::ToggleClickThroughLock);
        menu.append(&lock_item)?;

        menu.append(&PredefinedMenuItem::separator())?;

        let exit_item = MenuItem::new("終了", true, None);
        action_map.insert(exit_item.id().clone(), TrayAction::Exit);
        menu.append(&exit_item)?;

        Ok(Self { menu, action_map })
    }

    /// Resolves a `MenuId` click event to its semantic [`TrayAction`].
    pub fn resolve_action(&self, id: &MenuId) -> Option<&TrayAction> {
        self.action_map.get(id)
    }

    /// Returns all registered ID-to-action pairs (useful for diagnostics and tests).
    #[cfg(test)]
    pub fn actions(&self) -> &HashMap<MenuId, TrayAction> {
        &self.action_map
    }
}

fn append_expression_items(
    parent: &Submenu,
    items: &[ExpressionMenuItem],
    action_map: &mut HashMap<MenuId, TrayAction>,
) -> anyhow::Result<()> {
    for item in items {
        match item.item_type {
            ExpressionMenuItemType::SubMenu => {
                let sub = Submenu::new(&item.name, true);
                if item.sub_menu.is_empty() {
                    sub.append(&MenuItem::new("(空)", false, None))?;
                } else {
                    append_expression_items(&sub, &item.sub_menu, action_map)?;
                }
                parent.append(&sub)?;
            }
            ExpressionMenuItemType::Toggle
            | ExpressionMenuItemType::Button
            | ExpressionMenuItemType::RadialPuppet => {
                let label = if item.item_type == ExpressionMenuItemType::RadialPuppet {
                    format!("{} (Puppet)", item.name)
                } else {
                    item.name.clone()
                };
                let mi = MenuItem::new(label, true, None);
                action_map.insert(
                    mi.id().clone(),
                    TrayAction::TriggerExpression(item.name.clone()),
                );
                parent.append(&mi)?;
            }
        }
    }
    Ok(())
}

const TRAY_ICON_ID: &str = "vrc-desktop-mascot-tray";
const TRAY_ICON_GUID: u128 = 0x7A4B_9D12_4E3F_4B89_A1C0_3F5E_21D8_0942;

/// Manages the live Windows system tray icon and avatar right-click context menu.
pub struct TrayController {
    tray_icon: TrayIcon,
    tray_menu_model: TrayMenuModel,
    context_menu_model: TrayMenuModel,
}

impl TrayController {
    /// Creates the Windows system tray icon with a procedural 32x32 mascot icon.
    pub fn new(spec: &TrayMenuSpec<'_>) -> anyhow::Result<Self> {
        let icon = create_mascot_icon()?;
        let tray_menu_model = TrayMenuModel::build(spec)?;
        let context_menu_model = TrayMenuModel::build(spec)?;

        let tray_icon = TrayIconBuilder::new()
            .with_id(TRAY_ICON_ID)
            .with_guid(TRAY_ICON_GUID)
            .with_tooltip("VRC Desktop Mascot")
            .with_icon(icon)
            .with_menu(Box::new(tray_menu_model.menu.clone()))
            .build()?;

        tracing::info!(tray_id = TRAY_ICON_ID, "System tray icon initialized");

        Ok(Self {
            tray_icon,
            tray_menu_model,
            context_menu_model,
        })
    }

    /// Rebuilds both tray and context menus when avatars, expressions, or toggles change.
    pub fn rebuild(&mut self, spec: &TrayMenuSpec<'_>) -> anyhow::Result<()> {
        let new_tray = TrayMenuModel::build(spec)?;
        let new_ctx = TrayMenuModel::build(spec)?;
        self.tray_icon
            .set_menu(Some(Box::new(new_tray.menu.clone())));
        self.tray_menu_model = new_tray;
        self.context_menu_model = new_ctx;
        Ok(())
    }

    /// Resolves a `MenuId` against either the tray menu or the popup context menu.
    pub fn resolve_action(&self, id: &MenuId) -> Option<TrayAction> {
        self.tray_menu_model
            .resolve_action(id)
            .or_else(|| self.context_menu_model.resolve_action(id))
            .cloned()
    }

    /// Displays the context menu anchored at the current cursor position over `hwnd_raw`.
    pub fn show_context_menu(&self, hwnd_raw: isize) {
        if hwnd_raw == 0 {
            return;
        }
        unsafe {
            let _ = self
                .context_menu_model
                .menu
                .show_context_menu_for_hwnd(hwnd_raw, None);
        }
    }
}

/// Generates a clean 32x32 RGBA procedural mascot emblem icon (pastel silhouette + ears + eyes).
pub fn create_mascot_icon_rgba() -> Vec<u8> {
    const SIZE: usize = 32;
    let mut rgba = vec![0u8; SIZE * SIZE * 4];

    for y in 0..SIZE {
        for x in 0..SIZE {
            let fx = (x as f32 + 0.5) / SIZE as f32 * 2.0 - 1.0;
            let fy = (y as f32 + 0.5) / SIZE as f32 * 2.0 - 1.0;

            let head_dist = (fx * fx * 1.15 + (fy - 0.14) * (fy - 0.14) * 1.25).sqrt();
            let in_head = head_dist <= 0.72;

            // Left & right cat/kemono ears
            let in_left_ear = fy < -0.15
                && fy > -0.88
                && fx < -0.15
                && fx > -0.78
                && (fx + 0.50).abs() < (fy + 0.90) * 0.55;
            let in_right_ear = fy < -0.15
                && fy > -0.88
                && fx > 0.15
                && fx < 0.78
                && (fx - 0.50).abs() < (fy + 0.90) * 0.55;

            let idx = (y * SIZE + x) * 4;
            if in_head || in_left_ear || in_right_ear {
                // Soft outline ring
                let is_outline = head_dist > 0.63 && !in_left_ear && !in_right_ear;
                // Eyes
                let in_left_eye = ((fx + 0.24).powi(2) + (fy - 0.12).powi(2)).sqrt() < 0.11;
                let in_right_eye = ((fx - 0.24).powi(2) + (fy - 0.12).powi(2)).sqrt() < 0.11;

                let color = if in_left_eye || in_right_eye {
                    [52, 62, 118, 255]
                } else if is_outline {
                    [92, 68, 108, 255]
                } else if in_left_ear || in_right_ear {
                    [246, 178, 206, 255]
                } else {
                    [255, 224, 236, 255]
                };
                rgba[idx..idx + 4].copy_from_slice(&color);
            }
        }
    }
    rgba
}

/// Converts [`create_mascot_icon_rgba`] into a `tray_icon::Icon`.
pub fn create_mascot_icon() -> anyhow::Result<Icon> {
    let rgba = create_mascot_icon_rgba();
    Ok(Icon::from_rgba(rgba, 32, 32)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn procedural_icon_is_valid_32x32_with_transparent_corners() {
        let rgba = create_mascot_icon_rgba();
        assert_eq!(rgba.len(), 32 * 32 * 4);
        // Corner pixel (0, 0) is transparent
        assert_eq!(rgba[3], 0);
        // Center pixel (16, 18) is opaque face
        let center_idx = (18 * 32 + 16) * 4;
        assert_eq!(rgba[center_idx + 3], 255);
        assert!(create_mascot_icon().is_ok());
    }

    #[test]
    fn tray_menu_model_maps_static_avatars_and_nested_expressions() {
        let avatars = vec![AvatarEntry {
            avatar_id: "sample_01".to_owned(),
            avatar_name: "SampleAvatar".to_owned(),
            dir_path: PathBuf::from("avatars/sample_01"),
            manifest_path: PathBuf::from("avatars/sample_01/mascot_manifest.json"),
            glb_path: PathBuf::from("avatars/sample_01/avatar.glb"),
            schema_version: "1.0.0".to_owned(),
        }];
        let expressions = vec![
            ExpressionMenuItem {
                name: "Smile".to_owned(),
                item_type: ExpressionMenuItemType::Toggle,
                parameter: Some("FaceEmotion".to_owned()),
                value: 1.0,
                sub_menu: vec![],
            },
            ExpressionMenuItem {
                name: "Outfits".to_owned(),
                item_type: ExpressionMenuItemType::SubMenu,
                parameter: None,
                value: 0.0,
                sub_menu: vec![ExpressionMenuItem {
                    name: "Jacket".to_owned(),
                    item_type: ExpressionMenuItemType::Toggle,
                    parameter: Some("Jacket".to_owned()),
                    value: 1.0,
                    sub_menu: vec![],
                }],
            },
        ];

        let spec = TrayMenuSpec {
            avatars: &avatars,
            active_avatar_id: Some("sample_01"),
            expressions: &expressions,
            autonomous_enabled: true,
            click_through_locked: false,
            status_window_visible: false,
        };

        let model = TrayMenuModel::build(&spec).expect("build menu model");
        let actions: Vec<&TrayAction> = model.actions().values().collect();
        assert!(actions.contains(&&TrayAction::ToggleStatusWindow));
        assert!(actions.contains(&&TrayAction::SwitchAvatar(None)));
        assert!(actions.contains(&&TrayAction::SwitchAvatar(Some("sample_01".to_owned()))));
        assert!(actions.contains(&&TrayAction::TriggerExpression("Smile".to_owned())));
        assert!(actions.contains(&&TrayAction::TriggerExpression("Jacket".to_owned())));
        assert!(actions.contains(&&TrayAction::ResetPosition));
        assert!(actions.contains(&&TrayAction::ToggleAutonomous));
        assert!(actions.contains(&&TrayAction::ToggleClickThroughLock));
        assert!(actions.contains(&&TrayAction::Exit));
    }
}
