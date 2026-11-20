//! Privacy filtering and coarse window categorization for autonomous decision context.

use std::path::Path;

use vrc_schema::{AiConfig, DecisionContext, EmotionTag};

const BROWSER_APPS: &[&str] = &[
    "chrome", "msedge", "firefox", "brave", "vivaldi", "opera", "arc", "waterfox", "zen",
    "iexplore", "floorp",
];

const EDITOR_APPS: &[&str] = &[
    "code",
    "devenv",
    "cursor",
    "windsurf",
    "antigravity",
    "idea64",
    "clion64",
    "rustrover64",
    "pycharm64",
    "webstorm64",
    "goland64",
    "rider64",
    "sublime_text",
    "notepad++",
    "notepad",
    "vim",
    "nvim",
    "emacs",
    "zed",
    "obsidian",
    "winword",
    "excel",
    "powerpnt",
];

const TERMINAL_APPS: &[&str] = &[
    "windowsterminal",
    "wt",
    "powershell",
    "pwsh",
    "cmd",
    "alacritty",
    "wezterm",
    "wezterm-gui",
    "kitty",
    "mintty",
    "conhost",
    "tabby",
    "hyper",
];

const CHAT_APPS: &[&str] = &[
    "discord",
    "slack",
    "teams",
    "ms-teams",
    "line",
    "telegram",
    "signal",
    "zoom",
    "skype",
    "mattermost",
];

const MEDIA_APPS: &[&str] = &[
    "spotify",
    "vlc",
    "mpc-hc64",
    "mpc-hc",
    "mpv",
    "foobar2000",
    "itunes",
    "applemusic",
    "audacity",
    "obs64",
    "obs32",
    "streamlabs",
];

const GAME_APPS: &[&str] = &[
    "valorant",
    "valorant-win64-shipping",
    "vrchat",
    "steam",
    "steamwebhelper",
    "epicgameslauncher",
    "genshinimpact",
    "starrail",
    "ffxiv",
    "ffxiv_dx11",
    "eldenring",
    "minecraft",
    "javaw",
    "robloxplayerbeta",
    "r5apex",
    "overwatch",
    "cs2",
    "dota2",
    "hearthstone",
];

const SENSITIVE_APPS: &[&str] = &[
    "1password",
    "bitwarden",
    "keepass",
    "keepassxc",
    "lastpass",
    "dashlane",
    "roboform",
    "credentialuibroker",
];

fn normalize_process_stem(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let file_name = trimmed.rsplit(['/', '\\']).next().unwrap_or(trimmed).trim();
    let lower = file_name.to_ascii_lowercase();
    lower.strip_suffix(".exe").unwrap_or(&lower).to_owned()
}

/// Categorizes a process name into a coarse activity bucket:
/// `"browser"`, `"editor"`, `"terminal"`, `"chat"`, `"media"`, `"game"`, or `"other"`.
pub fn categorize_process(active_process_name: Option<&str>) -> &'static str {
    let Some(proc_name) = active_process_name else {
        return "other";
    };
    let stem = normalize_process_stem(proc_name);
    if stem.is_empty() {
        return "other";
    }
    let s = stem.as_str();
    if BROWSER_APPS.contains(&s) {
        "browser"
    } else if EDITOR_APPS.contains(&s) {
        "editor"
    } else if TERMINAL_APPS.contains(&s) {
        "terminal"
    } else if CHAT_APPS.contains(&s) {
        "chat"
    } else if MEDIA_APPS.contains(&s) {
        "media"
    } else if GAME_APPS.contains(&s) {
        "game"
    } else {
        "other"
    }
}

/// Returns `true` if `active_process_name` matches an entry in `exclude_apps` or a built-in
/// sensitive credential/password-manager process.
pub fn is_excluded_app(active_process_name: Option<&str>, exclude_apps: &[String]) -> bool {
    let Some(proc_name) = active_process_name else {
        return false;
    };
    let proc_stem = normalize_process_stem(proc_name);
    if proc_stem.is_empty() {
        return false;
    }

    if SENSITIVE_APPS.contains(&proc_stem.as_str()) {
        return true;
    }

    let proc_file = Path::new(proc_name.trim())
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(proc_name.trim());

    exclude_apps.iter().any(|entry| {
        let trimmed = entry.trim();
        if trimmed.is_empty() {
            return false;
        }
        let excl_file = Path::new(trimmed)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(trimmed);
        proc_file.eq_ignore_ascii_case(excl_file) || proc_stem == normalize_process_stem(trimmed)
    })
}

/// Builds a privacy-filtered [`DecisionContext`] suitable for external AI calls.
///
/// The raw `active_window_title` is omitted (`None`) unless
/// `ai_cfg.send_active_window_title` is `true` AND `active_process_name` is neither
/// in `exclude_apps` nor a known sensitive application (such as a password manager).
#[allow(clippy::too_many_arguments)]
pub fn sanitize_decision_context(
    idle_seconds: f32,
    pet_count_recent: u32,
    active_process_name: Option<&str>,
    active_window_title: Option<&str>,
    time_of_day: &str,
    current_mood: EmotionTag,
    ai_cfg: &AiConfig,
    exclude_apps: &[String],
) -> DecisionContext {
    let category = categorize_process(active_process_name).to_owned();
    let excluded = is_excluded_app(active_process_name, exclude_apps);

    let sanitized_title = if ai_cfg.send_active_window_title && !excluded {
        active_window_title
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .map(ToOwned::to_owned)
    } else {
        None
    };

    let safe_idle = if idle_seconds.is_finite() {
        idle_seconds.max(0.0)
    } else {
        0.0
    };

    DecisionContext {
        idle_seconds: safe_idle,
        pet_count_recent,
        active_window_category: category,
        active_window_title: sanitized_title,
        time_of_day: time_of_day.trim().to_owned(),
        current_mood,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categorizes_known_processes_and_unknown() {
        assert_eq!(categorize_process(Some("chrome.exe")), "browser");
        assert_eq!(
            categorize_process(Some("C:\\Program Files\\Mozilla Firefox\\firefox.exe")),
            "browser"
        );
        assert_eq!(categorize_process(Some("Code.exe")), "editor");
        assert_eq!(categorize_process(Some("WindowsTerminal.exe")), "terminal");
        assert_eq!(categorize_process(Some("Discord.exe")), "chat");
        assert_eq!(categorize_process(Some("Spotify.exe")), "media");
        assert_eq!(categorize_process(Some("VRChat.exe")), "game");
        assert_eq!(categorize_process(Some("my_custom_tool.exe")), "other");
        assert_eq!(categorize_process(None), "other");
    }

    #[test]
    fn omits_title_when_disabled_or_excluded_or_sensitive() {
        let mut cfg = AiConfig::default();
        let exclude = vec![
            "obs64.exe".to_owned(),
            "VALORANT-Win64-Shipping.exe".to_owned(),
        ];

        // Default send_active_window_title == false -> title omitted
        let ctx = sanitize_decision_context(
            12.5,
            2,
            Some("code.exe"),
            Some("secret_project - Visual Studio Code"),
            "afternoon",
            EmotionTag::Happy,
            &cfg,
            &exclude,
        );
        assert_eq!(ctx.active_window_category, "editor");
        assert_eq!(ctx.active_window_title, None);

        // Enabled and not excluded -> title preserved
        cfg.send_active_window_title = true;
        let ctx = sanitize_decision_context(
            -5.0,
            1,
            Some("code.exe"),
            Some("  main.rs - VS Code  "),
            "night",
            EmotionTag::Neutral,
            &cfg,
            &exclude,
        );
        assert_eq!(ctx.idle_seconds, 0.0);
        assert_eq!(
            ctx.active_window_title.as_deref(),
            Some("main.rs - VS Code")
        );

        // Enabled but in exclude_apps (case-insensitive) -> title omitted
        let ctx = sanitize_decision_context(
            3.0,
            0,
            Some("OBS64.EXE"),
            Some("Streaming Scene"),
            "evening",
            EmotionTag::Neutral,
            &cfg,
            &exclude,
        );
        assert_eq!(ctx.active_window_category, "media");
        assert_eq!(ctx.active_window_title, None);

        // Enabled but sensitive password manager -> title always omitted
        let ctx = sanitize_decision_context(
            1.0,
            0,
            Some("Bitwarden.exe"),
            Some("Vault - Passwords"),
            "morning",
            EmotionTag::Neutral,
            &cfg,
            &[],
        );
        assert_eq!(ctx.active_window_title, None);
    }
}
