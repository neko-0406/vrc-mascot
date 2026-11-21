//! Windows desktop mascot host application (`winit` + `egui` + Win32 + `tray-icon`).

mod app_state;
mod status_ui;
mod window_manager;

use std::path::PathBuf;

use clap::Parser;
use tracing_subscriber::EnvFilter;
use vrc_schema::AppConfig;

use crate::app_state::AppState;

#[derive(Debug, Parser)]
#[command(
    name = "vrc-mascot-app",
    about = "VRC Desktop Mascot runtime application"
)]
struct Cli {
    /// Optional path to a `mascot_manifest.json` to load on startup.
    #[arg(long)]
    manifest: Option<PathBuf>,

    /// Open the Status & Settings window immediately on startup.
    #[arg(long)]
    show_window: bool,
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,vrc_mascot_app=info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
}

fn main() -> anyhow::Result<()> {
    init_tracing();
    let cli = Cli::parse();

    let mut config = AppConfig::load_or_default()?;
    config.apply_env_fallbacks();

    tracing::info!(
        manifest = ?cli.manifest,
        show_window = cli.show_window,
        scale = config.display.scale,
        "Starting vrc-mascot-app"
    );

    let app_state = AppState::initialize(config, cli.manifest.as_deref(), cli.show_window)?;
    window_manager::run_event_loop(app_state)?;
    Ok(())
}
