//! Windows desktop mascot host application (`winit` + `egui` + Win32 + `tray-icon`).

use std::path::PathBuf;

use clap::Parser;
use tracing_subscriber::EnvFilter;
use vrc_schema::AppConfig;

#[derive(Debug, Parser)]
#[command(
    name = "vrc-mascot-app",
    about = "VRC Desktop Mascot runtime application"
)]
struct Cli {
    /// Optional path to a `mascot_manifest.json` to load on startup.
    #[arg(long)]
    manifest: Option<PathBuf>,
}

fn verify_crate_linkage() {
    let _ = std::mem::size_of::<wgpu::InstanceDescriptor>();
    let _ = std::mem::size_of::<egui::Context>();
    let _ = std::mem::size_of::<egui_wgpu::Renderer>();
    let _ = std::mem::size_of::<egui_winit::State>();
    let _ = std::mem::size_of::<winit::window::WindowId>();
    let _ = std::mem::size_of::<tray_icon::TrayIconAttributes>();
    let _ = std::mem::size_of::<windows::Win32::Foundation::HWND>();
    let _ = std::mem::size_of::<wasapi::WaveFormat>();
    let _ = std::mem::size_of::<rfd::FileDialog>();
    let _ = vrc_renderer::preferred_surface_format();
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    verify_crate_linkage();

    let mut config = AppConfig::load_or_default()?;
    config.apply_env_fallbacks();

    let _physics = vrc_physics::PhysicsWorld::default();
    let _ai = vrc_ai::AiCoordinator::new(config.ai.clone(), config.voice.clone())?;

    tracing::info!(
        manifest = ?cli.manifest,
        scale = config.display.scale,
        "vrc-mascot-app skeleton initialized"
    );
    Ok(())
}
