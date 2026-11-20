use std::path::PathBuf;

use clap::Parser;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(
    name = "vrc-extractor",
    about = "Extract VRChat avatars from Unity projects without Unity Editor"
)]
struct Cli {
    /// Path to the Unity project root directory.
    #[arg(long)]
    project: PathBuf,
    /// Relative path to the target Unity scene (e.g. Assets/SampleScene.unity).
    #[arg(long)]
    scene: PathBuf,
    /// Name of the target avatar GameObject.
    #[arg(long)]
    avatar: String,
    /// Output directory for avatar.glb and mascot_manifest.json.
    #[arg(long)]
    output: PathBuf,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    let req = vrc_extractor::ExtractRequest {
        project_dir: cli.project,
        scene_path: cli.scene,
        avatar_name: cli.avatar,
        output_dir: cli.output,
        modular_avatar_enabled: true,
        max_texture_size: 2048,
    };
    vrc_extractor::validate_request(&req)?;
    tracing::info!("vrc-extractor CLI skeleton initialized");
    Ok(())
}
