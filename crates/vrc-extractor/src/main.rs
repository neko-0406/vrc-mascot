use std::path::PathBuf;

use clap::Parser;
use tracing_subscriber::EnvFilter;
use vrc_extractor::{extract_avatar, ExtractOptions};

#[derive(Debug, Parser)]
#[command(
    name = "vrc-extractor",
    about = "Extract VRChat avatars from Unity projects without Unity Editor"
)]
struct Cli {
    /// Path to the Unity project root directory.
    #[arg(long)]
    project: PathBuf,
    /// Path to the target Unity scene (e.g. Assets/SampleScene.unity).
    #[arg(long)]
    scene: PathBuf,
    /// Name of the target avatar GameObject.
    #[arg(long)]
    avatar: String,
    /// Output directory for avatar.glb and mascot_manifest.json.
    #[arg(long)]
    output: PathBuf,
    /// Disable virtual Modular Avatar component application.
    #[arg(long, default_value_t = false)]
    no_modular_avatar: bool,
    /// Maximum embedded texture dimension in pixels (0 = no resize).
    #[arg(long, default_value_t = 2048)]
    max_texture_size: u32,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    let opts = ExtractOptions {
        project_dir: cli.project,
        scene_path: cli.scene,
        avatar_name: cli.avatar,
        output_dir: cli.output,
        modular_avatar_enabled: !cli.no_modular_avatar,
        max_texture_size: cli.max_texture_size,
    };

    let report = extract_avatar(&opts, |progress| {
        tracing::info!(
            stage = %progress.stage,
            progress = progress.progress,
            "Extraction progress"
        );
    })?;

    for warning in &report.warnings {
        tracing::warn!("{warning}");
    }

    println!(
        "Extracted '{}' -> GLB: {} | Manifest: {}",
        report.avatar_name,
        report.glb_path.display(),
        report.manifest_path.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_parses_all_flags() {
        let cli = Cli::try_parse_from([
            "vrc-extractor",
            "--project",
            "C:\\UnityProj",
            "--scene",
            "Assets/Sample.unity",
            "--avatar",
            "MyAvatar",
            "--output",
            "out/MyAvatar",
            "--no-modular-avatar",
            "--max-texture-size",
            "1024",
        ])
        .expect("parse cli flags");

        assert_eq!(cli.project, PathBuf::from("C:\\UnityProj"));
        assert_eq!(cli.scene, PathBuf::from("Assets/Sample.unity"));
        assert_eq!(cli.avatar, "MyAvatar");
        assert_eq!(cli.output, PathBuf::from("out/MyAvatar"));
        assert!(cli.no_modular_avatar);
        assert_eq!(cli.max_texture_size, 1024);
    }
}
