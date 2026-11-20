//! Unity project external parser and intermediate GLB + `mascot_manifest.json` exporter.

use std::path::PathBuf;

use vrc_schema::{ImportJobStatus, MascotManifest};

/// Extraction request parameters for converting a Unity VRChat avatar project.
#[derive(Debug, Clone)]
pub struct ExtractRequest {
    pub project_dir: PathBuf,
    pub scene_path: PathBuf,
    pub avatar_name: String,
    pub output_dir: PathBuf,
    pub modular_avatar_enabled: bool,
    pub max_texture_size: u32,
}

/// Validates basic extraction inputs before Phase 2 parser implementation.
pub fn validate_request(req: &ExtractRequest) -> anyhow::Result<()> {
    let _ = std::mem::size_of::<ufbx::Scene>();
    let _ = gltf::Error::Io;
    let _ = std::mem::size_of::<yaml_rust2::Yaml>();
    if req.avatar_name.trim().is_empty() {
        anyhow::bail!("avatar_name must not be empty");
    }
    Ok(())
}

/// Placeholder manifest builder for workspace compilation verification.
pub fn placeholder_manifest(avatar_name: &str) -> MascotManifest {
    let _ = ImportJobStatus::Idle;
    MascotManifest {
        avatar_name: avatar_name.to_owned(),
        ..Default::default()
    }
}
