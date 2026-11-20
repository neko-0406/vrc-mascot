use std::path::Path;

use vrc_schema::MascotManifest;

use crate::error::ExtractorError;

/// Validates and writes `mascot_manifest.json` to `output_path`.
pub fn export_manifest(
    output_path: &Path,
    manifest: &MascotManifest,
) -> Result<(), ExtractorError> {
    manifest.save(output_path)?;
    Ok(())
}
