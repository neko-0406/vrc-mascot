use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::error::SchemaError;
use crate::manifest::MascotManifest;

/// Default filename for the avatar metadata manifest inside each avatar package directory.
pub const MANIFEST_FILE_NAME: &str = "mascot_manifest.json";

/// Metadata record for a single extracted avatar package in `<extract_output_dir>/<avatar_id>/`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AvatarEntry {
    pub avatar_id: String,
    pub avatar_name: String,
    pub dir_path: PathBuf,
    pub manifest_path: PathBuf,
    pub glb_path: PathBuf,
    pub schema_version: String,
}

/// Manages discovered avatar packages under `<extract_output_dir>`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AvatarRegistry {
    root_dir: PathBuf,
    entries: Vec<AvatarEntry>,
}

impl AvatarRegistry {
    /// Creates an empty registry rooted at `root_dir`.
    pub fn new(root_dir: impl AsRef<Path>) -> Self {
        Self {
            root_dir: root_dir.as_ref().to_path_buf(),
            entries: Vec::new(),
        }
    }

    /// Scans `root_dir` for `<avatar_id>/mascot_manifest.json` packages with an accompanying GLB file.
    ///
    /// If `root_dir` does not exist yet, returns an empty registry without error.
    /// Invalid or incomplete subdirectories are skipped with a warning.
    pub fn scan(root_dir: impl AsRef<Path>) -> Result<Self, SchemaError> {
        let mut registry = Self::new(root_dir);
        registry.refresh()?;
        Ok(registry)
    }

    /// Re-scans `root_dir` and updates `entries` sorted by `avatar_name` (then `avatar_id`).
    pub fn refresh(&mut self) -> Result<(), SchemaError> {
        self.entries.clear();
        if !self.root_dir.exists() {
            return Ok(());
        }

        for entry in fs::read_dir(&self.root_dir)? {
            let entry = entry?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let Some(avatar_id) = path.file_name().and_then(|n| n.to_str()).map(str::to_owned)
            else {
                continue;
            };
            let manifest_path = path.join(MANIFEST_FILE_NAME);
            if !manifest_path.is_file() {
                continue;
            }

            match Self::load_entry(&avatar_id, &path, &manifest_path) {
                Ok(avatar_entry) => self.entries.push(avatar_entry),
                Err(err) => {
                    warn!(
                        avatar_id = %avatar_id,
                        error = %err,
                        "Skipping invalid avatar package during registry scan"
                    );
                }
            }
        }

        self.entries.sort_by(|a, b| {
            a.avatar_name
                .cmp(&b.avatar_name)
                .then_with(|| a.avatar_id.cmp(&b.avatar_id))
        });

        Ok(())
    }

    fn load_entry(
        avatar_id: &str,
        dir_path: &Path,
        manifest_path: &Path,
    ) -> Result<AvatarEntry, SchemaError> {
        let manifest = MascotManifest::load(manifest_path)?;
        let glb_path = dir_path.join(&manifest.glb_file);
        if !glb_path.is_file() {
            return Err(SchemaError::Validation(format!(
                "missing GLB file '{}' for avatar '{}'",
                glb_path.display(),
                avatar_id
            )));
        }

        Ok(AvatarEntry {
            avatar_id: avatar_id.to_owned(),
            avatar_name: manifest.avatar_name,
            dir_path: dir_path.to_path_buf(),
            manifest_path: manifest_path.to_path_buf(),
            glb_path,
            schema_version: manifest.schema_version,
        })
    }

    /// Returns the root directory scanned by this registry.
    pub fn root_dir(&self) -> &Path {
        &self.root_dir
    }

    /// Returns all discovered valid avatar entries.
    pub fn entries(&self) -> &[AvatarEntry] {
        &self.entries
    }

    /// Looks up an avatar entry by its directory `avatar_id`.
    pub fn find_by_id(&self, avatar_id: &str) -> Option<&AvatarEntry> {
        self.entries.iter().find(|e| e.avatar_id == avatar_id)
    }

    /// Returns the matching `default_avatar_id` entry if present, otherwise the first available entry.
    pub fn default_or_first(&self, default_avatar_id: &str) -> Option<&AvatarEntry> {
        if !default_avatar_id.trim().is_empty() {
            if let Some(entry) = self.find_by_id(default_avatar_id) {
                return Some(entry);
            }
        }
        self.entries.first()
    }

    /// Registers or updates an entry directly from a `mascot_manifest.json` path.
    pub fn register_manifest(
        &mut self,
        avatar_id: &str,
        manifest_path: impl AsRef<Path>,
    ) -> Result<&AvatarEntry, SchemaError> {
        let manifest_path = manifest_path.as_ref();
        let dir_path = manifest_path
            .parent()
            .ok_or_else(|| {
                SchemaError::Validation("manifest_path has no parent directory".to_owned())
            })?
            .to_path_buf();

        let entry = Self::load_entry(avatar_id, &dir_path, manifest_path)?;
        if let Some(idx) = self.entries.iter().position(|e| e.avatar_id == avatar_id) {
            self.entries[idx] = entry;
            Ok(&self.entries[idx])
        } else {
            self.entries.push(entry);
            Ok(self.entries.last().expect("entry was just pushed"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn write_avatar_package(root: &Path, id: &str, name: &str, with_glb: bool) {
        let dir = root.join(id);
        fs::create_dir_all(&dir).expect("create avatar dir");
        let manifest = MascotManifest {
            avatar_name: name.to_owned(),
            ..Default::default()
        };
        manifest
            .save(dir.join(MANIFEST_FILE_NAME))
            .expect("save manifest");
        if with_glb {
            fs::write(dir.join("avatar.glb"), b"glTF").expect("write dummy glb");
        }
    }

    #[test]
    fn scan_handles_missing_root_and_discovers_valid_entries() {
        let dir = tempdir().expect("create tempdir");
        let missing_root = dir.path().join("avatars_missing");
        let empty = AvatarRegistry::scan(&missing_root).expect("missing dir should succeed");
        assert!(empty.entries().is_empty());

        let root = dir.path().join("avatars");
        write_avatar_package(&root, "avatar_b", "Beta", true);
        write_avatar_package(&root, "avatar_a", "Alpha", true);
        write_avatar_package(&root, "broken_no_glb", "Broken", false);

        let registry = AvatarRegistry::scan(&root).expect("scan registry");
        assert_eq!(registry.entries().len(), 2);
        assert_eq!(registry.entries()[0].avatar_id, "avatar_a");
        assert_eq!(registry.entries()[1].avatar_id, "avatar_b");
        assert_eq!(
            registry
                .default_or_first("avatar_b")
                .map(|e| e.avatar_id.as_str()),
            Some("avatar_b")
        );
        assert_eq!(
            registry
                .default_or_first("unknown")
                .map(|e| e.avatar_id.as_str()),
            Some("avatar_a")
        );
    }
}
