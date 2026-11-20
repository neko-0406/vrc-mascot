use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use vrc_schema::ShaderFamily;
use yaml_rust2::YamlLoader;

use crate::error::ExtractorError;

/// Known Modular Avatar component categories resolved via script GUID or field signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaComponentKind {
    MergeArmature,
    MenuInstaller,
    MenuItem,
    Parameters,
    MeshSettings,
    BoneProxy,
    MergeAnimator,
}

/// Known VRChat SDK component categories resolved via script GUID or field signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VrcComponentKind {
    AvatarDescriptor,
    PhysBone,
    PhysBoneCollider,
    ExpressionsMenu,
    ExpressionParameters,
}

/// Built-in fallback GUIDs for Modular Avatar components (`nadena.dev.modular-avatar`).
const KNOWN_MA_GUIDS: &[(&str, MaComponentKind)] = &[
    (
        "c7576e54b19f47b785f4b3e5b3f8c2a1",
        MaComponentKind::MergeArmature,
    ),
    (
        "87131919c349c94418193d803e0690f7",
        MaComponentKind::MergeArmature,
    ),
    (
        "e4b5b3e4a9a24c7ea85f10d3f5e7c901",
        MaComponentKind::MenuInstaller,
    ),
    (
        "5229166a4122245429d53e384e6f2e2b",
        MaComponentKind::MenuInstaller,
    ),
    (
        "b78d3c1a2e4f4f9b8c7d6e5a4b3c2d1e",
        MaComponentKind::MenuItem,
    ),
    (
        "09c514275b2f8084d95a45f3e2e1d57d",
        MaComponentKind::MenuItem,
    ),
    (
        "f1a2b3c4d5e64f7a8b9c0d1e2f3a4b5c",
        MaComponentKind::Parameters,
    ),
    (
        "9a7f34e8c45646f48f5c7a9d35b7e892",
        MaComponentKind::Parameters,
    ),
    (
        "3b7e2f1c4d5a4b8c9d0e1f2a3b4c5d6e",
        MaComponentKind::MeshSettings,
    ),
    (
        "4c8f3a2d5e6b4c9d0e1f2a3b4c5d6e7f",
        MaComponentKind::BoneProxy,
    ),
    (
        "6e0b5c4f7a8d4e1f2a3b4c5d6e7f8a9b",
        MaComponentKind::MergeAnimator,
    ),
];

/// Built-in fallback GUIDs for VRChat SDK components.
const KNOWN_VRC_GUIDS: &[(&str, VrcComponentKind)] = &[
    // VRCSDK3A / VRCSDK3 compiled assembly GUIDs are often paired with specific fileIDs;
    // these fallback GUIDs handle source-based or well-known SDK metadata entries.
    (
        "5306449567914474e8b191844e865571",
        VrcComponentKind::AvatarDescriptor,
    ),
    (
        "5a34e5a04416d9c4d97b96b0502c03d9",
        VrcComponentKind::PhysBone,
    ),
    (
        "f0b817025f81d1841b771c3a85826d2c",
        VrcComponentKind::PhysBoneCollider,
    ),
];

/// Built-in fallback shader GUIDs -> `ShaderFamily`.
const KNOWN_SHADER_GUIDS: &[(&str, ShaderFamily)] = &[
    ("3c761f84f86b66f4fa2a6c7b45a1e8f0", ShaderFamily::LilToon),
    ("4e2e3b6f7c8a4b2d9e0f1a2b3c4d5e6f", ShaderFamily::Poiyomi),
    ("5f3f4c7a8d9b4c3e0f1a2b3c4d5e6f7a", ShaderFamily::Uts),
];

/// Index of all Unity assets discovered via `.meta` files under `Assets/`, `Packages/`, and `Library/PackageCache/`.
#[derive(Debug, Clone, Default)]
pub struct AssetIndex {
    project_dir: PathBuf,
    guid_to_path: HashMap<String, PathBuf>,
    rel_path_to_guid: HashMap<String, String>,
    ma_script_guids: HashMap<String, MaComponentKind>,
    vrc_script_guids: HashMap<String, VrcComponentKind>,
    shader_guids: HashMap<String, ShaderFamily>,
}

impl AssetIndex {
    /// Recursively scans `project_dir/Assets`, `project_dir/Packages`, and optionally `project_dir/Library/PackageCache`
    /// for `.meta` files, dynamically discovering script and shader GUIDs.
    pub fn build(project_dir: impl AsRef<Path>) -> Result<Self, ExtractorError> {
        let project_dir = project_dir.as_ref().to_path_buf();
        if !project_dir.is_dir() {
            return Err(ExtractorError::InvalidInput(format!(
                "Unity project directory does not exist: {}",
                project_dir.display()
            )));
        }

        let mut index = Self {
            project_dir: project_dir.clone(),
            ..Default::default()
        };

        for (guid, kind) in KNOWN_MA_GUIDS {
            index.ma_script_guids.insert((*guid).to_owned(), *kind);
        }
        for (guid, kind) in KNOWN_VRC_GUIDS {
            index.vrc_script_guids.insert((*guid).to_owned(), *kind);
        }
        for (guid, fam) in KNOWN_SHADER_GUIDS {
            index.shader_guids.insert((*guid).to_owned(), *fam);
        }

        let scan_roots = [
            (project_dir.join("Assets"), true),
            (project_dir.join("Packages"), true),
            (project_dir.join("Library").join("PackageCache"), false),
        ];

        for (root, overwrite) in scan_roots {
            if root.is_dir() {
                index.scan_dir(&root, overwrite)?;
            }
        }

        Ok(index)
    }

    fn scan_dir(&mut self, dir: &Path, overwrite: bool) -> Result<(), ExtractorError> {
        let entries = fs::read_dir(dir)?;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                self.scan_dir(&path, overwrite)?;
            } else if path.extension().and_then(|e| e.to_str()) == Some("meta") {
                self.index_meta_file(&path, overwrite)?;
            }
        }
        Ok(())
    }

    fn index_meta_file(&mut self, meta_path: &Path, overwrite: bool) -> Result<(), ExtractorError> {
        let Ok(content) = fs::read_to_string(meta_path) else {
            return Ok(());
        };
        let Some(guid) = extract_guid_from_meta(&content) else {
            return Ok(());
        };

        let asset_path = meta_path_to_asset_path(meta_path);
        if !overwrite && self.guid_to_path.contains_key(&guid) {
            return Ok(());
        }

        if let Ok(rel) = asset_path.strip_prefix(&self.project_dir) {
            let rel_norm = normalize_rel_path(rel);
            self.rel_path_to_guid.insert(rel_norm, guid.clone());
        }

        self.classify_asset_by_path(&guid, &asset_path);
        self.guid_to_path.insert(guid, asset_path);
        Ok(())
    }

    fn classify_asset_by_path(&mut self, guid: &str, asset_path: &Path) {
        let Some(file_name) = asset_path.file_name().and_then(|n| n.to_str()) else {
            return;
        };
        let lower_name = file_name.to_ascii_lowercase();

        if lower_name.ends_with(".cs") {
            let stem = file_name.strip_suffix(".cs").unwrap_or(file_name);
            if let Some(ma) = classify_ma_script_name(stem) {
                self.ma_script_guids.insert(guid.to_owned(), ma);
            }
            if let Some(vrc) = classify_vrc_script_name(stem) {
                self.vrc_script_guids.insert(guid.to_owned(), vrc);
            }
        } else if lower_name.ends_with(".shader") {
            let shader_header = read_shader_header_name(asset_path);
            let combined = format!(
                "{} {}",
                lower_name,
                shader_header
                    .as_deref()
                    .unwrap_or_default()
                    .to_ascii_lowercase()
            );
            if let Some(family) = classify_shader_name(&combined) {
                self.shader_guids.insert(guid.to_owned(), family);
            }
        }
    }

    /// Returns the Unity project root directory.
    pub fn project_dir(&self) -> &Path {
        &self.project_dir
    }

    /// Resolves a 32-char hex GUID (case-insensitive) to its asset path on disk.
    pub fn resolve_guid(&self, guid: &str) -> Option<&PathBuf> {
        let lower = guid.trim().to_ascii_lowercase();
        self.guid_to_path.get(&lower)
    }

    /// Reverse lookup from a project-relative asset path (forward or back slashes) to its GUID.
    pub fn guid_for_relative_path(&self, rel_path: impl AsRef<Path>) -> Option<&str> {
        let norm = normalize_rel_path(rel_path.as_ref());
        self.rel_path_to_guid.get(&norm).map(String::as_str)
    }

    /// Returns all indexed `(guid, path)` pairs.
    pub fn iter_assets(&self) -> impl Iterator<Item = (&str, &PathBuf)> {
        self.guid_to_path.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Returns the number of indexed assets.
    pub fn len(&self) -> usize {
        self.guid_to_path.len()
    }

    /// Returns `true` if no assets were indexed.
    pub fn is_empty(&self) -> bool {
        self.guid_to_path.is_empty()
    }

    /// Identifies a Modular Avatar script GUID if known from `.cs.meta` or built-in fallback table.
    pub fn ma_kind_by_guid(&self, guid: &str) -> Option<MaComponentKind> {
        self.ma_script_guids
            .get(&guid.trim().to_ascii_lowercase())
            .copied()
    }

    /// Identifies a VRChat SDK script GUID if known from `.cs.meta` or built-in fallback table.
    pub fn vrc_kind_by_guid(&self, guid: &str) -> Option<VrcComponentKind> {
        self.vrc_script_guids
            .get(&guid.trim().to_ascii_lowercase())
            .copied()
    }

    /// Identifies a `ShaderFamily` from a shader asset GUID.
    pub fn shader_family_by_guid(&self, guid: &str) -> Option<ShaderFamily> {
        let lower = guid.trim().to_ascii_lowercase();
        if let Some(family) = self.shader_guids.get(&lower) {
            return Some(*family);
        }
        if let Some(path) = self.guid_to_path.get(&lower) {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                return classify_shader_name(&name.to_ascii_lowercase());
            }
        }
        None
    }

    /// Resolves a project-relative or absolute path into an absolute path inside `project_dir`.
    pub fn resolve_scene_path(&self, scene_path: &Path) -> PathBuf {
        if scene_path.is_absolute() {
            scene_path.to_path_buf()
        } else {
            self.project_dir.join(scene_path)
        }
    }
}

/// Extracts `guid: <32hex>` from a Unity `.meta` file.
pub fn extract_guid_from_meta(meta_content: &str) -> Option<String> {
    let trimmed_bom = meta_content.trim_start_matches('\u{feff}');
    for line in trimmed_bom.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("guid:") {
            let guid = rest.trim().to_ascii_lowercase();
            if !guid.is_empty() {
                return Some(guid);
            }
        }
    }
    None
}

/// Parses `ModelImporter.humanDescription.human` from an `.fbx.meta` file into `humanName -> boneName`.
pub fn parse_fbx_meta_human_description(
    meta_path: &Path,
) -> Result<BTreeMap<String, String>, ExtractorError> {
    let raw = fs::read_to_string(meta_path)?;
    parse_fbx_meta_human_description_str(&raw, meta_path)
}

/// Parses `ModelImporter.humanDescription.human` from raw `.fbx.meta` YAML text.
pub fn parse_fbx_meta_human_description_str(
    raw: &str,
    path_for_error: &Path,
) -> Result<BTreeMap<String, String>, ExtractorError> {
    let clean = raw.trim_start_matches('\u{feff}');
    let docs = YamlLoader::load_from_str(clean).map_err(|e| ExtractorError::YamlParse {
        path: path_for_error.to_path_buf(),
        message: e.to_string(),
    })?;

    let mut result = BTreeMap::new();
    let Some(doc) = docs.first() else {
        return Ok(result);
    };

    let human_list = &doc["ModelImporter"]["humanDescription"]["human"];
    if let Some(entries) = human_list.as_vec() {
        for entry in entries {
            let human_name = entry["humanName"].as_str().map(str::trim).unwrap_or("");
            let bone_name = entry["boneName"].as_str().map(str::trim).unwrap_or("");
            if !human_name.is_empty() && !bone_name.is_empty() {
                result.insert(human_name.to_owned(), bone_name.to_owned());
            }
        }
    }

    Ok(result)
}

fn meta_path_to_asset_path(meta_path: &Path) -> PathBuf {
    let s = meta_path.to_string_lossy();
    if let Some(stripped) = s.strip_suffix(".meta") {
        PathBuf::from(stripped)
    } else {
        meta_path.to_path_buf()
    }
}

fn normalize_rel_path(path: &Path) -> String {
    path.to_string_lossy()
        .replace('\\', "/")
        .trim_start_matches("./")
        .to_ascii_lowercase()
}

fn classify_ma_script_name(stem: &str) -> Option<MaComponentKind> {
    let lower = stem.to_ascii_lowercase();
    match lower.as_str() {
        "modularavatarmergearmature" => Some(MaComponentKind::MergeArmature),
        "modularavatarmenuinstaller" => Some(MaComponentKind::MenuInstaller),
        "modularavatarmenuitem" => Some(MaComponentKind::MenuItem),
        "modularavatarparameters" => Some(MaComponentKind::Parameters),
        "modularavatarmeshsettings" => Some(MaComponentKind::MeshSettings),
        "modularavatarboneproxy" => Some(MaComponentKind::BoneProxy),
        "modularavatarmergeanimator" => Some(MaComponentKind::MergeAnimator),
        _ => None,
    }
}

fn classify_vrc_script_name(stem: &str) -> Option<VrcComponentKind> {
    let lower = stem.to_ascii_lowercase();
    match lower.as_str() {
        "vrcavatardescriptor" | "vrc_avatardescriptor" => Some(VrcComponentKind::AvatarDescriptor),
        "vrcphysbone" => Some(VrcComponentKind::PhysBone),
        "vrcphysbonecollider" => Some(VrcComponentKind::PhysBoneCollider),
        "vrcexpressionsmenu" => Some(VrcComponentKind::ExpressionsMenu),
        "vrcexpressionparameters" => Some(VrcComponentKind::ExpressionParameters),
        _ => None,
    }
}

/// Classifies a shader name or path fragment into a `ShaderFamily`.
pub fn classify_shader_name(lower_name: &str) -> Option<ShaderFamily> {
    if lower_name.contains("liltoon") || lower_name.contains("lts/") || lower_name.contains("_lil")
    {
        Some(ShaderFamily::LilToon)
    } else if lower_name.contains("poiyomi") || lower_name.contains(".poiyomi") {
        Some(ShaderFamily::Poiyomi)
    } else if lower_name.contains("unitychantoon")
        || lower_name.contains("uts")
        || lower_name.contains("toon_double")
    {
        Some(ShaderFamily::Uts)
    } else if lower_name == "standard" || lower_name.contains("standard.shader") {
        Some(ShaderFamily::Standard)
    } else {
        None
    }
}

fn read_shader_header_name(shader_path: &Path) -> Option<String> {
    let raw = fs::read_to_string(shader_path).ok()?;
    for line in raw.lines().take(16) {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("Shader") {
            let rest = rest.trim();
            if let Some(quoted) = rest.strip_prefix('"') {
                if let Some(end) = quoted.find('"') {
                    return Some(quoted[..end].to_owned());
                }
            }
        }
    }
    None
}
