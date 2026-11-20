use std::collections::HashMap;
use std::fs;
use std::path::Path;

use yaml_rust2::{Yaml, YamlLoader};

use crate::error::ExtractorError;

/// Reference to a Unity object within the same file or an external asset by `(fileID, guid)`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct UnityRef {
    pub file_id: i64,
    pub guid: Option<String>,
    pub asset_type: Option<i64>,
}

impl UnityRef {
    /// Returns `true` if this reference points to null (`fileID == 0` and no GUID).
    pub fn is_null(&self) -> bool {
        self.file_id == 0 && self.guid.is_none()
    }

    /// Parses a `{fileID: ..., guid: ..., type: ...}` YAML mapping.
    pub fn from_yaml(yaml: &Yaml) -> Option<Self> {
        let hash = yaml.as_hash()?;
        let file_id = yaml_get_key(hash, "fileID")
            .and_then(yaml_as_i64)
            .unwrap_or(0);
        let guid = yaml_get_key(hash, "guid")
            .and_then(|v| match v {
                Yaml::String(s) => Some(s.clone()),
                Yaml::Integer(i) if *i >= 0 => Some(format!("{i:032}")),
                Yaml::Real(s) => Some(s.clone()),
                _ => None,
            })
            .map(|s| s.trim().to_ascii_lowercase())
            .filter(|s| !s.is_empty());
        let asset_type = yaml_get_key(hash, "type").and_then(yaml_as_i64);
        Some(Self {
            file_id,
            guid,
            asset_type,
        })
    }
}

/// Stripped placeholder provenance linking a scene/variant document to its source prefab object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrippedRef {
    pub corresponding_source: UnityRef,
    pub prefab_instance_file_id: i64,
}

/// Single Unity YAML document (`--- !u!<class_id> &<file_id> [stripped]`).
#[derive(Debug, Clone)]
pub struct UnityDocument {
    pub class_id: i64,
    pub file_id: i64,
    pub stripped: bool,
    pub root_key: String,
    pub data: Yaml,
    pub stripped_ref: Option<StrippedRef>,
}

/// Parsed multi-document Unity YAML file indexed by `fileID`.
#[derive(Debug, Clone)]
pub struct UnityYamlFile {
    pub documents: Vec<UnityDocument>,
    by_file_id: HashMap<i64, usize>,
}

impl UnityYamlFile {
    /// Reads and parses a Unity YAML file from disk.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ExtractorError> {
        let path = path.as_ref();
        let raw = fs::read_to_string(path)?;
        Self::parse_str(&raw, path)
    }

    /// Parses multi-document Unity YAML text into indexed `UnityDocument`s.
    pub fn parse_str(raw: &str, path_for_error: &Path) -> Result<Self, ExtractorError> {
        let clean = raw.trim_start_matches('\u{feff}');
        let chunks = split_unity_yaml_chunks(clean);

        let mut documents = Vec::with_capacity(chunks.len());
        let mut by_file_id = HashMap::with_capacity(chunks.len());

        for chunk in chunks {
            let trimmed_body = chunk.body.trim();
            let parsed_root = if trimmed_body.is_empty() {
                Yaml::Null
            } else {
                let mut docs = YamlLoader::load_from_str(trimmed_body).map_err(|e| {
                    ExtractorError::YamlParse {
                        path: path_for_error.to_path_buf(),
                        message: format!("chunk !u!{} &{}: {}", chunk.class_id, chunk.file_id, e),
                    }
                })?;
                docs.pop().unwrap_or(Yaml::Null)
            };

            let (root_key, data) = unwrap_root_mapping(parsed_root);
            let stripped_ref = if chunk.stripped {
                let corr =
                    UnityRef::from_yaml(&data["m_CorrespondingSourceObject"]).unwrap_or_default();
                let pi = UnityRef::from_yaml(&data["m_PrefabInstance"])
                    .map(|r| r.file_id)
                    .unwrap_or(0);
                Some(StrippedRef {
                    corresponding_source: corr,
                    prefab_instance_file_id: pi,
                })
            } else {
                None
            };

            let idx = documents.len();
            by_file_id.insert(chunk.file_id, idx);
            documents.push(UnityDocument {
                class_id: chunk.class_id,
                file_id: chunk.file_id,
                stripped: chunk.stripped,
                root_key,
                data,
                stripped_ref,
            });
        }

        Ok(Self {
            documents,
            by_file_id,
        })
    }

    /// Looks up a document by its `fileID`.
    pub fn get(&self, file_id: i64) -> Option<&UnityDocument> {
        self.by_file_id
            .get(&file_id)
            .and_then(|&idx| self.documents.get(idx))
    }

    /// Resolves a local `fileID`, following `stripped` placeholders to `(target_prefab_instance_id, source_ref)` if stripped.
    pub fn resolve_stripped(&self, file_id: i64) -> Option<&StrippedRef> {
        self.get(file_id).and_then(|doc| doc.stripped_ref.as_ref())
    }
}

struct RawChunk {
    class_id: i64,
    file_id: i64,
    stripped: bool,
    body: String,
}

fn split_unity_yaml_chunks(input: &str) -> Vec<RawChunk> {
    let mut chunks = Vec::new();
    let mut current_header: Option<(i64, i64, bool)> = None;
    let mut current_body = String::new();

    for line in input.lines() {
        let trimmed_end = line.trim_end_matches('\r');
        if let Some((class_id, file_id, stripped)) = parse_unity_doc_header(trimmed_end) {
            if let Some((prev_class, prev_file, prev_stripped)) = current_header.take() {
                chunks.push(RawChunk {
                    class_id: prev_class,
                    file_id: prev_file,
                    stripped: prev_stripped,
                    body: std::mem::take(&mut current_body),
                });
            }
            current_header = Some((class_id, file_id, stripped));
            current_body.clear();
            continue;
        }

        if current_header.is_some() {
            current_body.push_str(trimmed_end);
            current_body.push('\n');
        } else {
            // Skip %YAML / %TAG preamble before the first `--- !u!...` header.
            let t = trimmed_end.trim_start();
            if !t.is_empty() && !t.starts_with('%') && !t.starts_with('#') {
                // Fallback for plain YAML files without `--- !u!` header (e.g. standalone assets)
                current_header = Some((0, 0, false));
                current_body.push_str(trimmed_end);
                current_body.push('\n');
            }
        }
    }

    if let Some((prev_class, prev_file, prev_stripped)) = current_header.take() {
        chunks.push(RawChunk {
            class_id: prev_class,
            file_id: prev_file,
            stripped: prev_stripped,
            body: current_body,
        });
    }

    chunks
}

fn parse_unity_doc_header(line: &str) -> Option<(i64, i64, bool)> {
    let rest = line.trim_start().strip_prefix("--- !u!")?;
    let mut tokens = rest.split_whitespace();
    let class_str = tokens.next()?;
    let file_token = tokens.next()?;
    let class_id = class_str.parse::<i64>().ok()?;
    let file_str = file_token.strip_prefix('&')?;
    let file_id = file_str.parse::<i64>().ok()?;
    let stripped = tokens.any(|tok| tok == "stripped");
    Some((class_id, file_id, stripped))
}

fn unwrap_root_mapping(yaml: Yaml) -> (String, Yaml) {
    if let Yaml::Hash(mut map) = yaml {
        if map.len() == 1 {
            if let Some((k, v)) = map.pop_front() {
                let key = k.as_str().unwrap_or("").to_owned();
                return (key, v);
            }
        }
        return (String::new(), Yaml::Hash(map));
    }
    (String::new(), yaml)
}

fn yaml_get_key<'a>(hash: &'a yaml_rust2::yaml::Hash, key: &str) -> Option<&'a Yaml> {
    hash.get(&Yaml::String(key.to_owned()))
}

fn yaml_as_i64(yaml: &Yaml) -> Option<i64> {
    match yaml {
        Yaml::Integer(i) => Some(*i),
        Yaml::Real(s) => s.parse::<i64>().ok(),
        Yaml::String(s) => s.trim().parse::<i64>().ok(),
        _ => None,
    }
}
