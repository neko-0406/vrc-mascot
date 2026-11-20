use yaml_rust2::yaml::Hash;
use yaml_rust2::Yaml;

use crate::indexer::UnityRef;

/// Looks up a string key in a `yaml_rust2::yaml::Hash`.
pub fn hash_get<'a>(hash: &'a Hash, key: &str) -> Option<&'a Yaml> {
    hash.get(&Yaml::String(key.to_owned()))
}

/// Looks up a string key in any `Yaml::Hash`.
pub fn yaml_get<'a>(yaml: &'a Yaml, key: &str) -> Option<&'a Yaml> {
    yaml.as_hash().and_then(|h| hash_get(h, key))
}

/// Extracts a string from `Yaml` (coercing integers/reals/bools if needed).
pub fn yaml_as_string(yaml: &Yaml) -> Option<String> {
    match yaml {
        Yaml::String(s) => Some(s.clone()),
        Yaml::Integer(i) => Some(i.to_string()),
        Yaml::Real(s) => Some(s.clone()),
        Yaml::Boolean(b) => Some(if *b { "true" } else { "false" }.to_owned()),
        _ => None,
    }
}

/// Extracts an `i64` from `Yaml`.
pub fn yaml_as_i64(yaml: &Yaml) -> Option<i64> {
    match yaml {
        Yaml::Integer(i) => Some(*i),
        Yaml::Real(s) => s.trim().parse::<f64>().ok().map(|v| v as i64),
        Yaml::String(s) => s.trim().parse::<i64>().ok(),
        Yaml::Boolean(b) => Some(i64::from(*b)),
        _ => None,
    }
}

/// Extracts an `f32` from `Yaml`.
pub fn yaml_as_f32(yaml: &Yaml) -> Option<f32> {
    match yaml {
        Yaml::Real(s) => s.trim().parse::<f32>().ok(),
        Yaml::Integer(i) => Some(*i as f32),
        Yaml::String(s) => s.trim().parse::<f32>().ok(),
        Yaml::Boolean(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

/// Extracts a `bool` from `Yaml` (Unity often stores booleans as `0`/`1`).
pub fn yaml_as_bool(yaml: &Yaml) -> Option<bool> {
    match yaml {
        Yaml::Boolean(b) => Some(*b),
        Yaml::Integer(i) => Some(*i != 0),
        Yaml::Real(s) => s.trim().parse::<f32>().ok().map(|v| v != 0.0),
        Yaml::String(s) => match s.trim().to_ascii_lowercase().as_str() {
            "true" | "1" => Some(true),
            "false" | "0" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

/// Extracts a `[f32; 3]` from `{x, y, z}` YAML mapping.
pub fn yaml_as_vec3(yaml: &Yaml) -> Option<[f32; 3]> {
    let h = yaml.as_hash()?;
    let x = hash_get(h, "x").and_then(yaml_as_f32).unwrap_or(0.0);
    let y = hash_get(h, "y").and_then(yaml_as_f32).unwrap_or(0.0);
    let z = hash_get(h, "z").and_then(yaml_as_f32).unwrap_or(0.0);
    Some([x, y, z])
}

/// Extracts a `[f32; 4]` quaternion from `{x, y, z, w}` YAML mapping.
pub fn yaml_as_quat(yaml: &Yaml) -> Option<[f32; 4]> {
    let h = yaml.as_hash()?;
    let x = hash_get(h, "x").and_then(yaml_as_f32).unwrap_or(0.0);
    let y = hash_get(h, "y").and_then(yaml_as_f32).unwrap_or(0.0);
    let z = hash_get(h, "z").and_then(yaml_as_f32).unwrap_or(0.0);
    let w = hash_get(h, "w").and_then(yaml_as_f32).unwrap_or(1.0);
    Some([x, y, z, w])
}

/// Extracts an `[f32; 4]` RGBA color from `{r, g, b, a}` YAML mapping.
pub fn yaml_as_color(yaml: &Yaml) -> Option<[f32; 4]> {
    let h = yaml.as_hash()?;
    let r = hash_get(h, "r").and_then(yaml_as_f32).unwrap_or(1.0);
    let g = hash_get(h, "g").and_then(yaml_as_f32).unwrap_or(1.0);
    let b = hash_get(h, "b").and_then(yaml_as_f32).unwrap_or(1.0);
    let a = hash_get(h, "a").and_then(yaml_as_f32).unwrap_or(1.0);
    Some([r, g, b, a])
}

/// Extracts an internal `__virtual_id` reference if present.
pub fn yaml_as_virtual_id(yaml: &Yaml) -> Option<usize> {
    let h = yaml.as_hash()?;
    hash_get(h, "__virtual_id")
        .and_then(yaml_as_i64)
        .and_then(|v| usize::try_from(v).ok())
}

/// Extracts a `UnityRef` from a YAML mapping (ignoring `__virtual_id` unless needed).
pub fn yaml_as_unity_ref(yaml: &Yaml) -> Option<UnityRef> {
    UnityRef::from_yaml(yaml)
}

/// Token in a Unity `m_Modifications[].propertyPath`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PropertyToken {
    Field(String),
    ArrayIndex(usize),
}

/// Splits a Unity `propertyPath` (e.g. `m_Materials.Array.data[0]` or `m_LocalPosition.x`) into tokens.
pub fn tokenize_property_path(path: &str) -> Vec<PropertyToken> {
    let mut tokens = Vec::new();
    let segments: Vec<&str> = path.split('.').collect();
    let mut i = 0;
    while i < segments.len() {
        let seg = segments[i];
        if seg == "Array" && i + 1 < segments.len() {
            if let Some(rest) = segments[i + 1].strip_prefix("data[") {
                if let Some(idx_str) = rest.strip_suffix(']') {
                    if let Ok(idx) = idx_str.parse::<usize>() {
                        tokens.push(PropertyToken::ArrayIndex(idx));
                        i += 2;
                        continue;
                    }
                }
            }
        }
        if let Some(bracket_pos) = seg.find('[') {
            let field = &seg[..bracket_pos];
            if !field.is_empty() {
                tokens.push(PropertyToken::Field(field.to_owned()));
            }
            if let Some(idx_str) = seg[bracket_pos + 1..].strip_suffix(']') {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    tokens.push(PropertyToken::ArrayIndex(idx));
                }
            }
        } else if !seg.is_empty() {
            tokens.push(PropertyToken::Field(seg.to_owned()));
        }
        i += 1;
    }
    tokens
}

/// Overwrites a nested property in `root` following `tokens`, creating intermediate hashes/arrays as needed.
pub fn set_yaml_by_tokens(root: &mut Yaml, tokens: &[PropertyToken], value: Yaml) {
    if tokens.is_empty() {
        *root = value;
        return;
    }

    match &tokens[0] {
        PropertyToken::Field(key) => {
            if !matches!(root, Yaml::Hash(_)) {
                *root = Yaml::Hash(Hash::new());
            }
            if let Yaml::Hash(map) = root {
                let k = Yaml::String(key.clone());
                if tokens.len() == 1 {
                    map.insert(k, value);
                } else {
                    if !map.contains_key(&k) {
                        let next_default = match &tokens[1] {
                            PropertyToken::Field(_) => Yaml::Hash(Hash::new()),
                            PropertyToken::ArrayIndex(_) => Yaml::Array(Vec::new()),
                        };
                        map.insert(k.clone(), next_default);
                    }
                    if let Some(child) = map.get_mut(&k) {
                        set_yaml_by_tokens(child, &tokens[1..], value);
                    }
                }
            }
        }
        PropertyToken::ArrayIndex(idx) => {
            if !matches!(root, Yaml::Array(_)) {
                *root = Yaml::Array(Vec::new());
            }
            if let Yaml::Array(arr) = root {
                if *idx >= arr.len() {
                    arr.resize(*idx + 1, Yaml::Null);
                }
                if tokens.len() == 1 {
                    arr[*idx] = value;
                } else {
                    if matches!(arr[*idx], Yaml::Null) {
                        arr[*idx] = match &tokens[1] {
                            PropertyToken::Field(_) => Yaml::Hash(Hash::new()),
                            PropertyToken::ArrayIndex(_) => Yaml::Array(Vec::new()),
                        };
                    }
                    set_yaml_by_tokens(&mut arr[*idx], &tokens[1..], value);
                }
            }
        }
    }
}
