use std::collections::HashMap;
use std::path::Path;

use vrc_schema::{AlphaMode, CullMode, MaterialDef, MaterialParams, ShaderFamily};
use yaml_rust2::Yaml;

use crate::error::ExtractorError;
use crate::indexer::{classify_shader_name, AssetIndex, UnityRef, UnityYamlFile};
use crate::yaml_parser::helpers::{
    yaml_as_color, yaml_as_f32, yaml_as_i64, yaml_as_string, yaml_get,
};

/// Extracted material properties and referenced texture GUIDs from a Unity `.mat` file.
#[derive(Debug, Clone)]
pub struct ExtractedMaterial {
    pub def: MaterialDef,
    pub main_tex_guid: Option<String>,
    pub shadow_tex_guid: Option<String>,
    pub matcap_tex_guid: Option<String>,
    pub emission_tex_guid: Option<String>,
    pub fur_mask_tex_guid: Option<String>,
}

/// Parses a Unity `.mat` file and maps shader properties to `vrc_schema::MaterialDef`.
pub fn parse_material_asset(
    mat_path: &Path,
    gltf_material_index: usize,
    index: &AssetIndex,
) -> Result<ExtractedMaterial, ExtractorError> {
    let yaml_file = UnityYamlFile::load(mat_path)?;
    let doc = yaml_file
        .documents
        .iter()
        .find(|d| d.class_id == 21 || d.root_key == "Material")
        .or_else(|| yaml_file.documents.first())
        .ok_or_else(|| ExtractorError::YamlParse {
            path: mat_path.to_path_buf(),
            message: "no Material document found".to_owned(),
        })?;

    let data = &doc.data;
    let name = yaml_get(data, "m_Name")
        .and_then(yaml_as_string)
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| {
            mat_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("Material")
                .to_owned()
        });

    let shader_guid = yaml_get(data, "m_Shader")
        .and_then(UnityRef::from_yaml)
        .and_then(|r| r.guid);

    let custom_rq = yaml_get(data, "m_CustomRenderQueue")
        .and_then(yaml_as_i64)
        .unwrap_or(-1);

    let saved = &data["m_SavedProperties"];
    let tex_envs = collect_saved_property_map(&saved["m_TexEnvs"]);
    let floats = collect_saved_property_map(&saved["m_Floats"]);
    let colors = collect_saved_property_map(&saved["m_Colors"]);

    let shader_name_hint = shader_guid
        .as_deref()
        .and_then(|g| index.resolve_guid(g))
        .and_then(|p| p.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();

    let shader_family = detect_shader_family(
        shader_guid.as_deref(),
        &shader_name_hint,
        &floats,
        &colors,
        &tex_envs,
        index,
    );

    let mut params = MaterialParams::default();
    let main_tex_guid =
        extract_tex_guid(&tex_envs, "_MainTex").or_else(|| extract_tex_guid(&tex_envs, "_BaseMap"));
    let mut shadow_tex_guid = None;
    let mut matcap_tex_guid = None;
    let emission_tex_guid;
    let mut fur_mask_tex_guid = None;

    match shader_family {
        ShaderFamily::LilToon => {
            if let Some(c) = get_color(&colors, "_Color") {
                params.main_color = c;
            }
            if let Some(c) = get_color(&colors, "_ShadowColor") {
                params.shadow_color_1 = c;
            }
            if let Some(c) = get_color(&colors, "_Shadow2ndColor") {
                params.shadow_color_2 = c;
            }
            if let Some(v) = get_float(&floats, "_ShadowBorder") {
                params.shadow_border = v;
            }
            if let Some(v) = get_float(&floats, "_ShadowBlur") {
                params.shadow_blur = v;
            }
            if let Some(v) = get_float(&floats, "_Shadow2ndBorder") {
                params.shadow_2nd_border = v;
            }
            if let Some(v) = get_float(&floats, "_Shadow2ndBlur") {
                params.shadow_2nd_blur = v;
            }
            shadow_tex_guid = extract_tex_guid(&tex_envs, "_ShadowColorTex");

            params.rim_enable = get_bool(&floats, "_UseRim").unwrap_or(false);
            if let Some(c) = get_color(&colors, "_RimColor") {
                params.rim_color = c;
            }
            if let Some(v) = get_float(&floats, "_RimBorder") {
                params.rim_border = v;
            }
            if let Some(v) = get_float(&floats, "_RimBlur") {
                params.rim_blur = v;
            }
            if let Some(v) = get_float(&floats, "_RimFresnelPower") {
                params.rim_fresnel_power = v;
            }

            params.matcap_enable = get_bool(&floats, "_UseMatCap").unwrap_or(false);
            if let Some(c) = get_color(&colors, "_MatCapColor") {
                params.matcap_color = c;
            }
            if let Some(v) = get_float(&floats, "_MatCapBlendMode") {
                params.matcap_blend_mode = v.max(0.0) as u32;
            }
            matcap_tex_guid = extract_tex_guid(&tex_envs, "_MatCapTex");

            params.emission_enable = get_bool(&floats, "_UseEmission").unwrap_or(false);
            if let Some(c) = get_color(&colors, "_EmissionColor") {
                params.emission_color = c;
            }
            emission_tex_guid = extract_tex_guid(&tex_envs, "_EmissionMap");

            let outline_width = get_float(&floats, "_OutlineWidth");
            params.outline_enable = get_bool(&floats, "_UseOutline").unwrap_or_else(|| {
                outline_width.is_some_and(|w| w > 0.0) || shader_name_hint.contains("outline")
            });
            if let Some(w) = outline_width {
                params.outline_width = w;
            }
            if let Some(c) = get_color(&colors, "_OutlineColor") {
                params.outline_color = c;
            }

            params.fur_enable =
                get_bool(&floats, "_UseFur").unwrap_or_else(|| shader_name_hint.contains("fur"));
            if let Some(v) =
                get_float(&floats, "_FurLength").or_else(|| get_float(&floats, "_FurVectorScale"))
            {
                params.fur_length = v.abs();
            }
            if let Some(v) = get_float(&floats, "_FurLayerNum") {
                params.fur_layers = (v.round() as i64).max(1) as u32;
            }
            if let Some(v) = get_float(&floats, "_FurGravity") {
                params.fur_gravity = v;
            }
            fur_mask_tex_guid = extract_tex_guid(&tex_envs, "_FurMask")
                .or_else(|| extract_tex_guid(&tex_envs, "_FurLengthMask"));

            params.audiolink_enable = get_bool(&floats, "_UseAudioLink")
                .or_else(|| get_bool(&floats, "_AudioLinkEnable"))
                .unwrap_or(false);
            if let Some(v) = get_float(&floats, "_AudioLinkBand") {
                params.audiolink_band = v.max(0.0) as u32;
            }
            if let Some(v) = get_float(&floats, "_AudioLinkStrength") {
                params.audiolink_strength = v;
            }
        }
        ShaderFamily::Poiyomi => {
            if let Some(c) = get_color(&colors, "_Color") {
                params.main_color = c;
            }
            if let Some(c) = get_color(&colors, "_ShadowColor")
                .or_else(|| get_color(&colors, "_1st_ShadeColor"))
                .or_else(|| get_color(&colors, "_ShadowTint"))
            {
                params.shadow_color_1 = c;
            }
            if let Some(v) =
                get_float(&floats, "_ShadowBorder").or_else(|| get_float(&floats, "_ShadowOffset"))
            {
                params.shadow_border = v;
            }
            if let Some(v) =
                get_float(&floats, "_ShadowBlur").or_else(|| get_float(&floats, "_ShadowFeather"))
            {
                params.shadow_blur = v;
            }

            params.rim_enable = get_bool(&floats, "_EnableRimLighting")
                .or_else(|| get_bool(&floats, "_RimLightEnable"))
                .unwrap_or_else(|| colors.contains_key("_RimLightColor"));
            if let Some(c) =
                get_color(&colors, "_RimLightColor").or_else(|| get_color(&colors, "_RimColor"))
            {
                params.rim_color = c;
            }
            if let Some(v) =
                get_float(&floats, "_RimWidth").or_else(|| get_float(&floats, "_RimBorder"))
            {
                params.rim_border = v;
            }
            if let Some(v) =
                get_float(&floats, "_RimPower").or_else(|| get_float(&floats, "_RimSharpness"))
            {
                params.rim_fresnel_power = v;
            }

            matcap_tex_guid = extract_tex_guid(&tex_envs, "_MatCap")
                .or_else(|| extract_tex_guid(&tex_envs, "_Matcap"));
            params.matcap_enable = get_bool(&floats, "_MatCapEnable")
                .or_else(|| get_bool(&floats, "_MatcapEnable"))
                .unwrap_or_else(|| matcap_tex_guid.is_some());
            if let Some(c) =
                get_color(&colors, "_MatCapColor").or_else(|| get_color(&colors, "_MatcapColor"))
            {
                params.matcap_color = c;
            }

            emission_tex_guid = extract_tex_guid(&tex_envs, "_EmissionMap");
            params.emission_enable = get_bool(&floats, "_EnableEmission")
                .or_else(|| get_bool(&floats, "_EmissionEnable"))
                .unwrap_or(false);
            if let Some(c) = get_color(&colors, "_EmissionColor") {
                params.emission_color = c;
            }

            let outline_w = get_float(&floats, "_OutlineWidth");
            params.outline_enable = get_bool(&floats, "_EnableOutline")
                .unwrap_or_else(|| outline_w.is_some_and(|w| w > 0.0));
            if let Some(w) = outline_w {
                params.outline_width = w;
            }
            if let Some(c) = get_color(&colors, "_OutlineColor") {
                params.outline_color = c;
            }

            params.audiolink_enable = get_bool(&floats, "_EnableAudioLink")
                .or_else(|| get_bool(&floats, "_AudioLink"))
                .unwrap_or(false);
            if let Some(v) = get_float(&floats, "_AudioLinkBand") {
                params.audiolink_band = v.max(0.0) as u32;
            }
            if let Some(v) = get_float(&floats, "_AudioLinkStrength") {
                params.audiolink_strength = v;
            }
        }
        ShaderFamily::Uts => {
            if let Some(c) =
                get_color(&colors, "_BaseColor").or_else(|| get_color(&colors, "_Color"))
            {
                params.main_color = c;
            }
            if let Some(c) = get_color(&colors, "_1st_ShadeColor") {
                params.shadow_color_1 = c;
            }
            if let Some(c) = get_color(&colors, "_2nd_ShadeColor") {
                params.shadow_color_2 = c;
            }
            if let Some(v) = get_float(&floats, "_BaseColor_Step") {
                params.shadow_border = v;
            }
            if let Some(v) = get_float(&floats, "_BaseShade_Feather") {
                params.shadow_blur = v;
            }
            if let Some(v) = get_float(&floats, "_ShadeColor_Step") {
                params.shadow_2nd_border = v;
            }
            if let Some(v) = get_float(&floats, "_1st2nd_Shades_Feather") {
                params.shadow_2nd_blur = v;
            }

            params.rim_enable = get_bool(&floats, "_RimLight").unwrap_or(false);
            if let Some(c) = get_color(&colors, "_RimLightColor") {
                params.rim_color = c;
            }
            if let Some(v) = get_float(&floats, "_RimLight_Power") {
                params.rim_fresnel_power = v;
            }

            matcap_tex_guid = extract_tex_guid(&tex_envs, "_MatCap_Sampler");
            params.matcap_enable = get_bool(&floats, "_MatCap").unwrap_or(false);

            emission_tex_guid = extract_tex_guid(&tex_envs, "_Emissive_Tex");
            if let Some(c) = get_color(&colors, "_Emissive_Color") {
                params.emission_color = c;
                if c[0] > 0.0 || c[1] > 0.0 || c[2] > 0.0 {
                    params.emission_enable = true;
                }
            }
            if emission_tex_guid.is_some() {
                params.emission_enable = true;
            }

            let outline_w = get_float(&floats, "_Outline_Width");
            params.outline_enable = outline_w.is_some_and(|w| w > 0.0);
            if let Some(w) = outline_w {
                params.outline_width = w;
            }
            if let Some(c) = get_color(&colors, "_Outline_Color") {
                params.outline_color = c;
            }
        }
        ShaderFamily::Standard => {
            if let Some(c) =
                get_color(&colors, "_Color").or_else(|| get_color(&colors, "_BaseColor"))
            {
                params.main_color = c;
            }
            emission_tex_guid = extract_tex_guid(&tex_envs, "_EmissionMap");
            if let Some(c) = get_color(&colors, "_EmissionColor") {
                params.emission_color = c;
                if c[0] > 0.0 || c[1] > 0.0 || c[2] > 0.0 {
                    params.emission_enable = true;
                }
            }
            if emission_tex_guid.is_some() {
                params.emission_enable = true;
            }
        }
    }

    if let Some(cutoff) = get_float(&floats, "_Cutoff") {
        params.alpha_cutoff = cutoff;
    }
    if let Some(cull) = get_float(&floats, "_Cull").or_else(|| get_float(&floats, "_CullMode")) {
        params.cull_mode = match cull.round() as i32 {
            0 => CullMode::None,
            1 => CullMode::Front,
            _ => CullMode::Back,
        };
    }

    let alpha_mode = infer_alpha_mode(&shader_name_hint, &floats, custom_rq);
    let render_queue = if custom_rq >= 0 {
        custom_rq as i32
    } else {
        match alpha_mode {
            AlphaMode::Opaque => 2000,
            AlphaMode::Mask => 2450,
            AlphaMode::Blend => 3000,
        }
    };

    Ok(ExtractedMaterial {
        def: MaterialDef {
            gltf_material_index,
            name,
            shader_family,
            render_queue,
            alpha_mode,
            params,
        },
        main_tex_guid,
        shadow_tex_guid,
        matcap_tex_guid,
        emission_tex_guid,
        fur_mask_tex_guid,
    })
}

fn detect_shader_family(
    shader_guid: Option<&str>,
    shader_name_hint: &str,
    floats: &HashMap<String, &Yaml>,
    colors: &HashMap<String, &Yaml>,
    tex_envs: &HashMap<String, &Yaml>,
    index: &AssetIndex,
) -> ShaderFamily {
    if let Some(guid) = shader_guid {
        if let Some(family) = index.shader_family_by_guid(guid) {
            return family;
        }
    }
    if let Some(family) = classify_shader_name(shader_name_hint) {
        return family;
    }

    // Fallback: property signature detection
    if floats.contains_key("_UseRim")
        || floats.contains_key("_UseMatCap")
        || floats.contains_key("_UseFur")
        || colors.contains_key("_Shadow2ndColor")
        || floats.contains_key("_ShadowBorder")
    {
        ShaderFamily::LilToon
    } else if floats.contains_key("_EnableRimLighting")
        || floats.contains_key("_EnableAudioLink")
        || tex_envs.contains_key("_ShadowGradient")
        || floats.contains_key("_ShadowStrength")
        || colors.contains_key("_RimLightColor") && floats.contains_key("_RimWidth")
    {
        ShaderFamily::Poiyomi
    } else if colors.contains_key("_1st_ShadeColor")
        || colors.contains_key("_2nd_ShadeColor")
        || floats.contains_key("_BaseColor_Step")
        || tex_envs.contains_key("_MatCap_Sampler")
    {
        ShaderFamily::Uts
    } else {
        ShaderFamily::Standard
    }
}

fn infer_alpha_mode(
    shader_name_hint: &str,
    floats: &HashMap<String, &Yaml>,
    custom_rq: i64,
) -> AlphaMode {
    if let Some(mode) = get_float(floats, "_TransparentMode")
        .or_else(|| get_float(floats, "_RenderingMode"))
        .or_else(|| get_float(floats, "_Mode"))
    {
        match mode.round() as i32 {
            1 => return AlphaMode::Mask,
            2 | 3 => return AlphaMode::Blend,
            _ => {}
        }
    }
    if shader_name_hint.contains("transparent") {
        return AlphaMode::Blend;
    }
    if shader_name_hint.contains("cutout") || get_bool(floats, "_AlphaCutout").unwrap_or(false) {
        return AlphaMode::Mask;
    }
    if custom_rq >= 2900 {
        AlphaMode::Blend
    } else if custom_rq >= 2400 {
        AlphaMode::Mask
    } else {
        AlphaMode::Opaque
    }
}

/// Unity serializes `m_TexEnvs`, `m_Floats`, `m_Colors` as a sequence of single-key mappings
/// (e.g. `- _MainTex: ...`), or occasionally as a flat mapping. This normalizes both into a map.
fn collect_saved_property_map(yaml: &Yaml) -> HashMap<String, &Yaml> {
    let mut out = HashMap::new();
    match yaml {
        Yaml::Array(items) => {
            for item in items {
                if let Some(h) = item.as_hash() {
                    for (k, v) in h {
                        if let Some(name) = k.as_str() {
                            out.insert(name.to_owned(), v);
                        }
                    }
                }
            }
        }
        Yaml::Hash(h) => {
            for (k, v) in h {
                if let Some(name) = k.as_str() {
                    out.insert(name.to_owned(), v);
                }
            }
        }
        _ => {}
    }
    out
}

fn extract_tex_guid(tex_envs: &HashMap<String, &Yaml>, key: &str) -> Option<String> {
    let entry = tex_envs.get(key)?;
    UnityRef::from_yaml(&entry["m_Texture"]).and_then(|r| r.guid)
}

fn get_float(floats: &HashMap<String, &Yaml>, key: &str) -> Option<f32> {
    floats.get(key).and_then(|y| yaml_as_f32(y))
}

fn get_bool(floats: &HashMap<String, &Yaml>, key: &str) -> Option<bool> {
    get_float(floats, key).map(|v| v > 0.5)
}

fn get_color(colors: &HashMap<String, &Yaml>, key: &str) -> Option<[f32; 4]> {
    colors.get(key).and_then(|y| yaml_as_color(y))
}
