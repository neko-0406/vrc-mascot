use std::collections::{HashMap, HashSet};
use std::f32::consts::PI;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use glam::Vec3;
use vrc_renderer::{
    preferred_surface_format, AudioLinkAnalyzer, AudioLinkCapture, AvatarModel, CameraRig,
    HitTestResult, MascotRenderer, PipelineCache, PipelineFlags, PipelineKey, RendererError,
    FFT_SIZE, FUR_WGSL, HIT_TEST_WGSL, OUTLINE_WGSL, TOON_WGSL,
};
use vrc_schema::{
    AlphaMode, AudioLinkData, CullMode, MascotManifest, MaterialDef, MaterialParams,
    PhysicsShaderConfig, QualityPreset, SCHEMA_VERSION,
};

#[test]
fn wgsl_shaders_have_valid_entry_points_and_bindings() {
    for (name, src) in [
        ("toon", TOON_WGSL),
        ("outline", OUTLINE_WGSL),
        ("fur", FUR_WGSL),
        ("hit_test", HIT_TEST_WGSL),
    ] {
        assert!(!src.trim().is_empty(), "{name} shader must not be empty");
        assert!(src.contains("fn vs_main"), "{name} missing vs_main");
        assert!(src.contains("fn fs_main"), "{name} missing fs_main");
        assert!(
            src.contains("FrameUniforms"),
            "{name} missing FrameUniforms"
        );
        assert!(src.contains("DrawUniforms"), "{name} missing DrawUniforms");
        assert!(
            src.contains("MaterialUniforms"),
            "{name} missing MaterialUniforms"
        );
        assert!(
            src.contains("@group(0) @binding(0)"),
            "{name} missing group 0 binding 0"
        );
        assert!(
            src.contains("@group(1) @binding(0)"),
            "{name} missing group 1 binding 0"
        );
        assert!(
            src.contains("@group(2) @binding(0)"),
            "{name} missing group 2 binding 0"
        );
    }
    assert!(TOON_WGSL.contains("audiolink_bands"));
    assert!(FUR_WGSL.contains("@builtin(instance_index)"));
    assert!(HIT_TEST_WGSL.contains("0.05"));
}

#[test]
fn pipeline_flags_and_key_hashing() {
    let mut flags = PipelineFlags::empty();
    assert_eq!(flags.bits(), 0);
    flags.insert(PipelineFlags::ALPHA_CLIP);
    flags.insert(PipelineFlags::AUDIOLINK);
    assert!(flags.contains(PipelineFlags::ALPHA_CLIP));
    assert!(flags.contains(PipelineFlags::AUDIOLINK));
    assert!(!flags.contains(PipelineFlags::OUTLINE));

    let combined = PipelineFlags::TRANSPARENT | PipelineFlags::FUR_SHELL;
    assert!(combined.intersects(PipelineFlags::TRANSPARENT));
    assert!((combined & PipelineFlags::FUR_SHELL) == PipelineFlags::FUR_SHELL);

    flags.remove(PipelineFlags::ALPHA_CLIP);
    assert!(!flags.contains(PipelineFlags::ALPHA_CLIP));

    let mat = MaterialDef {
        gltf_material_index: 0,
        name: "Hair".to_owned(),
        shader_family: vrc_schema::ShaderFamily::LilToon,
        render_queue: 3000,
        alpha_mode: AlphaMode::Blend,
        params: MaterialParams {
            audiolink_enable: true,
            ..Default::default()
        },
    };
    let cfg = PhysicsShaderConfig {
        audiolink_enabled: true,
        ..Default::default()
    };
    let derived = PipelineFlags::from_material(&mat, &cfg);
    assert!(derived.contains(PipelineFlags::TRANSPARENT));
    assert!(derived.contains(PipelineFlags::AUDIOLINK));

    let k1 = PipelineKey::new(derived, CullMode::Back);
    let k2 = PipelineKey::new(derived, CullMode::Back);
    let k3 = PipelineKey::new(derived, CullMode::None);
    let mut set = HashSet::new();
    set.insert(k1);
    assert!(set.contains(&k2));
    assert!(!set.contains(&k3));
}

#[test]
fn procedural_fallback_integrity_and_morph_targets() {
    let (model, manifest) = AvatarModel::procedural_fallback();
    manifest
        .validate()
        .expect("procedural fallback manifest must validate");

    assert!(!model.nodes.is_empty());
    assert!(!model.primitives.is_empty());
    assert!(!model.skins.is_empty());

    let skin = &model.skins[0];
    assert_eq!(skin.joints.len(), skin.inverse_bind_matrices.len());
    for &j in &skin.joints {
        assert!(j < model.nodes.len());
    }

    for prim in &model.primitives {
        assert!(!prim.vertices.is_empty());
        assert!(!prim.indices.is_empty());
        for &idx in &prim.indices {
            assert!((idx as usize) < prim.vertices.len());
        }
        for v in &prim.vertices {
            let n = Vec3::from_array(v.normal);
            assert!(n.is_finite() && n.length() > 0.5);
        }
    }

    for (bone_name, path) in &manifest.humanoid_bones {
        assert!(
            model.find_node_index(path).is_some(),
            "missing humanoid bone {bone_name} at {path}"
        );
    }
    for pb in &manifest.phys_bones {
        assert!(
            model.find_node_index(&pb.root_node).is_some(),
            "missing phys_bone root {}",
            pb.root_node
        );
    }

    // Verify Viseme morph targets visibly deform mouth vertices.
    let face_prim = &model.primitives[0];
    assert!(face_prim.morph_targets.iter().any(|t| t.name == "vrc.v_aa"));
    assert!(face_prim.morph_targets.iter().any(|t| t.name == "Blink"));

    let mut deformed = Vec::new();
    let weights: Vec<f32> = face_prim
        .morph_targets
        .iter()
        .map(|t| if t.name == "vrc.v_aa" { 1.0 } else { 0.0 })
        .collect();
    face_prim.evaluate_morph_targets(&weights, &mut deformed);
    assert_eq!(deformed.len(), face_prim.vertices.len());
    let max_delta = face_prim
        .vertices
        .iter()
        .zip(&deformed)
        .map(|(a, b)| Vec3::from_array(a.position).distance(Vec3::from_array(b.position)))
        .fold(0.0_f32, f32::max);
    assert!(
        max_delta > 0.01,
        "vrc.v_aa morph should move mouth vertices"
    );
}

#[test]
fn load_from_manifest_loads_valid_glb_and_errors_when_missing() {
    let err = AvatarModel::load_from_manifest(
        PathBuf::from("Z:/nonexistent_dir_12345/mascot_manifest.json").as_path(),
    )
    .unwrap_err();
    assert!(matches!(
        err,
        RendererError::Schema(_) | RendererError::Io(_)
    ));

    let dir = std::env::temp_dir().join(format!("vrc_renderer_glb_test_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create test temp dir");

    let glb_bytes = build_minimal_triangle_glb();
    let glb_path = dir.join("avatar.glb");
    fs::write(&glb_path, &glb_bytes).expect("write minimal glb");

    let manifest = MascotManifest {
        schema_version: SCHEMA_VERSION.to_owned(),
        avatar_name: "MiniTri".to_owned(),
        glb_file: "avatar.glb".to_owned(),
        ..Default::default()
    };
    let manifest_path = dir.join("mascot_manifest.json");
    manifest.save(&manifest_path).expect("save manifest");

    let (loaded_model, loaded_manifest) =
        AvatarModel::load_from_manifest(&manifest_path).expect("load minimal glb + manifest");
    assert_eq!(loaded_manifest.avatar_name, "MiniTri");
    assert_eq!(loaded_model.primitives.len(), 1);
    assert_eq!(loaded_model.primitives[0].vertices.len(), 3);
    assert_eq!(loaded_model.primitives[0].indices, vec![0, 1, 2]);
    assert_eq!(loaded_model.primitives[0].morph_targets.len(), 2);
    assert_eq!(loaded_model.primitives[0].morph_targets[0].name, "vrc.v_aa");
    assert_eq!(loaded_model.primitives[0].morph_targets[1].name, "Blink");

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn audiolink_fft_separates_bass_and_treble() {
    let sample_rate = 48_000_u32;
    let sine_window = |freq_hz: f32| -> Vec<f32> {
        (0..FFT_SIZE)
            .map(|i| (2.0 * PI * freq_hz * (i as f32 / sample_rate as f32)).sin())
            .collect()
    };

    let mut bass_analyzer = AudioLinkAnalyzer::new(sample_rate);
    let bass_samples = sine_window(90.0);
    let mut bass_out = AudioLinkData::default();
    for _ in 0..4 {
        bass_out = bass_analyzer.analyze_frame(&bass_samples);
    }
    assert!(bass_out.active);
    assert!(
        bass_out.bands[0] > 0.5 && bass_out.bands[0] > bass_out.bands[3] * 10.0,
        "expected Bass dominance for 90Hz, got {:?}",
        bass_out.bands
    );
    assert_eq!(bass_out.history[0], bass_out.bands);

    let mut treble_analyzer = AudioLinkAnalyzer::new(sample_rate);
    let treble_samples = sine_window(8_000.0);
    let mut treble_out = AudioLinkData::default();
    for _ in 0..4 {
        treble_out = treble_analyzer.analyze_frame(&treble_samples);
    }
    assert!(
        treble_out.bands[3] > 0.5 && treble_out.bands[3] > treble_out.bands[0] * 10.0,
        "expected Treble dominance for 8000Hz, got {:?}",
        treble_out.bands
    );

    let capture = AudioLinkCapture::new(false);
    assert!(!capture.is_enabled());
    assert_eq!(capture.latest_data(), AudioLinkData::default());

    // Streaming push_samples with 50% overlap (2048 samples -> multiple hops)
    let mut streamer = AudioLinkAnalyzer::new(sample_rate);
    let stream: Vec<f32> = (0..2048)
        .map(|i| (2.0 * PI * 600.0 * (i as f32 / sample_rate as f32)).sin())
        .collect();
    let streamed = streamer
        .push_samples(&stream[..1500])
        .expect("1500 >= 1024 should emit at least one frame");
    assert!(streamed.bands[1] > streamed.bands[3]);
    let streamed2 = streamer
        .push_samples(&stream[1500..])
        .expect("remaining samples should emit hop frame");
    assert!(streamed2.bands[1] > 0.2);
}

#[test]
fn camera_rig_projection_and_ray_plane_round_trip() {
    let (model, _) = AvatarModel::procedural_fallback();
    let cam = CameraRig::for_bounds(model.bounds_min, model.bounds_max, 400, 600);
    assert!((cam.aspect - (400.0 / 600.0)).abs() < 1e-4);

    let world_pt = Vec3::new(0.12, 0.85, 0.08);
    let screen_pt = cam
        .project_world_point(world_pt, 400, 600)
        .expect("target point should project in front of camera");
    assert!(screen_pt[0] > 0.0 && screen_pt[0] < 400.0);
    assert!(screen_pt[1] > 0.0 && screen_pt[1] < 600.0);

    let recovered = cam.screen_ray_to_world_plane(screen_pt[0], screen_pt[1], 400, 600, world_pt.z);
    assert!(
        recovered.distance(world_pt) < 1e-3,
        "expected round-trip {world_pt:?}, got {recovered:?}"
    );
}

#[test]
fn hit_test_pixel_decoding_matches_contract() {
    let bg = HitTestResult::from_pixel([0, 0, 0, 0]);
    assert!(!bg.is_opaque);
    assert_eq!(bg.node_index, None);
    assert!(!bg.is_head);

    // 12 / 255 = 0.047 <= 0.05 -> click-through
    let faint = HitTestResult::from_pixel([12, 1, 0, 1]);
    assert!(!faint.is_opaque);
    assert_eq!(faint.node_index, None);
    assert!(!faint.is_head);

    // 13 / 255 = 0.05098 > 0.05 -> interactive, encoded_id = 1 -> node_index = 0
    let body = HitTestResult::from_pixel([13, 1, 0, 0]);
    assert!(body.is_opaque);
    assert_eq!(body.node_index, Some(0));
    assert!(!body.is_head);

    // 16-bit encoded node_id = 0x0105 = 261 -> node_index = 260, head flag set
    let head = HitTestResult::from_pixel([255, 0x05, 0x01, 0x01]);
    assert!(head.is_opaque);
    assert_eq!(head.node_index, Some(260));
    assert!(head.is_head);
}

#[test]
fn headless_wgpu_shader_and_renderer_smoke_test() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
        ..Default::default()
    }))
    .or_else(|_| {
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: true,
            ..Default::default()
        }))
    });

    let Ok(adapter) = adapter else {
        // No GPU or software adapter available in this environment.
        return;
    };

    let Ok((device, queue)) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
    else {
        return;
    };

    // 1. Validate all 4 WGSL shaders and pipeline permutations compile cleanly.
    let format = preferred_surface_format();
    let mut cache = PipelineCache::new(&device);
    for flags in [
        PipelineFlags::NONE,
        PipelineFlags::ALPHA_CLIP,
        PipelineFlags::TRANSPARENT,
        PipelineFlags::OUTLINE,
        PipelineFlags::FUR_SHELL,
        PipelineFlags::AUDIOLINK,
    ] {
        let _ = cache.get_or_create(&device, PipelineKey::new(flags, CullMode::Back), format);
    }
    let _ = cache.get_or_create_hit_test(&device, CullMode::Back);
    assert_eq!(cache.cached_pipeline_count(), 6);

    // 2. Render procedural fallback mascot offscreen and verify hit-test readback.
    let (w, h) = (120_u32, 180_u32);
    let mut renderer = MascotRenderer::new(&device, &queue, format, w, h);
    let (model, manifest) = AvatarModel::procedural_fallback();
    let rest_world = model.compute_rest_world_matrices();
    renderer.set_model(&device, &queue, model, &manifest);

    let mut visemes = HashMap::new();
    visemes.insert("vrc.v_aa".to_owned(), 0.75);
    renderer.update_pose(&queue, &rest_world, &visemes, &HashMap::new());
    renderer.update_audiolink(
        &queue,
        &AudioLinkData {
            bands: [0.8, 0.4, 0.2, 0.1],
            active: true,
            ..Default::default()
        },
    );

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test_target"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());

    let cfg = PhysicsShaderConfig {
        outline_enabled: true,
        fur_enabled: true,
        audiolink_enabled: true,
        quality_preset: QualityPreset::High,
        ..Default::default()
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    renderer.render(&device, &queue, &mut encoder, &target_view, &cfg);
    queue.submit(Some(encoder.finish()));

    for _ in 0..50 {
        renderer.poll_hit_test_readback(&device);
        if renderer.hit_test(w / 2, h / 2).is_opaque {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }

    let corner = renderer.hit_test(0, 0);
    assert!(
        !corner.is_opaque,
        "top-left background pixel should be transparent/click-through"
    );

    let center = renderer.hit_test(w / 2, h / 2);
    assert!(
        center.is_opaque,
        "center of rendered mascot should be opaque/interactive"
    );

    let head_px = renderer
        .project_node_screen("Head", &rest_world)
        .expect("Head bone should project onto viewport");
    let head_hit = renderer.hit_test(
        (head_px[0] as u32).min(w - 1),
        (head_px[1] as u32).saturating_sub(8).min(h - 1),
    );
    assert!(
        head_hit.is_opaque && head_hit.is_head,
        "projected head pixel should report is_head"
    );

    // 3. Test resize + node visibility toggle (hiding Body makes center transparent).
    renderer.resize(&device, 160, 240);
    let ray_pt = renderer.screen_ray_to_world_plane(80.0, 120.0, 0.0);
    assert!(ray_pt.is_finite());

    let mut hidden = HashMap::new();
    hidden.insert("Body".to_owned(), false);
    renderer.update_pose(&queue, &rest_world, &visemes, &hidden);

    let target2 = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test_target_2"),
        size: wgpu::Extent3d {
            width: 160,
            height: 240,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let target2_view = target2.create_view(&wgpu::TextureViewDescriptor::default());
    let mut enc2 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    renderer.render(&device, &queue, &mut enc2, &target2_view, &cfg);
    queue.submit(Some(enc2.finish()));
    for _ in 0..30 {
        renderer.poll_hit_test_readback(&device);
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(!renderer.hit_test(80, 120).is_opaque);
}

fn build_minimal_triangle_glb() -> Vec<u8> {
    let mut bin: Vec<u8> = Vec::new();
    let positions: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    for p in &positions {
        for f in p {
            bin.extend_from_slice(&f.to_le_bytes());
        }
    }
    let indices: [u16; 3] = [0, 1, 2];
    for idx in &indices {
        bin.extend_from_slice(&idx.to_le_bytes());
    }
    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }

    // Two morph target POSITION delta buffers (3 vertices * 3 f32 = 36 bytes each)
    let morph0_offset = bin.len();
    let morph0: [[f32; 3]; 3] = [[0.0, -0.1, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, 0.0]];
    for d in &morph0 {
        for f in d {
            bin.extend_from_slice(&f.to_le_bytes());
        }
    }
    let morph1_offset = bin.len();
    let morph1: [[f32; 3]; 3] = [[0.0, 0.05, 0.0], [0.0, 0.05, 0.0], [0.0, 0.05, 0.0]];
    for d in &morph1 {
        for f in d {
            bin.extend_from_slice(&f.to_le_bytes());
        }
    }

    let json = serde_json::json!({
        "asset": { "version": "2.0" },
        "scene": 0,
        "scenes": [{ "nodes": [0] }],
        "nodes": [{ "name": "TriNode", "mesh": 0 }],
        "meshes": [{
            "extras": { "targetNames": ["vrc.v_aa", "Blink"] },
            "primitives": [{
                "attributes": { "POSITION": 0 },
                "indices": 1,
                "targets": [
                    { "POSITION": 2 },
                    { "POSITION": 3 }
                ]
            }]
        }],
        "buffers": [{ "byteLength": bin.len() }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962 },
            { "buffer": 0, "byteOffset": 36, "byteLength": 6, "target": 34963 },
            { "buffer": 0, "byteOffset": morph0_offset, "byteLength": 36, "target": 34962 },
            { "buffer": 0, "byteOffset": morph1_offset, "byteLength": 36, "target": 34962 }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [0.0, 0.0, 0.0],
                "max": [1.0, 1.0, 0.0]
            },
            {
                "bufferView": 1,
                "componentType": 5123,
                "count": 3,
                "type": "SCALAR"
            },
            {
                "bufferView": 2,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [0.0, -0.1, 0.0],
                "max": [0.0, 0.0, 0.0]
            },
            {
                "bufferView": 3,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [0.0, 0.05, 0.0],
                "max": [0.0, 0.05, 0.0]
            }
        ]
    });
    let mut json_bytes = serde_json::to_vec(&json).expect("serialize gltf json");
    while !json_bytes.len().is_multiple_of(4) {
        json_bytes.push(b' ');
    }

    let total_len = 12 + 8 + json_bytes.len() + 8 + bin.len();
    let mut glb = Vec::with_capacity(total_len);
    glb.extend_from_slice(b"glTF");
    glb.extend_from_slice(&2_u32.to_le_bytes());
    glb.extend_from_slice(&(total_len as u32).to_le_bytes());

    glb.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    glb.extend_from_slice(&0x4E4F_534A_u32.to_le_bytes()); // JSON
    glb.extend_from_slice(&json_bytes);

    glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    glb.extend_from_slice(&0x004E_4942_u32.to_le_bytes()); // BIN
    glb.extend_from_slice(&bin);
    glb
}
