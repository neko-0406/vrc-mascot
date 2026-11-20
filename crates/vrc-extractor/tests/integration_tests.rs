use std::fs;
use std::path::Path;

use image::{ImageBuffer, Rgba};
use tempfile::tempdir;
use vrc_extractor::{discover_avatars, discover_scenes, extract_avatar, ExtractOptions};
use vrc_schema::{
    AlphaMode, CurveProperty, ExpressionMenuItemType, ExpressionParamType, FxConditionMode,
    MascotManifest, PhysBoneLimitType, ShaderFamily, VisemeKey,
};

fn write_file(path: &Path, content: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent dirs");
    }
    fs::write(path, content).expect("write test file");
}

fn write_meta(asset_path: &Path, guid: &str) {
    let meta_path = Path::new(&format!("{}.meta", asset_path.display())).to_path_buf();
    write_file(&meta_path, &format!("fileFormatVersion: 2\nguid: {guid}\n"));
}

fn write_png(path: &Path, width: u32, height: u32) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent dirs");
    }
    let img: ImageBuffer<Rgba<u8>, Vec<u8>> =
        ImageBuffer::from_pixel(width, height, Rgba([220, 120, 180, 255]));
    img.save(path).expect("save test png");
}

/// Minimal valid ASCII FBX 7.4 with 2-bone skeleton (Hips -> Head), a quad mesh skinned to Hips/Head,
/// UVs, Normals, Vertex Colors, and 2 BlendShapes (`vrc.v_aa`, `eye_smile`).
fn sample_ascii_fbx() -> &'static str {
    r#"; FBX 7.4.0 project file
FBXHeaderExtension:  {
    FBXHeaderVersion: 1003
    FBXVersion: 7400
}
GlobalSettings:  {
    Version: 1000
    Properties70:  {
        P: "UpAxis", "int", "Integer", "",1
        P: "UpAxisSign", "int", "Integer", "",1
        P: "FrontAxis", "int", "Integer", "",2
        P: "FrontAxisSign", "int", "Integer", "",1
        P: "CoordAxis", "int", "Integer", "",0
        P: "CoordAxisSign", "int", "Integer", "",1
        P: "UnitScaleFactor", "double", "Number", "",100
    }
}
Documents:  {
    Count: 1
    Document: 100, "", "Scene" {
        RootNode: 0
    }
}
Definitions:  {
    Version: 100
    Count: 6
    ObjectType: "Model" { Count: 3 }
    ObjectType: "Geometry" { Count: 3 }
    ObjectType: "Material" { Count: 1 }
    ObjectType: "Deformer" { Count: 6 }
    ObjectType: "NodeAttribute" { Count: 2 }
}
Objects:  {
    NodeAttribute: 1001, "NodeAttribute::Hips", "LimbNode" {
        TypeFlags: "Skeleton"
    }
    NodeAttribute: 1002, "NodeAttribute::Head", "LimbNode" {
        TypeFlags: "Skeleton"
    }
    Model: 2001, "Model::Hips", "LimbNode" {
        Version: 232
        Properties70:  {
            P: "Lcl Translation", "Lcl Translation", "", "A",0,100,0
        }
    }
    Model: 2002, "Model::Head", "LimbNode" {
        Version: 232
        Properties70:  {
            P: "Lcl Translation", "Lcl Translation", "", "A",0,50,0
        }
    }
    Model: 2003, "Model::Body", "Mesh" {
        Version: 232
    }
    Geometry: 3001, "Geometry::Body", "Mesh" {
        Vertices: *12 {
            a: -50,0,0, 50,0,0, 50,150,0, -50,150,0
        }
        PolygonVertexIndex: *4 {
            a: 0,1,2,-4
        }
        LayerElementNormal: 0 {
            Version: 102
            MappingInformationType: "ByPolygonVertex"
            ReferenceInformationType: "Direct"
            Normals: *12 {
                a: 0,0,1, 0,0,1, 0,0,1, 0,0,1
            }
        }
        LayerElementUV: 0 {
            Version: 101
            MappingInformationType: "ByPolygonVertex"
            ReferenceInformationType: "IndexToDirect"
            UV: *8 {
                a: 0,0, 1,0, 1,1, 0,1
            }
            UVIndex: *4 {
                a: 0,1,2,3
            }
        }
        LayerElementColor: 0 {
            Version: 101
            MappingInformationType: "ByPolygonVertex"
            ReferenceInformationType: "Direct"
            Colors: *16 {
                a: 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1
            }
        }
        LayerElementMaterial: 0 {
            Version: 101
            MappingInformationType: "AllSame"
            ReferenceInformationType: "IndexToDirect"
            Materials: *1 {
                a: 0
            }
        }
        Layer: 0 {
            Version: 100
            LayerElement:  {
                Type: "LayerElementNormal"
                TypedIndex: 0
            }
            LayerElement:  {
                Type: "LayerElementUV"
                TypedIndex: 0
            }
            LayerElement:  {
                Type: "LayerElementColor"
                TypedIndex: 0
            }
            LayerElement:  {
                Type: "LayerElementMaterial"
                TypedIndex: 0
            }
        }
    }
    Material: 4001, "Material::BodyMat", "" {
        Version: 102
    }
    Deformer: 5001, "Deformer::Skin", "Skin" {
        Version: 101
    }
    Deformer: 5002, "SubDeformer::HipsCluster", "Cluster" {
        Version: 100
        Indexes: *2 {
            a: 0,1
        }
        Weights: *2 {
            a: 1.0,1.0
        }
        Transform: *16 {
            a: 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,-100,0,1
        }
        TransformLink: *16 {
            a: 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,100,0,1
        }
    }
    Deformer: 5003, "SubDeformer::HeadCluster", "Cluster" {
        Version: 100
        Indexes: *2 {
            a: 2,3
        }
        Weights: *2 {
            a: 1.0,1.0
        }
        Transform: *16 {
            a: 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,-150,0,1
        }
        TransformLink: *16 {
            a: 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,150,0,1
        }
    }
    Deformer: 6001, "Deformer::Blend", "BlendShape" {
        Version: 100
    }
    Deformer: 6002, "SubDeformer::vrc.v_aa", "BlendShapeChannel" {
        Version: 100
        DeformPercent: 0
        FullWeights: *1 {
            a: 100
        }
    }
    Geometry: 6003, "Geometry::vrc.v_aa", "Shape" {
        Version: 100
        Indexes: *1 {
            a: 2
        }
        Vertices: *3 {
            a: 0,-5,0
        }
        Normals: *3 {
            a: 0,0,0
        }
    }
    Deformer: 6004, "SubDeformer::eye_smile", "BlendShapeChannel" {
        Version: 100
        DeformPercent: 0
        FullWeights: *1 {
            a: 100
        }
    }
    Geometry: 6005, "Geometry::eye_smile", "Shape" {
        Version: 100
        Indexes: *1 {
            a: 3
        }
        Vertices: *3 {
            a: 0,3,0
        }
        Normals: *3 {
            a: 0,0,0
        }
    }
}
Connections:  {
    C: "OO",2001,0
    C: "OO",2002,2001
    C: "OO",2003,0
    C: "OO",1001,2001
    C: "OO",1002,2002
    C: "OO",3001,2003
    C: "OO",4001,2003
    C: "OO",5001,3001
    C: "OO",5002,5001
    C: "OO",5003,5001
    C: "OO",2001,5002
    C: "OO",2002,5003
    C: "OO",6001,3001
    C: "OO",6002,6001
    C: "OO",6003,6002
    C: "OO",6004,6001
    C: "OO",6005,6004
}
"#
}

#[test]
fn end_to_end_prefab_variant_ma_shaders_fx_fbx_and_glb_validation() {
    let tmp = tempdir().expect("create tempdir");
    let proj = tmp.path().join("UnityProject");

    // 1. Dynamic script & shader .meta files
    let ma_dir = proj.join("Packages/nadena.dev.modular-avatar/Runtime");
    let ma_merge_arm_guid = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1";
    let ma_menu_inst_guid = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa2";
    let ma_menu_item_guid = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa3";
    let ma_params_guid = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa4";
    let ma_mesh_set_guid = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa5";
    let ma_bone_proxy_guid = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa6";
    let ma_merge_anim_guid = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa7";

    for (fname, guid) in [
        ("ModularAvatarMergeArmature.cs", ma_merge_arm_guid),
        ("ModularAvatarMenuInstaller.cs", ma_menu_inst_guid),
        ("ModularAvatarMenuItem.cs", ma_menu_item_guid),
        ("ModularAvatarParameters.cs", ma_params_guid),
        ("ModularAvatarMeshSettings.cs", ma_mesh_set_guid),
        ("ModularAvatarBoneProxy.cs", ma_bone_proxy_guid),
        ("ModularAvatarMergeAnimator.cs", ma_merge_anim_guid),
    ] {
        let p = ma_dir.join(fname);
        write_file(&p, "// stub");
        write_meta(&p, guid);
    }

    let vrc_dir = proj.join("Packages/com.vrchat.avatars/Runtime");
    let vrc_desc_guid = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb1";
    let vrc_pb_guid = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb2";
    let vrc_col_guid = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb3";
    for (fname, guid) in [
        ("VRCAvatarDescriptor.cs", vrc_desc_guid),
        ("VRCPhysBone.cs", vrc_pb_guid),
        ("VRCPhysBoneCollider.cs", vrc_col_guid),
    ] {
        let p = vrc_dir.join(fname);
        write_file(&p, "// stub");
        write_meta(&p, guid);
    }

    // Dynamic shader discovery
    let liltoon_shader_guid = "ccccccccccccccccccccccccccccccc1";
    let liltoon_shader_path = proj.join("Assets/Shaders/lilToon.shader");
    write_file(&liltoon_shader_path, "Shader \"lilToon\" { Properties {} }");
    write_meta(&liltoon_shader_path, liltoon_shader_guid);

    // 2. Texture (64x64 PNG -> resized to max_texture_size=32)
    let tex_guid = "ddddddddddddddddddddddddddddddd1";
    let tex_path = proj.join("Assets/Textures/Body.png");
    write_png(&tex_path, 64, 64);
    write_meta(&tex_path, tex_guid);

    // 3. FBX + .fbx.meta with HumanDescription
    let fbx_guid = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeee1";
    let fbx_path = proj.join("Assets/Models/SampleAvatar.fbx");
    write_file(&fbx_path, sample_ascii_fbx());
    write_file(
        Path::new(&format!("{}.meta", fbx_path.display())),
        &format!(
            "fileFormatVersion: 2\nguid: {fbx_guid}\nModelImporter:\n  humanDescription:\n    human:\n    - boneName: Hips\n      humanName: Hips\n    - boneName: Head\n      humanName: Head\n"
        ),
    );

    // 4. Material (lilToon)
    let mat_guid = "fffffffffffffffffffffffffffffff1";
    let mat_path = proj.join("Assets/Materials/Body.mat");
    write_file(
        &mat_path,
        &format!(
            r#"%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!21 &2100000
Material:
  m_Name: Body_Mat
  m_Shader: {{fileID: 4800000, guid: {liltoon_shader_guid}, type: 3}}
  m_CustomRenderQueue: 2000
  m_SavedProperties:
    m_TexEnvs:
    - _MainTex:
        m_Texture: {{fileID: 2800000, guid: {tex_guid}, type: 3}}
        m_Scale: {{x: 1, y: 1}}
        m_Offset: {{x: 0, y: 0}}
    m_Floats:
    - _ShadowBorder: 0.45
    - _ShadowBlur: 0.12
    - _UseRim: 1
    - _UseOutline: 1
    - _OutlineWidth: 0.003
    - _UseFur: 1
    - _FurLayerNum: 6
    - _UseAudioLink: 1
    m_Colors:
    - _Color: {{r: 1, g: 0.95, b: 0.95, a: 1}}
    - _ShadowColor: {{r: 0.8, g: 0.75, b: 0.85, a: 1}}
"#
        ),
    );
    write_meta(&mat_path, mat_guid);

    // 5. Expression Parameters & Menu
    let params_guid = "11111111111111111111111111111111";
    let params_path = proj.join("Assets/Expressions/Params.asset");
    write_file(
        &params_path,
        r#"%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!114 &11400000
MonoBehaviour:
  m_Name: Params
  parameters:
  - name: FaceEmotion
    valueType: 0
    saved: 1
    defaultValue: 0
"#,
    );
    write_meta(&params_path, params_guid);

    let menu_guid = "22222222222222222222222222222222";
    let menu_path = proj.join("Assets/Expressions/Menu.asset");
    write_file(
        &menu_path,
        r#"%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!114 &11400000
MonoBehaviour:
  m_Name: Menu
  controls:
  - name: Smile
    type: 102
    parameter:
      name: FaceEmotion
    value: 1
"#,
    );
    write_meta(&menu_path, menu_guid);

    // 6. Animation clips & FX Controllers (base + MA MergeAnimator)
    let anim_smile_guid = "33333333333333333333333333333331";
    let anim_smile_path = proj.join("Assets/Animations/Smile.anim");
    write_file(
        &anim_smile_path,
        r#"%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!74 &7400000
AnimationClip:
  m_Name: Smile
  m_FloatCurves:
  - curve:
      m_Curve:
      - time: 0
        value: 0
      - time: 0.2
        value: 100
    attribute: blendShape.eye_smile
    path: Body
  - curve:
      m_Curve:
      - time: 0
        value: 1
    attribute: m_IsActive
    path: Outfit/Ribbon_L
"#,
    );
    write_meta(&anim_smile_path, anim_smile_guid);

    let fx_guid = "44444444444444444444444444444441";
    let fx_path = proj.join("Assets/Controllers/FX.controller");
    write_file(
        &fx_path,
        &format!(
            r#"%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!91 &9100000
AnimatorController:
  m_Name: FX
  m_AnimatorParameters:
  - m_Name: FaceEmotion
    m_Type: 3
    m_DefaultInt: 0
  m_AnimatorLayers:
  - m_Name: FaceLayer
    m_StateMachine: {{fileID: 11070001}}
    m_DefaultWeight: 1
--- !u!1107 &11070001
AnimatorStateMachine:
  m_Name: FaceLayer
  m_States:
  - state: {{fileID: 11020001}}
  - state: {{fileID: 11020002}}
  m_DefaultState: {{fileID: 11020001}}
  m_AnyStateTransitions:
  - {{fileID: 11010001}}
--- !u!1102 &11020001
AnimatorState:
  m_Name: Neutral
  m_Transitions: []
--- !u!1102 &11020002
AnimatorState:
  m_Name: Smile
  m_Motion: {{fileID: 7400000, guid: {anim_smile_guid}, type: 2}}
  m_Transitions: []
--- !u!1101 &11010001
AnimatorStateTransition:
  m_DstState: {{fileID: 11020002}}
  m_HasExitTime: 0
  m_ExitTime: 0.75
  m_TransitionDuration: 0.1
  m_Conditions:
  - m_ConditionMode: 6
    m_ConditionEvent: FaceEmotion
    m_EventTreshold: 1
"#
        ),
    );
    write_meta(&fx_path, fx_guid);

    // Extra MA controller merged via ModularAvatarMergeAnimator
    let ma_fx_guid = "44444444444444444444444444444442";
    let ma_fx_path = proj.join("Assets/Controllers/OutfitFX.controller");
    write_file(
        &ma_fx_path,
        r#"%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!91 &9100000
AnimatorController:
  m_Name: OutfitFX
  m_AnimatorParameters:
  - m_Name: RibbonGlow
    m_Type: 1
    m_DefaultFloat: 0.5
  m_AnimatorLayers:
  - m_Name: RibbonLayer
    m_StateMachine: {fileID: 11070010}
    m_DefaultWeight: 1
--- !u!1107 &11070010
AnimatorStateMachine:
  m_Name: RibbonLayer
  m_States:
  - state: {fileID: 11020010}
  m_DefaultState: {fileID: 11020010}
--- !u!1102 &11020010
AnimatorState:
  m_Name: Idle
  m_Transitions: []
"#,
    );
    write_meta(&ma_fx_path, ma_fx_guid);

    // 7. Base Avatar Prefab -> Prefab Variant (modifies position & material, adds component, removes component)
    let base_prefab_guid = "55555555555555555555555555555551";
    let base_prefab_path = proj.join("Assets/Prefabs/BaseAvatar.prefab");
    write_file(
        &base_prefab_path,
        &format!(
            r#"%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!1 &100
GameObject:
  m_Name: BaseAvatar
  m_IsActive: 1
  m_Component:
  - component: {{fileID: 101}}
  - component: {{fileID: 102}}
  - component: {{fileID: 103}}
--- !u!4 &101
Transform:
  m_GameObject: {{fileID: 100}}
  m_LocalPosition: {{x: 0, y: 0, z: 0}}
  m_LocalRotation: {{x: 0, y: 0, z: 0, w: 1}}
  m_LocalScale: {{x: 1, y: 1, z: 1}}
  m_Children:
  - {{fileID: 201}}
  - {{fileID: 301}}
  m_Father: {{fileID: 0}}
--- !u!114 &102
MonoBehaviour:
  m_GameObject: {{fileID: 100}}
  m_Script: {{fileID: 11500000, guid: {vrc_desc_guid}, type: 3}}
  VisemeSkinnedMesh: {{fileID: 302}}
  VisemeBlendShapes:
  - ""
  - ""
  - ""
  - ""
  - ""
  - ""
  - ""
  - ""
  - ""
  - ""
  - vrc.v_aa
  - ""
  - ""
  - ""
  - ""
  expressionsMenu: {{fileID: 11400000, guid: {menu_guid}, type: 3}}
  expressionParameters: {{fileID: 11400000, guid: {params_guid}, type: 3}}
  baseAnimationLayers:
  - type: 5
    animatorController: {{fileID: 9100000, guid: {fx_guid}, type: 3}}
--- !u!114 &103
MonoBehaviour:
  m_GameObject: {{fileID: 100}}
  m_Script: {{fileID: 11500000, guid: 99999999999999999999999999999999, type: 3}}
  dummy: 1
--- !u!1 &200
GameObject:
  m_Name: Armature
  m_IsActive: 1
  m_Component:
  - component: {{fileID: 201}}
--- !u!4 &201
Transform:
  m_GameObject: {{fileID: 200}}
  m_LocalPosition: {{x: 0, y: 0, z: 0}}
  m_LocalRotation: {{x: 0, y: 0, z: 0, w: 1}}
  m_LocalScale: {{x: 1, y: 1, z: 1}}
  m_Children:
  - {{fileID: 211}}
  m_Father: {{fileID: 101}}
--- !u!1 &210
GameObject:
  m_Name: Hips
  m_IsActive: 1
  m_Component:
  - component: {{fileID: 211}}
--- !u!4 &211
Transform:
  m_GameObject: {{fileID: 210}}
  m_LocalPosition: {{x: 0, y: 1, z: 0}}
  m_LocalRotation: {{x: 0, y: 0, z: 0, w: 1}}
  m_LocalScale: {{x: 1, y: 1, z: 1}}
  m_Children:
  - {{fileID: 221}}
  m_Father: {{fileID: 201}}
--- !u!1 &220
GameObject:
  m_Name: Head
  m_IsActive: 1
  m_Component:
  - component: {{fileID: 221}}
  - component: {{fileID: 222}}
--- !u!4 &221
Transform:
  m_GameObject: {{fileID: 220}}
  m_LocalPosition: {{x: 0, y: 0.5, z: 0}}
  m_LocalRotation: {{x: 0, y: 0, z: 0, w: 1}}
  m_LocalScale: {{x: 1, y: 1, z: 1}}
  m_Children: []
  m_Father: {{fileID: 211}}
--- !u!114 &222
MonoBehaviour:
  m_GameObject: {{fileID: 220}}
  m_Script: {{fileID: 11500000, guid: {vrc_col_guid}, type: 3}}
  shapeType: 1
  radius: 0.1
  height: 0.25
  position: {{x: 0, y: 0.08, z: 0}}
  rotation: {{x: 0, y: 0, z: 0, w: 1}}
  insideBounds: 0
--- !u!1 &300
GameObject:
  m_Name: Body
  m_IsActive: 1
  m_Component:
  - component: {{fileID: 301}}
  - component: {{fileID: 302}}
--- !u!4 &301
Transform:
  m_GameObject: {{fileID: 300}}
  m_LocalPosition: {{x: 0, y: 0, z: 0}}
  m_LocalRotation: {{x: 0, y: 0, z: 0, w: 1}}
  m_LocalScale: {{x: 1, y: 1, z: 1}}
  m_Children: []
  m_Father: {{fileID: 101}}
--- !u!137 &302
SkinnedMeshRenderer:
  m_GameObject: {{fileID: 300}}
  m_Mesh: {{fileID: 4300000, guid: {fbx_guid}, type: 3}}
  m_RootBone: {{fileID: 211}}
  m_Bones:
  - {{fileID: 211}}
  - {{fileID: 221}}
  m_Materials:
  - {{fileID: 2100000, guid: {mat_guid}, type: 3}}
"#
        ),
    );
    write_meta(&base_prefab_path, base_prefab_guid);

    // Prefab Variant overriding BaseAvatar name, Head position, removing &103, adding ModularAvatarMeshSettings
    let variant_guid = "55555555555555555555555555555552";
    let variant_path = proj.join("Assets/Prefabs/SampleAvatar.prefab");
    write_file(
        &variant_path,
        &format!(
            r#"%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!1001 &1001
PrefabInstance:
  m_Modification:
    m_TransformParent: {{fileID: 0}}
    m_Modifications:
    - target: {{fileID: 100, guid: {base_prefab_guid}, type: 3}}
      propertyPath: m_Name
      value: SampleAvatar
      objectReference: {{fileID: 0}}
    - target: {{fileID: 221, guid: {base_prefab_guid}, type: 3}}
      propertyPath: m_LocalPosition.y
      value: 0.55
      objectReference: {{fileID: 0}}
    m_RemovedComponents:
    - {{fileID: 103, guid: {base_prefab_guid}, type: 3}}
    m_AddedComponents:
    - targetCorrespondingSourceObject: {{fileID: 100, guid: {base_prefab_guid}, type: 3}}
      addedObject: {{fileID: 8001}}
  m_SourcePrefab: {{fileID: 100100000, guid: {base_prefab_guid}, type: 3}}
--- !u!1 &100 stripped
GameObject:
  m_CorrespondingSourceObject: {{fileID: 100, guid: {base_prefab_guid}, type: 3}}
  m_PrefabInstance: {{fileID: 1001}}
--- !u!4 &101 stripped
Transform:
  m_CorrespondingSourceObject: {{fileID: 101, guid: {base_prefab_guid}, type: 3}}
  m_PrefabInstance: {{fileID: 1001}}
--- !u!4 &211 stripped
Transform:
  m_CorrespondingSourceObject: {{fileID: 211, guid: {base_prefab_guid}, type: 3}}
  m_PrefabInstance: {{fileID: 1001}}
--- !u!114 &8001
MonoBehaviour:
  m_GameObject: {{fileID: 100}}
  m_Script: {{fileID: 11500000, guid: {ma_mesh_set_guid}, type: 3}}
  InheritProbeAnchor: 1
  ProbeAnchor:
    targetObject: {{fileID: 211}}
"#
        ),
    );
    write_meta(&variant_path, variant_guid);

    // 8. Scene placing SampleAvatar Prefab Variant + Modular Avatar Outfit child + BoneProxy + PhysBone
    let scene_rel = Path::new("Assets/SampleScene.unity");
    let scene_path = proj.join(scene_rel);
    write_file(
        &scene_path,
        &format!(
            r#"%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!1001 &9000
PrefabInstance:
  m_Modification:
    m_TransformParent: {{fileID: 0}}
    m_Modifications: []
  m_SourcePrefab: {{fileID: 100100000, guid: {variant_guid}, type: 3}}
--- !u!4 &9001 stripped
Transform:
  m_CorrespondingSourceObject: {{fileID: 101, guid: {base_prefab_guid}, type: 3}}
  m_PrefabInstance: {{fileID: 9000}}
--- !u!4 &9002 stripped
Transform:
  m_CorrespondingSourceObject: {{fileID: 201, guid: {base_prefab_guid}, type: 3}}
  m_PrefabInstance: {{fileID: 9000}}
--- !u!114 &9003 stripped
MonoBehaviour:
  m_CorrespondingSourceObject: {{fileID: 222, guid: {base_prefab_guid}, type: 3}}
  m_PrefabInstance: {{fileID: 9000}}
--- !u!1 &9100
GameObject:
  m_Name: Outfit
  m_IsActive: 1
  m_Component:
  - component: {{fileID: 9101}}
  - component: {{fileID: 9102}}
  - component: {{fileID: 9103}}
  - component: {{fileID: 9104}}
  - component: {{fileID: 9105}}
  - component: {{fileID: 9106}}
--- !u!4 &9101
Transform:
  m_GameObject: {{fileID: 9100}}
  m_LocalPosition: {{x: 0, y: 0, z: 0}}
  m_LocalRotation: {{x: 0, y: 0, z: 0, w: 1}}
  m_LocalScale: {{x: 1, y: 1, z: 1}}
  m_Children:
  - {{fileID: 9111}}
  m_Father: {{fileID: 9001}}
--- !u!114 &9102
MonoBehaviour:
  m_GameObject: {{fileID: 9100}}
  m_Script: {{fileID: 11500000, guid: {ma_merge_arm_guid}, type: 3}}
  mergeTarget:
    targetObject: {{fileID: 9002}}
  prefix: "Outfit_"
  suffix: ""
--- !u!114 &9103
MonoBehaviour:
  m_GameObject: {{fileID: 9100}}
  m_Script: {{fileID: 11500000, guid: {ma_params_guid}, type: 3}}
  parameters:
  - nameOrPrefix: RibbonGlow
    syncType: 2
    defaultValue: 0.75
    saved: 1
    internalParameter: 1
  - nameOrPrefix: FaceEmotion
    syncType: 1
    defaultValue: 0
    saved: 1
--- !u!114 &9104
MonoBehaviour:
  m_GameObject: {{fileID: 9100}}
  m_Script: {{fileID: 11500000, guid: {ma_menu_inst_guid}, type: 3}}
--- !u!114 &9105
MonoBehaviour:
  m_GameObject: {{fileID: 9100}}
  m_Script: {{fileID: 11500000, guid: {ma_menu_item_guid}, type: 3}}
  PortableControl:
    name: Ribbon Toggle
    type: 102
    parameter:
      name: RibbonGlow
    value: 1
--- !u!114 &9106
MonoBehaviour:
  m_GameObject: {{fileID: 9100}}
  m_Script: {{fileID: 11500000, guid: {ma_merge_anim_guid}, type: 3}}
  animator: {{fileID: 9100000, guid: {ma_fx_guid}, type: 3}}
  layerType: 5
--- !u!1 &9110
GameObject:
  m_Name: Outfit_Hips
  m_IsActive: 1
  m_Component:
  - component: {{fileID: 9111}}
--- !u!4 &9111
Transform:
  m_GameObject: {{fileID: 9110}}
  m_LocalPosition: {{x: 0, y: 1, z: 0}}
  m_LocalRotation: {{x: 0, y: 0, z: 0, w: 1}}
  m_LocalScale: {{x: 1, y: 1, z: 1}}
  m_Children:
  - {{fileID: 9121}}
  m_Father: {{fileID: 9101}}
--- !u!1 &9120
GameObject:
  m_Name: Outfit_Head
  m_IsActive: 1
  m_Component:
  - component: {{fileID: 9121}}
--- !u!4 &9121
Transform:
  m_GameObject: {{fileID: 9120}}
  m_LocalPosition: {{x: 0, y: 0.55, z: 0}}
  m_LocalRotation: {{x: 0, y: 0, z: 0, w: 1}}
  m_LocalScale: {{x: 1, y: 1, z: 1}}
  m_Children:
  - {{fileID: 9131}}
  m_Father: {{fileID: 9111}}
--- !u!1 &9130
GameObject:
  m_Name: Ribbon_L
  m_IsActive: 1
  m_Component:
  - component: {{fileID: 9131}}
  - component: {{fileID: 9132}}
--- !u!4 &9131
Transform:
  m_GameObject: {{fileID: 9130}}
  m_LocalPosition: {{x: -0.1, y: 0.1, z: 0}}
  m_LocalRotation: {{x: 0, y: 0, z: 0, w: 1}}
  m_LocalScale: {{x: 1, y: 1, z: 1}}
  m_Children: []
  m_Father: {{fileID: 9121}}
--- !u!114 &9132
MonoBehaviour:
  m_GameObject: {{fileID: 9130}}
  m_Script: {{fileID: 11500000, guid: {vrc_pb_guid}, type: 3}}
  pull: 0.3
  spring: 0.4
  stiffness: 0.2
  gravity: 0.05
  radius: 0.03
  limitType: 1
  maxAngleX: 45
  colliders:
  - {{fileID: 9003}}
--- !u!1 &9200
GameObject:
  m_Name: HatProp
  m_IsActive: 1
  m_Component:
  - component: {{fileID: 9201}}
  - component: {{fileID: 9202}}
--- !u!4 &9201
Transform:
  m_GameObject: {{fileID: 9200}}
  m_LocalPosition: {{x: 1, y: 2, z: 3}}
  m_LocalRotation: {{x: 0, y: 0, z: 0, w: 1}}
  m_LocalScale: {{x: 1, y: 1, z: 1}}
  m_Children: []
  m_Father: {{fileID: 9001}}
--- !u!114 &9202
MonoBehaviour:
  m_GameObject: {{fileID: 9200}}
  m_Script: {{fileID: 11500000, guid: {ma_bone_proxy_guid}, type: 3}}
  boneReference: Head
  attachmentMode: 1
"#
        ),
    );

    // Verify discovery helpers
    let scenes = discover_scenes(&proj).expect("discover scenes");
    assert_eq!(scenes, vec![scene_rel.to_path_buf()]);

    let avatars = discover_avatars(&proj, scene_rel).expect("discover avatars");
    assert_eq!(avatars.len(), 1, "discovered avatars: {avatars:?}");
    assert_eq!(avatars[0].name, "SampleAvatar");
    assert!(avatars[0].has_descriptor);

    // Run full extraction
    let out_dir = tmp.path().join("out");
    let opts = ExtractOptions {
        project_dir: proj.clone(),
        scene_path: scene_rel.to_path_buf(),
        avatar_name: "SampleAvatar".to_owned(),
        output_dir: out_dir.clone(),
        modular_avatar_enabled: true,
        max_texture_size: 32,
    };

    let mut progress_log = Vec::new();
    let report = extract_avatar(&opts, |p| progress_log.push(p)).expect("extract avatar");
    assert!(!progress_log.is_empty());
    assert!((progress_log.last().unwrap().progress - 1.0).abs() < 1e-5);
    assert!(report.glb_path.is_file());
    assert!(report.manifest_path.is_file());

    // Verify avatar.glb with gltf crate (including extras.targetNames in GLB JSON chunk)
    let glb_bytes = fs::read(&report.glb_path).expect("read glb");
    let glb = gltf::binary::Glb::from_slice(&glb_bytes).expect("parse GLB container");
    let json_val: serde_json::Value = serde_json::from_slice(&glb.json).expect("parse GLB JSON");
    assert_eq!(
        json_val["meshes"][0]["extras"]["targetNames"],
        serde_json::json!(["vrc.v_aa", "eye_smile"])
    );

    let (document, buffers, images) =
        gltf::import_slice(&glb_bytes).expect("gltf::import_slice should parse valid GLB 2.0");

    assert!(!buffers.is_empty());
    assert_eq!(images.len(), 1);
    assert!(images[0].width <= 32 && images[0].height <= 32);

    // Verify nodes include transplanted Ribbon_L under Armature/Hips/Head and BoneProxy HatProp under Head!
    let mut head_children = Vec::new();
    for node in document.nodes() {
        if node.name() == Some("Head") {
            head_children = node
                .children()
                .filter_map(|c| c.name().map(str::to_owned))
                .collect();
        }
    }
    assert!(
        head_children.contains(&"Ribbon_L".to_owned()),
        "Ribbon_L should be transplanted under Head: {:?}",
        head_children
    );
    assert!(
        head_children.contains(&"HatProp".to_owned()),
        "HatProp should be reparented under Head via BoneProxy: {:?}",
        head_children
    );

    // Verify mesh attributes, skinning, morph targets
    let mesh = document.meshes().next().expect("at least 1 mesh");
    let prim = mesh.primitives().next().expect("at least 1 primitive");
    assert!(prim.get(&gltf::Semantic::Positions).is_some());
    assert!(prim.get(&gltf::Semantic::Normals).is_some());
    assert!(prim.get(&gltf::Semantic::TexCoords(0)).is_some());
    assert!(prim.get(&gltf::Semantic::Colors(0)).is_some());
    assert!(prim.get(&gltf::Semantic::Joints(0)).is_some());
    assert!(prim.get(&gltf::Semantic::Weights(0)).is_some());
    assert_eq!(prim.morph_targets().count(), 2);

    let skin = document.skins().next().expect("skin present");
    assert_eq!(skin.joints().count(), 2);

    // Verify mascot_manifest.json
    let manifest = MascotManifest::load(&report.manifest_path).expect("load & validate manifest");
    assert_eq!(manifest.avatar_name, "SampleAvatar");
    assert_eq!(
        manifest.humanoid_bones.get("Hips").map(String::as_str),
        Some("Armature/Hips")
    );
    assert_eq!(
        manifest.humanoid_bones.get("Head").map(String::as_str),
        Some("Armature/Hips/Head")
    );
    assert_eq!(
        manifest
            .viseme_map
            .get(VisemeKey::Aa)
            .map(|b| (b.mesh.as_str(), b.blendshape.as_str())),
        Some(("Body", "vrc.v_aa"))
    );

    // PhysBone transplanted onto Armature/Hips/Head/Ribbon_L and referencing col_Head_0!
    assert_eq!(manifest.colliders.len(), 1);
    assert_eq!(manifest.colliders[0].parent_node, "Armature/Hips/Head");
    assert_eq!(manifest.phys_bones.len(), 1);
    assert_eq!(
        manifest.phys_bones[0].root_node,
        "Armature/Hips/Head/Ribbon_L"
    );
    assert_eq!(manifest.phys_bones[0].limit_type, PhysBoneLimitType::Angle);
    assert_eq!(
        manifest.phys_bones[0].collider_refs,
        vec![manifest.colliders[0].id.clone()]
    );

    // Expressions: Base + ModularAvatar (Parameters deduped, MenuInstaller, MergeAnimator)
    assert_eq!(manifest.expressions.parameters.len(), 2);
    let ribbon_param = manifest
        .expressions
        .parameters
        .iter()
        .find(|p| p.name == "RibbonGlow")
        .expect("RibbonGlow parameter");
    assert_eq!(ribbon_param.param_type, ExpressionParamType::Float);
    assert!((ribbon_param.default - 0.75).abs() < 1e-5);

    assert_eq!(manifest.expressions.menu.len(), 2);
    assert!(manifest
        .expressions
        .menu
        .iter()
        .any(|m| m.name == "Ribbon Toggle" && m.item_type == ExpressionMenuItemType::Toggle));

    assert_eq!(manifest.expressions.fx_layers.len(), 2);
    let face_layer = &manifest.expressions.fx_layers[0];
    assert_eq!(face_layer.name, "FaceLayer");
    assert_eq!(face_layer.transitions.len(), 1);
    assert_eq!(face_layer.transitions[0].from_state, None);
    assert_eq!(face_layer.transitions[0].to_state, "Smile");
    assert_eq!(
        face_layer.transitions[0].conditions[0].mode,
        FxConditionMode::Equals
    );
    let smile_state = face_layer
        .states
        .iter()
        .find(|s| s.name == "Smile")
        .expect("Smile state");
    assert!(smile_state
        .curves
        .iter()
        .any(|c| matches!(&c.property, CurveProperty::BlendShape(b) if b == "eye_smile")));
    assert!(smile_state
        .curves
        .iter()
        .any(|c| matches!(&c.property, CurveProperty::GameObjectActive)));

    // Material: lilToon mapped
    assert_eq!(manifest.materials.len(), 1);
    let mat = &manifest.materials[0];
    assert_eq!(mat.shader_family, ShaderFamily::LilToon);
    assert_eq!(mat.alpha_mode, AlphaMode::Opaque);
    assert!(mat.params.rim_enable);
    assert!(mat.params.outline_enable);
    assert!(mat.params.fur_enable);
    assert_eq!(mat.params.fur_layers, 6);
    assert!(mat.params.audiolink_enable);
}

#[test]
fn material_parsing_supports_poiyomi_uts_and_standard() {
    let tmp = tempdir().expect("tempdir");
    let proj = tmp.path().join("Proj");
    fs::create_dir_all(proj.join("Assets/Materials")).unwrap();
    fs::create_dir_all(proj.join("Assets/Shaders")).unwrap();

    let poiyomi_shader = proj.join("Assets/Shaders/PoiyomiToon.shader");
    write_file(&poiyomi_shader, "Shader \".poiyomi/Toon\" {}");
    write_meta(&poiyomi_shader, "10000000000000000000000000000001");

    let uts_shader = proj.join("Assets/Shaders/UnityChanToonShader.shader");
    write_file(
        &uts_shader,
        "Shader \"UnityChanToonShader/Toon_DoubleShadeWithFeather\" {}",
    );
    write_meta(&uts_shader, "10000000000000000000000000000002");

    let poiyomi_mat = proj.join("Assets/Materials/Poi.mat");
    write_file(
        &poiyomi_mat,
        r#"%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!21 &2100000
Material:
  m_Name: PoiMat
  m_Shader: {fileID: 4800000, guid: 10000000000000000000000000000001, type: 3}
  m_CustomRenderQueue: 2450
  m_SavedProperties:
    m_Floats:
    - _EnableRimLighting: 1
    - _EnableAudioLink: 1
    - _EnableOutline: 1
    - _OutlineWidth: 0.005
    - _Cutoff: 0.35
    - _Mode: 1
    m_Colors:
    - _Color: {r: 0.9, g: 0.8, b: 0.7, a: 1}
    - _RimLightColor: {r: 0.2, g: 0.6, b: 1.0, a: 1}
"#,
    );

    let uts_mat = proj.join("Assets/Materials/Uts.mat");
    write_file(
        &uts_mat,
        r#"%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!21 &2100000
Material:
  m_Name: UtsMat
  m_Shader: {fileID: 4800000, guid: 10000000000000000000000000000002, type: 3}
  m_SavedProperties:
    m_Floats:
    - _BaseColor_Step: 0.6
    - _BaseShade_Feather: 0.05
    - _Outline_Width: 0.004
    m_Colors:
    - _BaseColor: {r: 1, g: 1, b: 1, a: 1}
    - _1st_ShadeColor: {r: 0.7, g: 0.7, b: 0.8, a: 1}
    - _2nd_ShadeColor: {r: 0.5, g: 0.5, b: 0.6, a: 1}
"#,
    );

    let std_mat = proj.join("Assets/Materials/Std.mat");
    write_file(
        &std_mat,
        r#"%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!21 &2100000
Material:
  m_Name: StdMat
  m_SavedProperties:
    m_Floats:
    - _Mode: 3
    m_Colors:
    - _Color: {r: 0.5, g: 0.5, b: 0.5, a: 0.5}
"#,
    );

    let index = vrc_extractor::indexer::AssetIndex::build(&proj).unwrap();
    let poi = vrc_extractor::yaml_parser::parse_material_asset(&poiyomi_mat, 0, &index).unwrap();
    assert_eq!(poi.def.shader_family, ShaderFamily::Poiyomi);
    assert_eq!(poi.def.alpha_mode, AlphaMode::Mask);
    assert!(poi.def.params.rim_enable);
    assert!(poi.def.params.audiolink_enable);
    assert!((poi.def.params.alpha_cutoff - 0.35).abs() < 1e-5);

    let uts = vrc_extractor::yaml_parser::parse_material_asset(&uts_mat, 1, &index).unwrap();
    assert_eq!(uts.def.shader_family, ShaderFamily::Uts);
    assert!((uts.def.params.shadow_border - 0.6).abs() < 1e-5);
    assert!(uts.def.params.outline_enable);

    let std = vrc_extractor::yaml_parser::parse_material_asset(&std_mat, 2, &index).unwrap();
    assert_eq!(std.def.shader_family, ShaderFamily::Standard);
    assert_eq!(std.def.alpha_mode, AlphaMode::Blend);
}

#[test]
fn unity_yaml_handles_bom_negative_and_large_file_ids_and_empty_blocks() {
    let raw = "\u{feff}%YAML 1.1\n%TAG !u! tag:unity3d.com,2011:\n--- !u!1 &-9876543210123\nGameObject:\n  m_Name: NegativeId\n  m_IsActive: 1\n--- !u!4 &919132149155446097 stripped\nTransform:\n  m_CorrespondingSourceObject: {fileID: 400000, guid: abcdef0123456789abcdef0123456789, type: 3}\n  m_PrefabInstance: {fileID: 1001}\n--- !u!114 &42\n";
    let parsed = vrc_extractor::indexer::UnityYamlFile::parse_str(raw, Path::new("test.unity"))
        .expect("parse tricky Unity YAML");
    assert_eq!(parsed.documents.len(), 3);
    assert_eq!(parsed.get(-9876543210123).unwrap().root_key, "GameObject");
    let stripped = parsed.resolve_stripped(919132149155446097).unwrap();
    assert_eq!(stripped.prefab_instance_file_id, 1001);
    assert_eq!(stripped.corresponding_source.file_id, 400000);
}

#[test]
fn asset_index_reverse_lookup_and_packages_scan() {
    let tmp = tempdir().expect("tempdir");
    let proj = tmp.path().join("Proj");
    let a1 = proj.join("Assets/Nested/Foo.mat");
    let p1 = proj.join("Packages/com.example/Bar.shader");
    write_file(&a1, "Material: {}");
    write_meta(&a1, "abcdef0123456789abcdef0123456780");
    write_file(&p1, "Shader \"Custom/lilToon\" {}");
    write_meta(&p1, "abcdef0123456789abcdef0123456781");

    let idx = vrc_extractor::indexer::AssetIndex::build(&proj).expect("build AssetIndex");
    assert_eq!(idx.len(), 2);
    assert_eq!(
        idx.resolve_guid("ABCDEF0123456789ABCDEF0123456780")
            .map(|p| p.ends_with("Foo.mat")),
        Some(true)
    );
    assert_eq!(
        idx.guid_for_relative_path("Assets\\Nested\\Foo.mat"),
        Some("abcdef0123456789abcdef0123456780")
    );
    assert_eq!(
        idx.shader_family_by_guid("abcdef0123456789abcdef0123456781"),
        Some(ShaderFamily::LilToon)
    );
}

#[test]
fn prefab_array_modification_reparent_and_cycle_detection() {
    let tmp = tempdir().expect("tempdir");
    let proj = tmp.path().join("Proj");

    let child_guid = "a1000000000000000000000000000001";
    let child_prefab = proj.join("Assets/Child.prefab");
    write_file(
        &child_prefab,
        r#"%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!1 &10
GameObject:
  m_Name: ChildObj
  m_IsActive: 1
  m_Component:
  - component: {fileID: 11}
  - component: {fileID: 12}
--- !u!4 &11
Transform:
  m_GameObject: {fileID: 10}
  m_LocalPosition: {x: 0, y: 0, z: 0}
  m_LocalRotation: {x: 0, y: 0, z: 0, w: 1}
  m_LocalScale: {x: 1, y: 1, z: 1}
  m_Children: []
  m_Father: {fileID: 0}
--- !u!23 &12
MeshRenderer:
  m_GameObject: {fileID: 10}
  m_Materials:
  - {fileID: 2100000, guid: 00000000000000000000000000000001, type: 3}
"#,
    );
    write_meta(&child_prefab, child_guid);

    let scene_path = proj.join("Assets/Test.unity");
    write_file(
        &scene_path,
        &format!(
            r#"%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!1 &1
GameObject:
  m_Name: SceneParent
  m_IsActive: 1
  m_Component:
  - component: {{fileID: 2}}
--- !u!4 &2
Transform:
  m_GameObject: {{fileID: 1}}
  m_LocalPosition: {{x: 0, y: 0, z: 0}}
  m_LocalRotation: {{x: 0, y: 0, z: 0, w: 1}}
  m_LocalScale: {{x: 1, y: 1, z: 1}}
  m_Children: []
  m_Father: {{fileID: 0}}
--- !u!1001 &100
PrefabInstance:
  m_Modification:
    m_TransformParent: {{fileID: 2}}
    m_Modifications:
    - target: {{fileID: 12, guid: {child_guid}, type: 3}}
      propertyPath: m_Materials.Array.data[0]
      value:
      objectReference: {{fileID: 2100000, guid: 00000000000000000000000000000099, type: 3}}
  m_SourcePrefab: {{fileID: 100100000, guid: {child_guid}, type: 3}}
"#
        ),
    );

    let index = vrc_extractor::indexer::AssetIndex::build(&proj).unwrap();
    let (vscene, _) =
        vrc_extractor::yaml_parser::VirtualScene::load_and_resolve(&scene_path, &index).unwrap();
    let hierarchy = vrc_extractor::yaml_parser::SceneHierarchy::from_virtual_scene(&vscene);

    let parent_id = hierarchy.find_by_name("SceneParent").unwrap();
    let child_id = hierarchy.find_by_name("ChildObj").unwrap();
    assert_eq!(hierarchy.nodes[&child_id].parent, Some(parent_id));
    let mr = &hierarchy.nodes[&child_id].components[0];
    let mat_ref = vrc_extractor::indexer::UnityRef::from_yaml(&mr.data["m_Materials"][0]).unwrap();
    assert_eq!(
        mat_ref.guid.as_deref(),
        Some("00000000000000000000000000000099")
    );

    // Cycle detection test
    let cyc_a_guid = "c1000000000000000000000000000001";
    let cyc_b_guid = "c1000000000000000000000000000002";
    let cyc_a = proj.join("Assets/CycA.prefab");
    let cyc_b = proj.join("Assets/CycB.prefab");
    write_file(
        &cyc_a,
        &format!(
            "--- !u!1001 &1\nPrefabInstance:\n  m_SourcePrefab: {{fileID: 1, guid: {cyc_b_guid}, type: 3}}\n"
        ),
    );
    write_meta(&cyc_a, cyc_a_guid);
    write_file(
        &cyc_b,
        &format!(
            "--- !u!1001 &1\nPrefabInstance:\n  m_SourcePrefab: {{fileID: 1, guid: {cyc_a_guid}, type: 3}}\n"
        ),
    );
    write_meta(&cyc_b, cyc_b_guid);
    let idx2 = vrc_extractor::indexer::AssetIndex::build(&proj).unwrap();
    let res = vrc_extractor::yaml_parser::VirtualScene::load_and_resolve(&cyc_a, &idx2);
    assert!(matches!(
        res,
        Err(vrc_extractor::ExtractorError::PrefabResolution { .. })
    ));
}

#[test]
fn animator_all_condition_modes_and_material_curves() {
    let tmp = tempdir().expect("tempdir");
    let proj = tmp.path().join("Proj");
    let anim_guid = "d1000000000000000000000000000001";
    let anim_path = proj.join("Assets/Glow.anim");
    write_file(
        &anim_path,
        r#"%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!74 &7400000
AnimationClip:
  m_Name: Glow
  m_FloatCurves:
  - curve:
      m_Curve:
      - time: 0
        value: 0
      - time: 0.5
        value: 1
    attribute: material._EmissionStrength
    path: Body
"#,
    );
    write_meta(&anim_path, anim_guid);

    let ctrl_path = proj.join("Assets/AllModes.controller");
    write_file(
        &ctrl_path,
        &format!(
            r#"%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!91 &9100000
AnimatorController:
  m_Name: AllModes
  m_AnimatorParameters:
  - m_Name: FlagA
    m_Type: 4
    m_DefaultBool: 0
  - m_Name: ValB
    m_Type: 1
    m_DefaultFloat: 0
  m_AnimatorLayers:
  - m_Name: Base
    m_StateMachine: {{fileID: 11070001}}
    m_DefaultWeight: 1
--- !u!1107 &11070001
AnimatorStateMachine:
  m_Name: Base
  m_States:
  - state: {{fileID: 11020001}}
  - state: {{fileID: 11020002}}
  m_DefaultState: {{fileID: 11020001}}
--- !u!1102 &11020001
AnimatorState:
  m_Name: Off
  m_Transitions:
  - {{fileID: 11010001}}
--- !u!1102 &11020002
AnimatorState:
  m_Name: On
  m_Motion: {{fileID: 7400000, guid: {anim_guid}, type: 2}}
  m_Transitions:
  - {{fileID: 11010002}}
--- !u!1101 &11010001
AnimatorStateTransition:
  m_DstState: {{fileID: 11020002}}
  m_HasExitTime: 1
  m_ExitTime: 0.9
  m_TransitionDuration: 0.2
  m_Conditions:
  - m_ConditionMode: 1
    m_ConditionEvent: FlagA
    m_EventTreshold: 0
  - m_ConditionMode: 3
    m_ConditionEvent: ValB
    m_EventTreshold: 0.5
  - m_ConditionMode: 7
    m_ConditionEvent: ValB
    m_EventTreshold: 2
--- !u!1101 &11010002
AnimatorStateTransition:
  m_DstState: {{fileID: 11020001}}
  m_HasExitTime: 0
  m_Conditions:
  - m_ConditionMode: 2
    m_ConditionEvent: FlagA
    m_EventTreshold: 0
  - m_ConditionMode: 4
    m_ConditionEvent: ValB
    m_EventTreshold: 0.1
"#
        ),
    );

    let index = vrc_extractor::indexer::AssetIndex::build(&proj).unwrap();
    let mut warnings = Vec::new();
    let parsed =
        vrc_extractor::yaml_parser::parse_animator_controller(&ctrl_path, &index, &mut warnings)
            .unwrap();
    assert_eq!(parsed.layers.len(), 1);
    let layer = &parsed.layers[0];
    assert_eq!(layer.transitions.len(), 2);
    assert!(layer.transitions[0].has_exit_time);
    assert_eq!(layer.transitions[0].conditions[0].mode, FxConditionMode::If);
    assert_eq!(
        layer.transitions[0].conditions[1].mode,
        FxConditionMode::Greater
    );
    assert_eq!(
        layer.transitions[0].conditions[2].mode,
        FxConditionMode::NotEqual
    );
    assert_eq!(
        layer.transitions[1].conditions[0].mode,
        FxConditionMode::IfNot
    );
    assert_eq!(
        layer.transitions[1].conditions[1].mode,
        FxConditionMode::Less
    );
    let on_state = layer.states.iter().find(|s| s.name == "On").unwrap();
    assert!(matches!(
        &on_state.curves[0].property,
        CurveProperty::MaterialFloat(p) if p == "_EmissionStrength"
    ));
}

#[test]
fn modular_avatar_disabled_skips_ma_merging_and_menu_asset_installer_works() {
    let tmp = tempdir().expect("tempdir");
    let proj = tmp.path().join("Proj");

    let ma_menu_inst_guid = "e1000000000000000000000000000001";
    let ma_inst_cs = proj.join("Packages/ma/ModularAvatarMenuInstaller.cs");
    write_file(&ma_inst_cs, "// stub");
    write_meta(&ma_inst_cs, ma_menu_inst_guid);

    let sub_menu_guid = "e1000000000000000000000000000002";
    let sub_menu_path = proj.join("Assets/SubMenu.asset");
    write_file(
        &sub_menu_path,
        r#"%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!114 &11400000
MonoBehaviour:
  m_Name: SubMenu
  controls:
  - name: Jacket
    type: 102
    parameter:
      name: JacketOn
    value: 1
"#,
    );
    write_meta(&sub_menu_path, sub_menu_guid);

    let append_menu_guid = "e1000000000000000000000000000003";
    let append_menu_path = proj.join("Assets/AppendMenu.asset");
    write_file(
        &append_menu_path,
        &format!(
            r#"%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!114 &11400000
MonoBehaviour:
  m_Name: AppendMenu
  controls:
  - name: Clothes
    type: 103
    subMenu: {{fileID: 11400000, guid: {sub_menu_guid}, type: 3}}
"#
        ),
    );
    write_meta(&append_menu_path, append_menu_guid);

    let scene_rel = Path::new("Assets/Scene.unity");
    let scene_path = proj.join(scene_rel);
    write_file(
        &scene_path,
        &format!(
            r#"%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!1 &1
GameObject:
  m_Name: TestAvatar
  m_IsActive: 1
  m_Component:
  - component: {{fileID: 2}}
  - component: {{fileID: 3}}
--- !u!4 &2
Transform:
  m_GameObject: {{fileID: 1}}
  m_LocalPosition: {{x: 0, y: 0, z: 0}}
  m_LocalRotation: {{x: 0, y: 0, z: 0, w: 1}}
  m_LocalScale: {{x: 1, y: 1, z: 1}}
  m_Children: []
  m_Father: {{fileID: 0}}
--- !u!114 &3
MonoBehaviour:
  m_GameObject: {{fileID: 1}}
  m_Script: {{fileID: 11500000, guid: {ma_menu_inst_guid}, type: 3}}
  menuToAppend: {{fileID: 11400000, guid: {append_menu_guid}, type: 3}}
"#
        ),
    );

    // 1. With modular_avatar_enabled = false -> menu stays empty
    let out_off = tmp.path().join("out_off");
    let report_off = extract_avatar(
        &ExtractOptions {
            project_dir: proj.clone(),
            scene_path: scene_rel.to_path_buf(),
            avatar_name: "TestAvatar".to_owned(),
            output_dir: out_off,
            modular_avatar_enabled: false,
            max_texture_size: 2048,
        },
        |_| {},
    )
    .unwrap();
    let m_off = MascotManifest::load(&report_off.manifest_path).unwrap();
    assert!(m_off.expressions.menu.is_empty());

    // 2. With modular_avatar_enabled = true -> SubMenu + nested Jacket item installed
    let out_on = tmp.path().join("out_on");
    let report_on = extract_avatar(
        &ExtractOptions {
            project_dir: proj,
            scene_path: scene_rel.to_path_buf(),
            avatar_name: "TestAvatar".to_owned(),
            output_dir: out_on,
            modular_avatar_enabled: true,
            max_texture_size: 2048,
        },
        |_| {},
    )
    .unwrap();
    let m_on = MascotManifest::load(&report_on.manifest_path).unwrap();
    assert_eq!(m_on.expressions.menu.len(), 1);
    assert_eq!(m_on.expressions.menu[0].name, "Clothes");
    assert_eq!(
        m_on.expressions.menu[0].item_type,
        ExpressionMenuItemType::SubMenu
    );
    assert_eq!(m_on.expressions.menu[0].sub_menu.len(), 1);
    assert_eq!(m_on.expressions.menu[0].sub_menu[0].name, "Jacket");
}
