struct FrameUniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    light_dir: vec4<f32>,
    light_color: vec4<f32>,
    audiolink_bands: vec4<f32>,
    audiolink_meta: vec4<f32>,
    head_sphere: vec4<f32>,
    audiolink_history: array<vec4<f32>, 32>,
};

struct DrawUniforms {
    node_world: mat4x4<f32>,
    node_index: u32,
    is_skinned: u32,
    part_flags: u32,
    fur_layers: u32,
};

struct MaterialUniforms {
    main_color: vec4<f32>,
    shadow_color_1: vec4<f32>,
    shadow_color_2: vec4<f32>,
    rim_color: vec4<f32>,
    matcap_color: vec4<f32>,
    emission_color: vec4<f32>,
    outline_color: vec4<f32>,
    shadow_params: vec4<f32>,
    rim_params: vec4<f32>,
    matcap_emission: vec4<f32>,
    outline_fur: vec4<f32>,
    audiolink_flags: vec4<f32>,
    mode_flags: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;

@group(1) @binding(0) var<storage, read> joint_matrices: array<mat4x4<f32>>;
@group(1) @binding(1) var<uniform> draw: DrawUniforms;

@group(2) @binding(0) var<uniform> material: MaterialUniforms;
@group(2) @binding(1) var base_tex: texture_2d<f32>;
@group(2) @binding(2) var base_sampler: sampler;
@group(2) @binding(3) var matcap_tex: texture_2d<f32>;
@group(2) @binding(4) var emission_tex: texture_2d<f32>;
@group(2) @binding(5) var fur_mask_tex: texture_2d<f32>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) texcoord_0: vec2<f32>,
    @location(4) color_0: vec4<f32>,
    @location(5) joints_0: vec4<u32>,
    @location(6) weights_0: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
};

fn compute_skin_matrix(joints: vec4<u32>, weights: vec4<f32>) -> mat4x4<f32> {
    if (draw.is_skinned == 0u) {
        return draw.node_world;
    }
    let w_sum = weights.x + weights.y + weights.z + weights.w;
    if (w_sum <= 0.0001) {
        return draw.node_world;
    }
    let count = arrayLength(&joint_matrices);
    let j0 = min(joints.x, count - 1u);
    let j1 = min(joints.y, count - 1u);
    let j2 = min(joints.z, count - 1u);
    let j3 = min(joints.w, count - 1u);
    let inv = 1.0 / w_sum;
    return joint_matrices[j0] * (weights.x * inv)
        + joint_matrices[j1] * (weights.y * inv)
        + joint_matrices[j2] * (weights.z * inv)
        + joint_matrices[j3] * (weights.w * inv);
}

fn select_audiolink_band(bands: vec4<f32>, band_idx: u32) -> f32 {
    switch (band_idx) {
        case 0u: { return bands.x; }
        case 1u: { return bands.y; }
        case 2u: { return bands.z; }
        default: { return bands.w; }
    }
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let skin_mat = compute_skin_matrix(in.joints_0, in.weights_0);
    let world_pos4 = skin_mat * vec4<f32>(in.position, 1.0);
    let world_n = normalize((skin_mat * vec4<f32>(in.normal, 0.0)).xyz);

    var out: VertexOutput;
    out.clip_position = frame.view_proj * world_pos4;
    out.world_pos = world_pos4.xyz;
    out.world_normal = world_n;
    out.uv = in.texcoord_0;
    out.color = in.color_0;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample all textures in uniform control flow before any discard.
    let base_sample = textureSample(base_tex, base_sampler, in.uv);
    let emission_sample = textureSample(emission_tex, base_sampler, in.uv);

    let n = normalize(in.world_normal);
    let v = normalize(frame.camera_pos.xyz - in.world_pos);
    let l = normalize(frame.light_dir.xyz);

    // View-space-like normal for MatCap UV
    let up_ref = vec3<f32>(0.0, 1.0, 0.0);
    let raw_right = cross(up_ref, v);
    let cam_right = select(vec3<f32>(1.0, 0.0, 0.0), normalize(raw_right), dot(raw_right, raw_right) > 1e-6);
    let cam_up = cross(v, cam_right);
    let matcap_uv = clamp(
        vec2<f32>(dot(n, cam_right) * 0.5 + 0.5, -dot(n, cam_up) * 0.5 + 0.5),
        vec2<f32>(0.001, 0.001),
        vec2<f32>(0.999, 0.999)
    );
    let matcap_sample = textureSample(matcap_tex, base_sampler, matcap_uv);

    let albedo = base_sample * material.main_color * in.color;
    let alpha_mode = material.mode_flags.x;
    let alpha_cutoff = material.matcap_emission.w;
    if (alpha_mode > 0.5 && alpha_mode < 1.5 && albedo.a < alpha_cutoff) {
        discard;
    }

    // Wrapped Half-Lambert diffuse for Toon 1st/2nd shadow ramp (lilToon / Poiyomi / UTS / Standard)
    let ndotl = dot(n, l) * 0.5 + 0.5;
    let border1 = material.shadow_params.x;
    let blur1 = max(material.shadow_params.y, 0.001);
    let border2 = material.shadow_params.z;
    let blur2 = max(material.shadow_params.w, 0.001);

    let shadow1_factor = smoothstep(border1 - blur1, border1 + blur1, ndotl);
    let shadow2_factor = smoothstep(border2 - blur2, border2 + blur2, ndotl);

    let family = material.audiolink_flags.w;
    let shadow1_tint = mix(albedo.rgb * material.shadow_color_1.rgb, material.shadow_color_1.rgb * albedo.rgb, 0.5);
    let shadow2_tint = albedo.rgb * material.shadow_color_2.rgb;

    var lit_rgb = mix(shadow1_tint, albedo.rgb, shadow1_factor);
    if (family < 2.5) {
        // lilToon / Poiyomi / UTS 2nd shadow layer
        lit_rgb = mix(shadow2_tint, lit_rgb, shadow2_factor);
    }

    // Light color & subtle ambient floor
    let light_energy = frame.light_color.rgb * frame.light_dir.w + vec3<f32>(frame.light_color.a);
    var color_rgb = lit_rgb * light_energy;

    // Rim lighting (lilToon-compatible clamped border/blur so border=0 never washes front faces)
    if (material.rim_params.x > 0.5) {
        let fresnel_power = max(material.rim_params.w, 0.1);
        let ndotv = clamp(abs(dot(n, v)), 0.0, 1.0);
        let fresnel = pow(clamp(1.0 - ndotv, 0.0, 1.0), fresnel_power);
        let rim_border = material.rim_params.y;
        let rim_blur = max(material.rim_params.z, 0.001);
        let rim_min = clamp(rim_border - rim_blur * 0.5, 0.0, 1.0);
        let rim_max = clamp(rim_border + rim_blur * 0.5, 0.0, 1.0);
        var rim_mask = 0.0;
        if (fresnel > 0.0001) {
            if (rim_max > rim_min + 1e-5) {
                rim_mask = smoothstep(rim_min, rim_max, fresnel);
            } else {
                rim_mask = step(rim_min, fresnel);
            }
        }
        let rim_shadow_atten = mix(0.5, 1.0, shadow1_factor);
        color_rgb += material.rim_color.rgb * (rim_mask * material.rim_color.a * rim_shadow_atten) * light_energy;
    }

    // MatCap blend (lilToon modes: 0=Normal, 1=Add, 2=Screen, 3=Multiply)
    if (material.matcap_emission.x > 0.5) {
        let mc_rgb = matcap_sample.rgb * material.matcap_color.rgb;
        let mc_alpha = clamp(material.matcap_color.a, 0.0, 1.0);
        let blend_mode = material.matcap_emission.y;
        if (blend_mode < 0.5) {
            color_rgb = mix(color_rgb, mc_rgb * light_energy, mc_alpha);
        } else if (blend_mode < 1.5) {
            color_rgb += mc_rgb * mc_alpha * light_energy;
        } else if (blend_mode < 2.5) {
            let mc_lit = mc_rgb * light_energy;
            let screened = max(color_rgb + mc_lit - color_rgb * mc_lit, color_rgb);
            color_rgb = mix(color_rgb, screened, mc_alpha);
        } else {
            color_rgb = mix(color_rgb, color_rgb * mc_rgb, mc_alpha);
        }
    }

    // Emission
    if (material.matcap_emission.z > 0.5) {
        color_rgb += emission_sample.rgb * material.emission_color.rgb;
    }

    // AudioLink reactive glow
    if (material.audiolink_flags.x > 0.5 && frame.audiolink_meta.x > 0.5) {
        let band_idx = u32(clamp(material.audiolink_flags.y, 0.0, 3.0));
        let band_amp = select_audiolink_band(frame.audiolink_bands, band_idx);
        let hist_prev = select_audiolink_band(frame.audiolink_history[4], band_idx);
        let pulse = clamp(mix(band_amp, hist_prev, 0.25) * material.audiolink_flags.z, 0.0, 2.0);
        let accent = max(material.emission_color.rgb, material.rim_color.rgb * 0.5);
        color_rgb += (accent + albedo.rgb * 0.35) * pulse;
    }

    // Premultiplied alpha output for Win32 DWM transparent compositing
    let out_alpha = select(1.0, clamp(albedo.a, 0.0, 1.0), alpha_mode > 1.5);
    return vec4<f32>(color_rgb * out_alpha, out_alpha);
}
