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
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) world_pos: vec3<f32>,
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

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let skin_mat = compute_skin_matrix(in.joints_0, in.weights_0);
    let world_pos4 = skin_mat * vec4<f32>(in.position, 1.0);

    var out: VertexOutput;
    out.clip_position = frame.view_proj * world_pos4;
    out.uv = in.texcoord_0;
    out.color = in.color_0;
    out.world_pos = world_pos4.xyz;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let base_sample = textureSample(base_tex, base_sampler, in.uv);
    let alpha_mode = material.mode_flags.x;
    let raw_alpha = clamp(base_sample.a * material.main_color.a * in.color.a, 0.0, 1.0);
    let effective_alpha = select(1.0, raw_alpha, alpha_mode > 0.5);

    if (alpha_mode > 0.5 && alpha_mode < 1.5 && raw_alpha < material.matcap_emission.w) {
        discard;
    }
    if (effective_alpha <= 0.05) {
        discard;
    }

    var flags = draw.part_flags & 0xFFu;
    if (frame.head_sphere.w > 0.0 && distance(in.world_pos, frame.head_sphere.xyz) <= frame.head_sphere.w) {
        flags = flags | 1u;
    }

    // Encode (alpha_byte, node_id_low, node_id_high, part_flags) into Rgba8Unorm.
    // node_id = draw.node_index + 1 so 0 remains reserved for background.
    let encoded_id = draw.node_index + 1u;
    let id_low = f32(encoded_id & 0xFFu) / 255.0;
    let id_high = f32((encoded_id >> 8u) & 0xFFu) / 255.0;
    let part_flags_byte = f32(flags) / 255.0;

    return vec4<f32>(effective_alpha, id_low, id_high, part_flags_byte);
}
